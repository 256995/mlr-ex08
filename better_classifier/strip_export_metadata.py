"""Delete the exporter's per-node stack traces from an exported `.onnx`, in place.

torch 2.15's ONNX export attaches a `pkg.torch.onnx.stack_trace` metadata entry to every
node of the graph, and those Python tracebacks quote absolute paths of the machine that ran
the export: the username, the project folder, the interpreter inside `.venv`. A model file
published in a public repository should not carry them, and nothing needs them — the runtime
never reads node metadata.

Run it after `train_leaves.py --export`:

    uv run python -I -X utf8 train_leaves.py --variant full --seed 0 --export
    uv run python -I -X utf8 strip_export_metadata.py leaf_model_v2.onnx

The script refuses to write anything unless it has proved the file still holds the same
network: same nodes in the same order with the same op types, inputs, outputs and
attributes; same initializers with byte-identical weights; same graph inputs and outputs;
the same ir_version and opsets; `onnx.checker` passes; and bit-for-bit identical logits from
onnxruntime on `check_image.jpg`, on the page's own preprocessing, one thread.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import checker
from onnx.onnx_pb import AttributeProto, ModelProto, NodeProto, StringStringEntryProto

HERE = Path(__file__).resolve().parent

# Exporter bookkeeping, by name, plus a net for anything else that quotes an absolute path.
DEBUG_KEYS = {"pkg.torch.onnx.stack_trace"}
ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/](?:Users|home)[\\/]|/home/)", re.IGNORECASE)


def holds_local_path(entry: StringStringEntryProto) -> bool:
    return entry.key in DEBUG_KEYS or bool(ABSOLUTE_PATH.search(entry.value))


def node_signature(node: NodeProto) -> tuple:
    """Everything about a node except its metadata — what the runtime actually executes."""
    attributes = tuple(
        (
            attribute.name,
            attribute.type,
            attribute.f,
            attribute.i,
            attribute.s,
            tuple(attribute.floats),
            tuple(attribute.ints),
            attribute.t.SerializeToString(deterministic=True)
            if attribute.type == AttributeProto.TENSOR
            else b"",
        )
        for attribute in sorted(node.attribute, key=lambda a: a.name)
    )
    return (node.name, node.op_type, node.domain, tuple(node.input), tuple(node.output), attributes)


def structure_problems(before: ModelProto, after: ModelProto) -> list[str]:
    """Name every difference in the executable part of the two models; empty means identical."""
    problems = []
    if [node_signature(n) for n in before.graph.node] != [node_signature(n) for n in after.graph.node]:
        problems.append("a node's name, op type, inputs, outputs or attributes changed")
    if [i.name for i in before.graph.initializer] != [i.name for i in after.graph.initializer]:
        problems.append("the initializer list changed")
    elif not all(
        a.name == b.name and a.raw_data == b.raw_data
        for a, b in zip(before.graph.initializer, after.graph.initializer, strict=False)
    ):
        problems.append("a weight tensor changed")
    for kind in ("input", "output"):
        if [x.name for x in getattr(before.graph, kind)] != [x.name for x in getattr(after.graph, kind)]:
            problems.append(f"the graph {kind}s changed")
    if before.ir_version != after.ir_version:
        problems.append("ir_version changed")
    if [(o.domain, o.version) for o in before.opset_import] != [
        (o.domain, o.version) for o in after.opset_import
    ]:
        problems.append("the opset imports changed")
    if list(before.metadata_props) != list(after.metadata_props):
        problems.append("the model-level metadata changed")
    return problems


def logits(model_path: Path) -> np.ndarray:
    """Run the model the way index.html does: the page's own preprocessing, one thread."""
    info = json.loads((HERE / "model_info.json").read_text(encoding="utf-8"))
    size, mean, std = info["input_size"], info["mean"], info["std"]
    from PIL import Image

    picture = Image.open(HERE / "check_image.jpg").convert("RGB").resize((size, size))
    pixels = np.asarray(picture, dtype=np.float32) / 255.0
    array = ((pixels - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)).transpose(
        2, 0, 1
    )[None]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    return np.asarray(session.run(None, {session.get_inputs()[0].name: array})[0]).ravel()


def strip(model: ModelProto) -> int:
    """Drop the entries that quote the exporting machine; return how many went."""
    dropped = 0
    for container in [model.graph, *model.graph.node]:
        kept = [entry for entry in container.metadata_props if not holds_local_path(entry)]
        dropped += len(container.metadata_props) - len(kept)
        if len(kept) != len(container.metadata_props):
            del container.metadata_props[:]
            container.metadata_props.extend(kept)
    return dropped


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "leaf_model_v2.onnx"
    before = onnx.load(source)
    found = len(ABSOLUTE_PATH.findall(source.read_bytes().decode("latin-1")))
    if found == 0:
        print(f"{source.name}: no absolute path of any machine in the file, nothing to do")
        return

    nodes = sum(1 for n in before.graph.node if any(holds_local_path(e) for e in n.metadata_props))
    print(
        f"{source.name}: {found} absolute paths, {nodes} of {len(before.graph.node)} nodes affected, "
        f"{source.stat().st_size} bytes"
    )
    reference = logits(source)

    after = onnx.load(source)
    dropped = strip(after)
    blob = after.SerializeToString(deterministic=True)

    problems = structure_problems(before, after)
    if dropped == 0:
        problems.append("nothing was dropped, yet the paths were in the file")
    if ABSOLUTE_PATH.search(blob.decode("latin-1")):
        problems.append("an absolute path is still in the serialised model")
    try:
        checker.check_model(onnx.load_model_from_string(blob))
    except Exception as error:  # onnx prints a specific error per model; report it verbatim
        problems.append(f"onnx.checker rejected the result: {error}")

    temporary = source.with_name(source.stem + ".stripped" + source.suffix)
    temporary.write_bytes(blob)
    try:
        cleaned = logits(temporary)
    finally:
        if problems or not np.array_equal(reference, cleaned):
            temporary.unlink(missing_ok=True)
    if not np.array_equal(reference, cleaned):
        problems.append(f"logits differ: {reference.tolist()} vs {cleaned.tolist()}")

    if problems:
        for problem in problems:
            print("REFUSING to write:", problem)
        raise SystemExit(1)

    temporary.replace(source)
    print(
        f"dropped {dropped} metadata entries; identical graph, identical logits; "
        f"{source.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
