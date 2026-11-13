"""Measure an exported .onnx file exactly the way the phone page uses it.

    uv run python measure_onnx.py ../leaf_model.onnx      # the model the page runs today
    uv run python measure_onnx.py leaf_model_v2.onnx      # the one this folder trains

Prints the file size, top-1 accuracy on the validation set and on the camera-like set, the
milliseconds of one classification with **one thread** (the setting the page must use, see
its comment on numThreads), and the largest difference between the exported file and
`check_expected.json` for the check image — the same comparison the built-in check on the
page makes, so a wrong export cannot pass here.

The preprocessing is deliberately the page's preprocessing and not the notebook's
transform: central square, resize, /255, standardize, CHW. That is what the phone does.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "_dev" / "leaves_data"


def preprocess(picture: Image.Image, mean: list[float], std: list[float], size: int) -> np.ndarray:
    """The four steps of classifySource() in index.html, in numpy."""
    square = min(picture.size)
    left = (picture.width - square) // 2
    top = (picture.height - square) // 2
    picture = picture.crop((left, top, left + square, top + square)).resize((size, size), Image.BILINEAR)
    pixels = np.asarray(picture, dtype=np.float32) / 255.0
    pixels = (pixels - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return pixels.transpose(2, 0, 1)[np.newaxis]  # H,W,C -> 1,C,H,W


def predict_all(session: ort.InferenceSession, folder: Path, info: dict) -> tuple[np.ndarray, np.ndarray]:
    """Logits and true classes of every photograph in one class/…/image.jpg folder."""
    input_name = session.get_inputs()[0].name
    logits, labels = [], []
    for class_index, class_folder in enumerate(sorted(folder.iterdir())):
        for image_file in sorted(class_folder.iterdir()):
            batch = preprocess(
                Image.open(image_file).convert("RGB"), info["mean"], info["std"], info["input_size"]
            )
            logits.append(session.run(None, {input_name: batch})[0])
            labels.append(class_index)
    return np.concatenate(logits), np.array(labels)


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def softmax(logits: np.ndarray) -> np.ndarray:
    """The page's softmax, row by row."""
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def confidence_table(probabilities: np.ndarray, labels: np.ndarray) -> None:
    """How often the page is right when it says it is right - and when it is not sure at all.

    The rows are the probability of the winning class, so the table says where the page can
    stop saying "maple 34 %" as if it were a decision. It is printed for the camera-like set
    only, because that is the situation the message will appear in.
    """
    top = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    print("             how often the page is right, by its own confidence:")
    for low in (0.0, 0.4, 0.6, 0.8, 0.95):
        high = {0.0: 0.4, 0.4: 0.6, 0.6: 0.8, 0.8: 0.95, 0.95: 1.001}[low]
        rows = (top >= low) & (top < high)
        if rows.any():
            print(
                f"   confidence {low:.2f}-{min(high, 1.0):.2f}  {rows.sum():3d} photos  "
                f"right {correct[rows].mean():.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("--repeat", type=int, default=20, help="classifications to time")
    parser.add_argument("--input-size", type=int, help="feed a different size than model_info.json says")
    parser.add_argument(
        "--check-folder",
        type=Path,
        default=HERE,
        help="folder with the check_image.jpg and check_expected.json of THIS model",
    )
    arguments = parser.parse_args()

    session = ort.InferenceSession(
        str(arguments.model), providers=[("CPUExecutionProvider", {"intra_op_num_threads": 1})]
    )
    metadata = session.get_modelmeta().custom_metadata_map
    if "model_info" in metadata:
        info = json.loads(metadata["model_info"])
    else:  # a bare export, e.g. one made by hand for a quick test
        info = {
            "classes": sorted(p.name for p in (DATA / "leaves" / "train").iterdir() if p.is_dir()),
            "input_size": 96,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        print("note: no model_info metadata inside the file, using the notebook's mean/std and 96x96")
    classes = info["classes"]
    if arguments.input_size:
        info = {**info, "input_size": arguments.input_size}
    print(
        f"{arguments.model.name}: {arguments.model.stat().st_size / 1e6:.2f} MB, "
        f"input {info['input_size']}, classes {classes}"
    )

    logits, labels = predict_all(session, DATA / "leaves" / "val", info)
    print(f"validation   {accuracy(logits, labels):.3f}  ({len(labels)} photos)")
    logits, labels = predict_all(session, DATA / "leaves_cam", info)
    print(f"camera-like  {accuracy(logits, labels):.3f}  ({len(labels)} photos)")
    confidence_table(softmax(logits), labels)

    check_image = arguments.check_folder / "check_image.jpg"
    picture_file = check_image if check_image.exists() else min((DATA / "leaves" / "val").rglob("*.jpg"))
    batch = preprocess(Image.open(picture_file).convert("RGB"), info["mean"], info["std"], info["input_size"])
    input_name = session.get_inputs()[0].name

    session.run(None, {input_name: batch})  # one run to warm the caches
    started = time.perf_counter()
    for _ in range(arguments.repeat):
        session.run(None, {input_name: batch})
    print(
        f"one thread   {(time.perf_counter() - started) / arguments.repeat * 1000:.1f} ms per classification"
    )

    expected_file = arguments.check_folder / "check_expected.json"
    if not (check_image.exists() and expected_file.exists()):
        print(f"check image  skipped, no check files for this model in {arguments.check_folder}")
        return
    got = session.run(None, {input_name: batch})[0][0]
    probabilities = np.exp(got - got.max()) / np.exp(got - got.max()).sum()  # the page's softmax
    expected = json.loads(expected_file.read_text(encoding="utf-8"))
    worst = max(abs(probabilities[classes.index(name)] - value) for name, value in expected.items())
    print(f"check image  {dict(zip(classes, probabilities.round(6), strict=True))}")
    print(f"             largest difference from check_expected.json: {worst:.2e}")


if __name__ == "__main__":
    main()
