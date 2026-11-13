"""Try the cheapest phone-side improvement: 8-bit weights instead of 32-bit ones.

    uv run python quantize_model.py leaf_model_v2.onnx

Writes ``<name>_int8.onnx`` next to the input. The phone downloads the model over the
network before it can classify anything, so half the bytes is half the waiting; what the
8-bit weights cost in accuracy is what measure_onnx.py then shows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    arguments = parser.parse_args()

    out = arguments.model.with_name(arguments.model.stem + "_int8.onnx")
    quantize_dynamic(arguments.model, out, weight_type=QuantType.QUInt8)
    print(
        f"{arguments.model.name}: {arguments.model.stat().st_size / 1e6:.2f} MB"
        f"  ->  {out.name}: {out.stat().st_size / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()
