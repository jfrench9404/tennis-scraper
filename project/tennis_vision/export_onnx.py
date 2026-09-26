"""Export local YOLO checkpoints to ONNX for GPU inference via ONNX Runtime DirectML.

This is a format conversion of the SAME weights, not a different model. GPU
execution changes floating-point results slightly, so detections are close to
but not byte-identical with PyTorch CPU. Provenance (source checkpoint hash,
export settings, library versions, output hash) is written beside each export.
Nothing is downloaded: the source checkpoints must already exist locally.
"""
import argparse
import json
import os
from pathlib import Path

from .longrun import file_sha256, save_json


def export(weights: Path, output_dir: Path, imgsz: int = 1280) -> Path:
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"{weights} does not exist; weights are never downloaded implicitly")
    config_dir = Path(__file__).resolve().parent.parent / ".inference-config"
    config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    import ultralytics
    from ultralytics import YOLO
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{weights.stem}-{imgsz}-dynamic.onnx"
    if target.exists():
        raise FileExistsError(f"{target} exists; delete it deliberately before re-exporting")
    # dynamic=True keeps Ultralytics' rectangular letterboxing (same preprocessing
    # as the .pt path); simplify=False avoids pulling in onnxslim.
    produced = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, dynamic=True, simplify=False, half=False))
    produced.replace(target)
    save_json(target.with_suffix(".json"), {
        "source_checkpoint": str(weights.resolve()), "source_sha256": file_sha256(weights),
        "onnx": target.name, "onnx_sha256": file_sha256(target),
        "export": {"format": "onnx", "imgsz": imgsz, "dynamic": True, "simplify": False, "half": False},
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "note": "Format conversion of the same weights for ONNX Runtime GPU inference. Results differ "
                "slightly from PyTorch CPU; compare before treating runs as interchangeable."})
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models/onnx-export"))
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    for weights in args.weights:
        print(export(weights, args.output_dir, args.imgsz))


if __name__ == "__main__":
    main()
