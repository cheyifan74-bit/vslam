#!/usr/bin/env python3
"""Export MixVPR (ResNet50) to ONNX for C++ / ONNX Runtime inference.

Default variant is the 512-D model used by D_VINS (256 channels x 2 rows).

Prerequisites:
  1. Clone MixVPR next to this workspace (or pass --mixvpr-root):
       git clone https://github.com/amaralibey/MixVPR.git /home/aa/MixVPR
  2. Download the official checkpoint (Google Drive links in MixVPR README):
       resnet50_MixVPR_512_channels(256)_rows(2).ckpt
  3. pip install torch torchvision onnx onnxruntime pillow
     (export does not need pytorch_lightning / MixVPR training deps)

Example:
  python3 tools/export_mixvpr_onnx.py \\
    --mixvpr-root /home/aa/MixVPR \\
    --ckpt /home/aa/MixVPR/LOGS/resnet50_MixVPR_512_channels\\(256\\)_rows\\(2\\).ckpt \\
    --out /home/aa/vslam_ws/src/vslam/data/models/mixvpr_resnet50_512.onnx
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F

# Official MixVPR ResNet50 aggregators. Output dim = out_channels * out_rows.
VARIANTS = {
    512: {
        "out_channels": 256,
        "out_rows": 2,
        "drive_id": "1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj",
        "ckpt_name": "resnet50_MixVPR_512_channels(256)_rows(2).ckpt",
    },
    4096: {
        "out_channels": 1024,
        "out_rows": 4,
        "drive_id": "1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L",
        "ckpt_name": "resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt",
    },
}


def _load_py(path: Path, module_name: str) -> ModuleType:
    """Load a MixVPR source file without executing package ``__init__.py``.

    ``models.backbones`` / ``models.aggregators`` pull in timm / Lightning
    extras we do not need for ONNX export.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def infer_variant(ckpt: Path, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    name = ckpt.name
    match = re.search(r"MixVPR_(\d+)_", name)
    if match:
        dim = int(match.group(1))
        if dim in VARIANTS:
            return dim
    return 512


class MixVprInfer(torch.nn.Module):
    """Inference graph only: ResNet50 + MixVPR aggregator.

    MixVPR's ``VPRModel`` lives in ``main.py`` and subclasses Lightning, which
    also constructs training loss/miner. Export does not need any of that.
    State-dict keys still match ``VPRModel`` (``backbone.*``, ``aggregator.*``).
    """

    def __init__(self, mixvpr_root: Path, out_channels: int, out_rows: int):
        super().__init__()
        ResNet = _load_py(
            mixvpr_root / "models" / "backbones" / "resnet.py",
            "mixvpr_export_resnet",
        ).ResNet
        MixVPR = _load_py(
            mixvpr_root / "models" / "aggregators" / "mixvpr.py",
            "mixvpr_export_agg",
        ).MixVPR

        # MixVPR ckpt overwrites all weights; skip ImageNet download.
        self.backbone = ResNet(
            "resnet50", pretrained=False, layers_to_freeze=1, layers_to_crop=[4]
        )
        self.aggregator = MixVPR(
            in_channels=1024,
            in_h=20,
            in_w=20,
            out_channels=out_channels,
            mix_depth=4,
            mlp_ratio=1,
            out_rows=out_rows,
        )

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregator output without the MixVPR L2 normalize."""
        feats = self.backbone(x)
        agg = self.aggregator
        y = feats.flatten(2)
        y = agg.mix(y)
        y = y.permute(0, 2, 1)
        y = agg.channel_proj(y)
        y = y.permute(0, 2, 1)
        y = agg.row_proj(y)
        return y.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(x)


def _looks_like_checkpoint(path: Path) -> bool:
    head = path.read_bytes()[:256]
    if head.lstrip().startswith((b"<!DOCTYPE", b"<html", b"<HTML")):
        return False
    return path.stat().st_size > 1_000_000


def load_mixvpr(
    mixvpr_root: Path, ckpt: Path, device: str, variant: int
) -> torch.nn.Module:
    cfg = VARIANTS[variant]
    if not _looks_like_checkpoint(ckpt):
        raise SystemExit(
            f"Not a MixVPR weight file: {ckpt} "
            f"({ckpt.stat().st_size} bytes)\n"
            "Google Drive often returns an HTML virus-scan page instead of the "
            f".ckpt. Download ResNet50 {variant}-D weights from the MixVPR README "
            f"table (file id {cfg['drive_id']}) and replace this path."
        )

    model = MixVprInfer(mixvpr_root, cfg["out_channels"], cfg["out_rows"])
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    # Lightning checkpoints sometimes wrap weights under "state_dict".
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Strip a leading "model." prefix if present.
    if isinstance(state, dict) and any(k.startswith("model.") for k in state):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    return model


class MixVprOnnxWrapper(torch.nn.Module):
    """Raw MixVPR descriptor; optional L2 for inner-product / cosine search."""

    def __init__(self, model: MixVprInfer, l2_normalize: bool):
        super().__init__()
        self.model = model
        self.l2_normalize = l2_normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, 320, 320), ImageNet-normalized RGB
        desc = self.model.forward_raw(x)
        if self.l2_normalize:
            desc = F.normalize(desc, p=2, dim=1)
        return desc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mixvpr-root",
        type=Path,
        default=Path("/home/aa/MixVPR"),
        help="Path to cloned amaralibey/MixVPR repository",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to MixVPR .ckpt weights",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("mixvpr_resnet50_512.onnx"),
        help="Output ONNX path",
    )
    parser.add_argument(
        "--variant",
        type=int,
        choices=sorted(VARIANTS),
        default=None,
        help="Descriptor dim. Default: infer from ckpt name, else 512.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--l2-normalize",
        action="store_true",
        help="Bake L2 normalize into the ONNX graph (off by default)",
    )
    parser.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare Torch vs ONNX Runtime on a random input",
    )
    args = parser.parse_args()
    variant = infer_variant(args.ckpt, args.variant)
    cfg = VARIANTS[variant]
    desc_dim = cfg["out_channels"] * cfg["out_rows"]

    if not args.mixvpr_root.is_dir():
        raise SystemExit(
            f"MixVPR root not found: {args.mixvpr_root}\n"
            "  git clone https://github.com/amaralibey/MixVPR.git "
            f"{args.mixvpr_root}"
        )
    if not args.ckpt.is_file():
        raise SystemExit(
            f"Checkpoint not found: {args.ckpt}\n"
            f"Download ResNet50 {variant}-D weights from the MixVPR README "
            f"table (file id {cfg['drive_id']})."
        )

    model = load_mixvpr(args.mixvpr_root, args.ckpt, args.device, variant)
    wrapped = MixVprOnnxWrapper(model, l2_normalize=args.l2_normalize)
    wrapped.eval()

    dummy = torch.randn(1, 3, 320, 320, device=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            str(args.out),
            input_names=["image"],
            output_names=["descriptor"],
            dynamic_axes={
                "image": {0: "batch"},
                "descriptor": {0: "batch"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,  # torch>=2.9 default dynamo exporter needs onnxscript
        )

    print(f"Wrote {args.out.resolve()}")
    print("Input : image       float32 [N,3,320,320]  (ImageNet mean/std)")
    print(
        f"Output: descriptor  float32 [N,{desc_dim}]       "
        f"({'L2-normalized' if args.l2_normalize else 'raw, no L2'})"
    )

    if args.verify:
        import numpy as np
        import onnxruntime as ort

        with torch.no_grad():
            torch_out = wrapped(dummy).cpu().numpy()
        sess = ort.InferenceSession(
            str(args.out), providers=["CPUExecutionProvider"]
        )
        ort_out = sess.run(
            None, {"image": dummy.cpu().numpy().astype(np.float32)}
        )[0]
        max_abs = float(np.max(np.abs(torch_out - ort_out)))
        cos = float(
            np.sum(torch_out * ort_out)
            / (np.linalg.norm(torch_out) * np.linalg.norm(ort_out) + 1e-12)
        )
        print(
            f"Verify dim={ort_out.shape} max|torch-onnx|={max_abs:.6e}  "
            f"cosine={cos:.8f}"
        )
        if ort_out.shape[-1] != desc_dim:
            raise SystemExit(
                f"ONNX descriptor dim {ort_out.shape[-1]} != expected {desc_dim}"
            )


if __name__ == "__main__":
    main()
