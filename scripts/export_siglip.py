#!/usr/bin/env python3
"""Export the SigLIP-384 image tower at a fixed batch of 8."""

import argparse
from pathlib import Path

import open_clip
import torch


class ImageTower(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model, preprocess = open_clip.create_model_from_pretrained(
        "hf-hub:timm/ViT-SO400M-14-SigLIP-384")
    model.eval().requires_grad_(False)
    print("preprocess", preprocess)
    # Only the image tower is exported; dropping the text tower halves peak memory.
    model.transformer = None
    model.text = None
    tower = ImageTower(model).eval()
    dummy = torch.zeros(8, 3, 384, 384)
    with torch.no_grad():
        out = tower(dummy)
        print("out", tuple(out.shape))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            tower,
            dummy,
            args.out,
            input_names=["image"],
            output_names=["embedding"],
            opset_version=17,
            dynamo=False,
        )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
