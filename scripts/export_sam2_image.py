#!/usr/bin/env python3
"""Export SAM 2.1 Hiera-Small image encoder and point decoder for TensorRT.

The decoder has no memory bank. One positive point yields the best of three masks.
"""

import argparse
from pathlib import Path

import torch
from sam2.build_sam import build_sam2


class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        out = self.model.forward_image(image)
        fpn = out["backbone_fpn"]
        return fpn[0], fpn[1], fpn[2]


class PointDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feat_s0, feat_s1, image_embed, point_coords, point_labels):
        batch = point_coords.shape[0]
        image_embed = image_embed.expand(batch, -1, -1, -1)
        high_res = [
            feat_s0.expand(batch, -1, -1, -1),
            feat_s1.expand(batch, -1, -1, -1),
        ]
        sparse, dense = self.model.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        low_res, ious, _, _ = self.model.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res,
        )
        best = torch.argmax(ious, dim=-1)
        index = torch.arange(batch, device=low_res.device)
        mask = low_res[index, best].unsqueeze(1)
        score = ious[index, best]
        return mask, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        args.checkpoint,
        device="cpu",
        mode="eval",
        apply_postprocessing=False,
    )
    model.eval()
    encoder = ImageEncoder(model).eval()
    decoder = PointDecoder(model).eval()

    image = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        feat_s0, feat_s1, image_embed = encoder(image)
    print("feat_s0", tuple(feat_s0.shape))
    print("feat_s1", tuple(feat_s1.shape))
    print("image_embed", tuple(image_embed.shape))

    points = torch.zeros(32, 1, 2, dtype=torch.float32)
    points[:, 0, :] = 512.0
    labels = torch.ones(32, 1, dtype=torch.int32)
    with torch.no_grad():
        mask, score = decoder(feat_s0, feat_s1, image_embed, points, labels)
    print("mask", tuple(mask.shape), "score", tuple(score.shape))

    torch.onnx.export(
        encoder,
        image,
        str(out / "sam2.1_hiera_s_encoder.onnx"),
        input_names=["image"],
        output_names=["feat_s0", "feat_s1", "image_embed"],
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        decoder,
        (feat_s0, feat_s1, image_embed, points, labels),
        str(out / "sam2.1_hiera_s_decoder.onnx"),
        input_names=["feat_s0", "feat_s1", "image_embed", "point_coords", "point_labels"],
        output_names=["low_res_mask", "iou"],
        opset_version=17,
        dynamo=False,
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
