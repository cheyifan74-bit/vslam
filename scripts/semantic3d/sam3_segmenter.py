"""SAM3 (Segment Anything Model 3) text-prompted segmentation wrapper.

Aligns with memory-over-maps' sam3_segmenter.py. SAM3 requires CUDA.

Install (one of):
  pip install sam3                                  # PyPI 0.1.4+
  git clone https://github.com/facebookresearch/sam3 && pip install -e sam3   # MoM's path
Model weights: build_sam3_image_model(load_from_HF=True) downloads once.

Returns {'best_mask': (H,W) bool | None, 'best_score': float}.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    from PIL import Image
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    Image = None  # type: ignore

logger = logging.getLogger(__name__)

_EMPTY_RESULT = {"best_mask": None, "best_score": 0.0}


def default_sam3_dirs():
    """Candidate local weight dirs (robust for both source and install layouts)."""
    try:
        from model_paths import candidate_dirs

        return candidate_dirs()
    except Exception:  # noqa: BLE001
        root = Path(__file__).resolve().parents[1]  # .../vslam
        return [
            root / "data" / "models" / "sam3",
            root / "data" / "model" / "sam3",
            root / "data" / "models",
            root / "data" / "model",
        ]


def find_local_sam3_weights() -> Optional[str]:
    """Return a local sam3.pt if downloaded under any data/models dir."""
    try:
        from model_paths import find_sam3_weights

        found = find_sam3_weights()
        if found is not None:
            return found
    except Exception:  # noqa: BLE001
        pass
    for d in default_sam3_dirs():
        for cand in (d / "sam3.pt", d / "sam3" / "sam3.pt", d / "sam3.pt"):
            if cand.is_file():
                return str(cand)
    return None


def find_local_bpe() -> Optional[str]:
    try:
        from model_paths import find_sam3_bpe

        found = find_sam3_bpe()
        if found is not None:
            return found
    except Exception:  # noqa: BLE001
        pass
    for d in default_sam3_dirs():
        for cand in (d / "bpe_simple_vocab_16e6.txt.gz", d / "sam3" / "bpe_simple_vocab_16e6.txt.gz"):
            if cand.is_file():
                return str(cand)
    return None


def _extract_best(mask_data, score_data) -> dict:
    """Pick the highest-scoring mask from raw mask/score arrays (robust to shapes)."""
    if mask_data is None or len(mask_data) == 0:
        return dict(_EMPTY_RESULT)

    if torch is not None and isinstance(mask_data, torch.Tensor):
        mask_data = (mask_data.float() if mask_data.dtype == torch.bfloat16 else mask_data).cpu().numpy()

    if score_data is not None:
        if hasattr(score_data, "cpu"):
            score_data = (score_data.float() if getattr(score_data, "dtype", None) == torch.bfloat16 else score_data).cpu().numpy()
        if isinstance(score_data, np.ndarray):
            scores = score_data.reshape(-1).astype(float) if score_data.size else np.array([0.0])
        else:
            scores = np.array([float(s) for s in score_data]) if len(score_data) else np.array([0.0])
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
    else:
        best_idx = 0
        best_score = 1.0

    if isinstance(mask_data, np.ndarray):
        if mask_data.ndim == 4:
            best_mask = mask_data[best_idx, 0].astype(bool)
        elif mask_data.ndim == 3:
            best_mask = mask_data[best_idx].astype(bool)
        elif mask_data.ndim == 2:
            best_mask = mask_data.astype(bool)
        else:
            return dict(_EMPTY_RESULT)
    else:
        m = mask_data[best_idx]
        if hasattr(m, "cpu"):
            m = m.cpu().numpy()
        if m.ndim == 3:
            m = m.squeeze(0)
        best_mask = m.astype(bool)

    return {"best_mask": best_mask, "best_score": best_score}


class SAM3Segmenter:
    """Text-prompted segmentation using SAM3."""

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.0,
        batch_size: int = 2,
        checkpoint_path: Optional[str] = None,
        load_from_HF: bool = True,
        bpe_path: Optional[str] = None,
        enable_segmentation: bool = True,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for SAM3.")
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as e:
            raise ImportError(
                "SAM3 not installed. Run `pip install sam3` or "
                "`git clone https://github.com/facebookresearch/sam3 && pip install -e sam3`."
            ) from e

        if not device.startswith("cuda"):
            logger.warning("SAM3 upstream only supports CUDA; requested device=%s", device)
        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 requires a CUDA GPU, but torch.cuda.is_available() is False.")

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

        # Auto-use weights under <vslam>/data/models/sam3/sam3.pt if present,
        # so no explicit path is needed after the manual download.
        if checkpoint_path is None:
            checkpoint_path = find_local_sam3_weights()
        if bpe_path is None:
            bpe_path = find_local_bpe()

        model_kwargs = dict(device=device, load_from_HF=load_from_HF, enable_segmentation=enable_segmentation)
        if checkpoint_path is not None:
            # Local checkpoint -> skip download_ckpt_from_hf() (offline; HF repo facebook/sam3 is gated).
            model_kwargs["checkpoint_path"] = checkpoint_path
        if bpe_path is not None:
            model_kwargs["bpe_path"] = bpe_path

        self.model = build_sam3_image_model(**model_kwargs)
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=confidence_threshold)
        self._cache: dict = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    @staticmethod
    def _to_pil(image) -> "Image.Image":
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        raise TypeError(f"Unsupported image type: {type(image)}")

    def segment(self, image, text_query: str, cache_key: Optional[tuple] = None) -> dict:
        """Segment objects matching `text_query`. Returns {'best_mask','best_score'}."""
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]

        image = self._to_pil(image)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            state = self.processor.set_image(image)
            state = self.processor.set_text_prompt(text_query, state)
        result = _extract_best(state.get("masks", None), state.get("scores", None))

        if cache_key is not None:
            self._cache[cache_key] = result
        return result

    def segment_batch(
        self,
        images: list,
        text_query: str,
        cache_keys: Optional[list] = None,
        max_batch_size: Optional[int] = None,
    ) -> list:
        """Batch-segment images with the same query.

        Robust implementation: sequential per image in chunks (SAM3 native
        batching in MoM can be added when the installed sam3 exposes it).
        Returns one {'best_mask','best_score'} per input image."""
        n = len(images)
        if n == 0:
            return []
        if cache_keys is not None and len(cache_keys) != n:
            raise ValueError("cache_keys must match images length")

        bs = max_batch_size or self.batch_size or n
        results: list = [dict(_EMPTY_RESULT) for _ in range(n)]
        for start in range(0, n, bs):
            for i in range(start, min(start + bs, n)):
                ck = cache_keys[i] if cache_keys is not None else None
                if ck is not None and ck in self._cache:
                    results[i] = self._cache[ck]
                    continue
                results[i] = self.segment(images[i], text_query, cache_key=ck)
        return results
