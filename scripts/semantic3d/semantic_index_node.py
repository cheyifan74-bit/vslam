#!/usr/bin/env python3
"""Phase-1 semantic keyframe index: SigLIP2 text→image top-K retrieval.

Reads the committed keyframe set straight from <map>/state/state.db (written by
MapManager). There is no append-only event log to replay: on every poll the node
reconciles its in-memory index against the committed rows, encoding the missing
images and dropping the ones that were culled.

  SELECT image_id, rgb_name FROM keyframes WHERE state='committed';

All SigLIP2 inference runs on a single background worker thread, so the ROS
main thread never blocks on a CUDA/CPU forward pass:

  - image encoding is batched (up to BATCH_SIZE at a time);
  - a query is answered as soon as the current batch finishes (low latency);
  - a frame whose RGB is missing retries for a short grace period, then is
    remembered in an in-memory blacklist so it is never re-encoded every poll;
  - a frame that fails to encode is blacklisted immediately.

Pre-download weights (once, offline from HF):
  python3 scripts/semantic3d/download_siglip2.py
  # → data/models/siglip2-so400m-patch14-384/

Query (ROS2):
  ros2 topic pub /semantic_query std_msgs/String "{data: 'a red chair'}"
  ros2 topic echo /semantic_result

The node loads ONLY from a local directory (local_files_only=True); it never
downloads weights at startup.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Header
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2 as pc2
except ImportError:  # pragma: no cover
    rclpy = None
    Node = object  # type: ignore
    String = None  # type: ignore
    Header = None  # type: ignore
    PointCloud2 = None  # type: ignore
    pc2 = None  # type: ignore


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def _default_local_siglip2() -> Path:
    try:
        from model_paths import default_siglip_dir

        return default_siglip_dir()
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parents[1] / "data" / "models" / "siglip2-so400m-patch14-384"


def _resolve_local_model(model_id: str) -> Path:
    """Require a local directory. Never download at runtime (real-time system)."""
    path = Path(model_id).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"SigLIP2 local model dir not found: {path}\n"
            f"Pre-download with: python3 scripts/semantic3d/download_siglip2.py\n"
            f"(expected default: {_default_local_siglip2()})"
        )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Incomplete SigLIP2 snapshot (missing config.json): {path}\n"
            f"Re-run: python3 scripts/semantic3d/download_siglip2.py --out {path}"
        )
    return path


class SigLIP2Index:
    """In-memory image embedding store + text query (SigLIP2).

    All model inference happens on one background worker thread, so image
    encoding, text queries and index updates never run on the ROS main thread.
    """

    BATCH_SIZE = 8
    MISSING_GRACE_S = 5.0

    def __init__(self, model_id: str, device: Optional[str] = None, log=print):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.log = log
        if device is None:
            # SigLIP2 on CPU by default: the GPU is reserved for SAM3 + dense S2M2.
            device = "cpu"
        self.device = device
        model_path = _resolve_local_model(model_id)
        self.model_id = str(model_path)
        self.log(f"[SEMANTIC] Loading local SigLIP2 from {model_path} on {device} (fp16 on cuda) ...")
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        # Load on CPU first, then move + cast: avoids a fp32 VRAM spike and
        # guarantees fp16 on GPU (transformers 5.x deprecates torch_dtype).
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).eval()
        self.model = self.model.to(device=device, dtype=dtype)

        # Shared state, guarded by _cv (worker writes, main thread reads).
        self._cv = threading.Condition()
        self.ids: List[int] = []
        self.paths: List[str] = []
        self.emb: Optional[np.ndarray] = None
        self.id_to_row: Dict[int, int] = {}
        self._queued: set = set()            # image_ids in the encode queue
        self._failed: set = set()            # image_ids to skip forever
        self._missing_since: Dict[int, float] = {}
        self._work_count = 0                 # queued + in-flight encodes
        self._dirty = False
        self._saved_count = 0                # rows already flushed to disk
        self._needs_compact = False          # a deletion shifted rows → full rewrite

        self._img_queue: "queue.Queue" = queue.Queue()
        self._query_queue: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self.log("[SEMANTIC] Model ready (offline).")

    # ------------------------------------------------------------------ #
    # Model inference (worker thread only)
    # ------------------------------------------------------------------ #
    def _as_feature_tensor(self, out):
        """Newer transformers SigLIP2 returns BaseModelOutputWithPooling."""
        if self.torch.is_tensor(out):
            return out
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "image_embeds") and out.image_embeds is not None:
            return out.image_embeds
        if hasattr(out, "text_embeds") and out.text_embeds is not None:
            return out.text_embeds
        if isinstance(out, (tuple, list)) and out:
            return out[0]
        raise TypeError(f"Unsupported feature output type: {type(out)}")

    def _encode_images(self, paths: List[str]) -> np.ndarray:
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt")
        dtype = getattr(self.model, "dtype", self.torch.float32)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=dtype)
        with self.torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                out = self.model.get_image_features(pixel_values=pixel_values)
            else:
                out = self.model(pixel_values=pixel_values)
            feats = self._as_feature_tensor(out)
        feats = feats.float().cpu().numpy()
        for im in images:
            im.close()
        return _l2_normalize(feats, axis=1)

    def _encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text], padding="max_length", max_length=64, return_tensors="pt"
        )
        inputs = {
            k: v.to(self.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")
        }
        with self.torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                out = self.model.get_text_features(**inputs)
            else:
                out = self.model(**inputs)
            feats = self._as_feature_tensor(out)
        feats = feats.float().cpu().numpy()
        return _l2_normalize(feats, axis=1)[0]

    # ------------------------------------------------------------------ #
    # Worker thread
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._run, daemon=True, name="siglip2-worker")
        self._worker.start()
        self.log("[SEMANTIC] Worker thread started (batch=%d)." % self.BATCH_SIZE)

    def _run(self) -> None:
        while not self._stop.is_set():
            # 1) Answer one query first (low latency).
            try:
                text, topk, cb = self._query_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                self._answer(text, topk, cb)
                continue

            # 2) Accumulate an image batch (bounded wait so we never spin).
            batch: List[Tuple[int, str]] = []
            try:
                batch.append(self._img_queue.get(timeout=0.2))
            except queue.Empty:
                time.sleep(0.02)
                continue
            deadline = time.time() + 0.1
            while len(batch) < self.BATCH_SIZE and time.time() < deadline:
                try:
                    batch.append(self._img_queue.get_nowait())
                except queue.Empty:
                    break
            self._encode_batch(batch)
        self._drain()

    def _drain(self) -> None:
        while True:
            try:
                text, topk, cb = self._query_queue.get_nowait()
            except queue.Empty:
                break
            self._answer(text, topk, cb)
        while True:
            try:
                item = self._img_queue.get_nowait()
            except queue.Empty:
                break
            self._encode_batch([item])

    def _answer(self, text: str, topk: int, cb) -> None:
        hits = self._search_impl(text, topk)
        try:
            cb(hits)
        except Exception as e:  # noqa: BLE001
            self.log(f"[SEMANTIC] query callback failed: {e}")

    def _encode_batch(self, batch: List[Tuple[int, str]]) -> None:
        ready: List[Tuple[int, str]] = []
        now = time.time()
        for image_id, path in batch:
            if not Path(path).is_file():
                with self._cv:
                    self._queued.discard(image_id)
                    self._missing_since.setdefault(image_id, now)
                continue
            ready.append((image_id, path))

        if not ready:
            with self._cv:
                self._work_count -= len(batch)
                self._cv.notify_all()
            return

        try:
            vecs = self._encode_images([p for _, p in ready])
        except Exception as e:  # noqa: BLE001
            self.log(f"[SEMANTIC] encode batch failed: {e}")
            with self._cv:
                for image_id, _ in ready:
                    self._failed.add(image_id)
                    self._queued.discard(image_id)
                    self._missing_since.pop(image_id, None)
                self._work_count -= len(batch)
                self._cv.notify_all()
            return

        with self._cv:
            new_vecs: List[np.ndarray] = []
            new_ids: List[int] = []
            new_paths: List[str] = []
            for (image_id, path), vec in zip(ready, vecs):
                if image_id in self.id_to_row:
                    # request_encode skips indexed ids, so this is unreachable;
                    # keep it a no-op to avoid in-place mutation of a published array.
                    continue
                new_ids.append(image_id)
                new_paths.append(path)
                new_vecs.append(vec.reshape(1, -1))
            if new_ids:
                block = np.vstack(new_vecs)
                self.emb = block if self.emb is None else np.vstack([self.emb, block])
                for image_id, path in zip(new_ids, new_paths):
                    self.id_to_row[image_id] = len(self.ids)
                    self.ids.append(image_id)
                    self.paths.append(path)
            for image_id, _ in ready:
                self._queued.discard(image_id)
                self._missing_since.pop(image_id, None)
            self._work_count -= len(batch)
            self._dirty = True
            self._cv.notify_all()

        self.log(f"[SEMANTIC] encoded {len(ready)} keyframe(s), N={len(self.ids)}")

    # ------------------------------------------------------------------ #
    # Query path (runs on the worker via _query_queue)
    # ------------------------------------------------------------------ #
    def _search_impl(self, text: str, topk: int) -> List[Tuple[int, str, float]]:
        with self._cv:
            emb = self.emb
            ids = list(self.ids)
            paths = list(self.paths)
        if emb is None or not ids:
            return []
        q = self._encode_text(text)
        scores = emb @ q
        order = np.argsort(-scores)
        hits: List[Tuple[int, str, float]] = []
        for i in order:
            if not Path(paths[i]).is_file():
                continue
            hits.append((int(ids[i]), paths[i], float(scores[i])))
            if len(hits) >= topk:
                break
        return hits

    # ------------------------------------------------------------------ #
    # Main-thread API
    # ------------------------------------------------------------------ #
    def request_encode(self, image_id: int, path: str) -> None:
        with self._cv:
            if image_id in self.id_to_row or image_id in self._queued or image_id in self._failed:
                return
            self._queued.add(image_id)
            self._work_count += 1
            self._img_queue.put((image_id, path))

    def request_query(self, text: str, topk: int, cb) -> None:
        self._query_queue.put((text, topk, cb))

    def delete_ids(self, image_ids: List[int]) -> None:
        remove = set(image_ids)
        if not remove:
            return
        with self._cv:
            keep = [i for i, iid in enumerate(self.ids) if iid not in remove]
            if len(keep) != len(self.ids):
                self.ids = [self.ids[i] for i in keep]
                self.paths = [self.paths[i] for i in keep]
                self.emb = self.emb[keep] if keep else None
                self.id_to_row = {iid: i for i, iid in enumerate(self.ids)}
                self._dirty = True
                self._needs_compact = True
            for iid in remove:
                self._failed.discard(iid)
                self._missing_since.pop(iid, None)

    def current_ids(self) -> List[int]:
        with self._cv:
            return list(self.ids)

    def known_ids(self) -> set:
        """image_ids that are indexed, queued, or failed — nothing left to do."""
        with self._cv:
            return set(self.ids) | set(self._queued) | set(self._failed)

    def missing_ids(self) -> Dict[int, float]:
        with self._cv:
            return dict(self._missing_since)

    def mark_missing_failed(self, image_id: int) -> None:
        with self._cv:
            self._failed.add(image_id)
            self._queued.discard(image_id)
            self._missing_since.pop(image_id, None)

    def prune_stale(self, desired: set) -> None:
        """Forget failed/missing bookkeeping for ids that are no longer committed."""
        with self._cv:
            for iid in list(self._failed):
                if iid not in desired:
                    self._failed.discard(iid)
            for iid in list(self._missing_since):
                if iid not in desired:
                    self._missing_since.pop(iid, None)

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait until every queued encode has been processed (offline use)."""
        with self._cv:
            return self._cv.wait_for(lambda: self._work_count == 0, timeout=timeout)

    def search_sync(self, text: str, topk: int = 5, timeout: float = 30.0) -> List[Tuple[int, str, float]]:
        """Blocking query routed through the worker (offline use)."""
        done = threading.Event()
        result: List[Tuple[int, str, float]] = []

        def _cb(hits: List[Tuple[int, str, float]]) -> None:
            result.extend(hits)
            done.set()

        self.request_query(text, topk, _cb)
        done.wait(timeout)
        return result

    def _write_meta(self, meta_path: Path, ids: List[int], paths: List[str], dim: int) -> None:
        """Write meta.json atomically (temp + rename); it is the commit marker."""
        meta = {"model": self.model_id, "dim": dim, "ids": ids, "paths": paths}
        tmp = meta_path.with_name(meta_path.name + ".tmp")
        tmp.write_text(json.dumps(meta))
        os.replace(str(tmp), str(meta_path))

    def save(self, index_dir: Path, force: bool = False) -> None:
        """Persist the index.

        Incremental: only newly encoded rows are appended to embeddings.npy
        (a memmap with spare capacity). A full rewrite happens only on the first
        write, on capacity growth, or after deletions shifted the rows. meta.json
        is written last via an atomic rename, so a crash never leaves a half
        written pair — load() detects any leftover mismatch and rebuilds.
        """
        with self._cv:
            if not force and not self._dirty:
                return
            ids = list(self.ids)
            paths = list(self.paths)
            n = len(ids)
            emb = self.emb  # immutable once published (worker replaces, never mutates)
            dim = int(emb.shape[1]) if emb is not None and n > 0 else 0
            saved_count = self._saved_count
            full = (n == 0) or self._needs_compact or saved_count > n
            if n > 0:
                if full:
                    payload = emb[:n]
                    start = 0
                else:
                    payload = emb[saved_count:n]
                    start = saved_count
            else:
                payload = None
                start = 0

        index_dir.mkdir(parents=True, exist_ok=True)
        emb_path = index_dir / "embeddings.npy"
        meta_path = index_dir / "meta.json"

        if n == 0:
            if emb_path.exists():
                try:
                    emb_path.unlink()
                except OSError:
                    pass
            self._write_meta(meta_path, ids, paths, 0)
            with self._cv:
                self._saved_count = 0
                self._needs_compact = False
                self._dirty = False
            return

        mm = None
        if emb_path.exists() and not full:
            try:
                mm = np.lib.format.open_memmap(str(emb_path), mode="r+")
                if mm.shape[1] != dim or mm.shape[0] < n:
                    del mm
                    mm = None
            except Exception:
                mm = None
        if mm is None:
            # First write, dim change or growth: allocate fresh capacity and rewrite.
            cap = max(n * 2, 1024)
            mm = np.lib.format.open_memmap(str(emb_path), mode="w+", dtype="float32", shape=(cap, dim))
            full = True
            payload = emb[:n]
            start = 0

        if full:
            mm[:n] = payload
        else:
            mm[start:start + len(payload)] = payload
        mm.flush()
        del mm  # close the memmap

        self._write_meta(meta_path, ids, paths, dim)
        with self._cv:
            self._saved_count = n
            self._needs_compact = False
            self._dirty = False

    def load(self, index_dir: Path) -> bool:
        meta_path = index_dir / "meta.json"
        emb_path = index_dir / "embeddings.npy"
        if not meta_path.is_file():
            return False
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            return False

        model = meta.get("model", "")
        if model and model != self.model_id:
            self.log(f"[SEMANTIC] model mismatch (stored={model!r}), rebuilding index")
            return False
        ids = [int(x) for x in meta.get("ids", [])]
        paths = list(meta.get("paths", []))
        dim = int(meta.get("dim", 0))
        if len(ids) != len(paths):
            self.log("[SEMANTIC] ids/paths mismatch in meta, rebuilding index")
            return False

        if not ids:
            self.ids = []
            self.paths = []
            self.emb = None
            self.id_to_row = {}
            self._saved_count = 0
            self._needs_compact = False
            return True

        if dim <= 0 or not emb_path.is_file():
            return False
        try:
            emb = np.load(str(emb_path))
        except Exception as e:  # noqa: BLE001
            self.log(f"[SEMANTIC] failed to load embeddings: {e}")
            return False
        if emb.ndim != 2 or emb.shape[0] < len(ids) or emb.shape[1] != dim:
            self.log(
                f"[SEMANTIC] embeddings shape {emb.shape} != ({len(ids)},{dim}), rebuilding index"
            )
            return False

        self.emb = np.ascontiguousarray(emb[:len(ids)])
        self.ids = ids
        self.paths = paths
        self.id_to_row = {iid: i for i, iid in enumerate(ids)}
        self._saved_count = len(ids)
        self._needs_compact = False
        self.log(f"[SEMANTIC] Loaded index N={len(ids)} dim={dim} from {index_dir}")
        return True

    def size(self) -> int:
        with self._cv:
            return len(self.ids)

    def shutdown(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
            self._worker = None


class StateDbPoller:
    """Reconciles the index against the committed rows of the state ledger."""

    def __init__(self, state_db_path: Path, images_dir: Path, index: SigLIP2Index, log=print):
        self.state_db_path = state_db_path
        self.images_dir = images_dir
        self.index = index
        self.log = log

    def _committed(self) -> Dict[int, str]:
        if not self.state_db_path.is_file():
            return {}
        import sqlite3

        try:
            conn = sqlite3.connect(
                f"file:{self.state_db_path}?mode=ro", uri=True, timeout=2.0
            )
        except Exception as e:  # noqa: BLE001
            self.log(f"[SEMANTIC] cannot open state db: {e}")
            return {}
        try:
            cur = conn.execute(
                "SELECT image_id, rgb_name FROM keyframes "
                "WHERE state='committed' AND rgb_name <> ''"
            )
            out = {int(i): str(rgb) for i, rgb in cur.fetchall()}
        except Exception as e:  # noqa: BLE001
            self.log(f"[SEMANTIC] state db query failed: {e}")
            out = {}
        finally:
            conn.close()
        return out

    def reconcile(self) -> int:
        """Enqueue missing committed frames; drop culled ones. Returns enqueued count."""
        committed = self._committed()
        desired = {iid: str(self.images_dir / rgb) for iid, rgb in committed.items() if rgb}
        desired_set = set(desired.keys())

        # Drop index entries that are no longer committed, and forget bookkeeping
        # for ids that are gone (keeps the blacklists bounded).
        drop = [iid for iid in self.index.current_ids() if iid not in desired_set]
        if drop:
            self.index.delete_ids(drop)
        self.index.prune_stale(desired_set)

        # Missing-file grace: give up after MISSING_GRACE_S.
        now = time.time()
        for iid, since in self.index.missing_ids().items():
            if now - since > self.index.MISSING_GRACE_S:
                self.index.mark_missing_failed(iid)
                self.log(
                    f"[SEMANTIC] give up image_id={iid}: RGB missing >"
                    f"{self.index.MISSING_GRACE_S:.1f}s"
                )

        # Enqueue anything committed but not yet indexed/queued/failed.
        known = self.index.known_ids()
        enqueued = 0
        for iid, path in desired.items():
            if iid in known:
                continue
            self.index.request_encode(iid, path)
            enqueued += 1
        return enqueued


class SemanticIndexNode(Node if rclpy else object):
    def __init__(
        self,
        map_root: str,
        model: str,
        topk: int,
        device: Optional[str],
        enable_3d: bool = True,
        min_sam3_score: float = 0.5,
        world_frame: str = "global",
        cloud_topic: str = "semantic_cloud",
    ):
        if rclpy:
            super().__init__("semantic_index_node")

            def _log(msg: str) -> None:
                self.get_logger().info(msg)

            log = _log
        else:
            log = print

        self.map_root = Path(map_root)
        self.map_dir = self.map_root / "map"
        self.semantic_dir = self.map_root / "map" / "semantic"
        self.state_db_path = self.map_root / "map" / "state" / "state.db"
        self.images_dir = self.map_root / "map" / "images"
        self.index_dir = self.semantic_dir / "index"
        self.topk = topk
        self.index = SigLIP2Index(model, device=device, log=log)
        self.index.load(self.index_dir)
        self.index.start()
        self.poller = StateDbPoller(self.state_db_path, self.images_dir, self.index, log=log)

        # Stage-2 3D localization (SAM3 + IR depth). Loaded lazily on first query.
        self.enable_3d = enable_3d
        self.min_sam3_score = min_sam3_score
        self.world_frame = world_frame
        self._sam3 = None
        self._sam3_tried = False
        self._sam3_lock = threading.Lock()

        if rclpy:
            self.query_sub = self.create_subscription(String, "semantic_query", self._on_query, 10)
            self.result_pub = self.create_publisher(String, "semantic_result", 10)
            self.cloud_pub = self.create_publisher(PointCloud2, cloud_topic, 1)
            # Reconcile is cheap (SQLite only): it enqueues encodes and drops culls.
            self.create_timer(0.5, self._on_poll)
            self.create_timer(10.0, self._periodic_save)
            self.get_logger().info(
                f"Reading committed keyframes state={self.state_db_path} "
                f"query=semantic_query topk={topk} device={self.index.device} "
                f"3d={enable_3d} cloud_topic={cloud_topic} frame={world_frame}"
            )

    def _on_poll(self) -> None:
        try:
            self.poller.reconcile()
        except Exception as e:  # noqa: BLE001
            if rclpy:
                self.get_logger().error(f"poll error: {e}")
            else:
                print(f"[SEMANTIC] poll error: {e}", flush=True)

    def _on_query(self, msg) -> None:
        query = msg.data.strip()
        if not query:
            return
        # Answer on the worker thread; publish from the callback.
        self.index.request_query(
            query, self.topk, lambda hits, q=query: self._on_query_done(q, hits)
        )

    def _on_query_done(self, query: str, hits: List[Tuple[int, str, float]]) -> None:
        # 1) retrieval ranking (JSON, no point cloud).
        self._publish_result(query, hits)
        # 2) 3D point cloud -> RViz (PointCloud2).
        if rclpy:
            self._publish_semantic_cloud(query, hits)

    def _publish_result(self, query: str, hits: List[Tuple[int, str, float]]) -> None:
        payload = {
            "query": query,
            "results": [
                {
                    "image_id": iid,
                    "file_name": Path(path).name,
                    "rgb_path": path,
                    "score": score,
                }
                for iid, path, score in hits
            ],
        }
        out = String()
        out.data = json.dumps(payload, ensure_ascii=False)
        self.result_pub.publish(out)
        self.get_logger().info(
            f"query={query!r} → " + ", ".join(f"{iid}:{score:.3f}" for iid, _, score in hits)
        )

    def _ensure_sam3(self):
        with self._sam3_lock:
            if self._sam3_tried:
                return self._sam3
            self._sam3_tried = True
            try:
                from sam3_segmenter import SAM3Segmenter

                self._sam3 = SAM3Segmenter(device="cuda", batch_size=2)
                self.get_logger().info("SAM3 loaded for 3D localization")
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"SAM3 unavailable, 3D disabled: {e}")
                self._sam3 = None
            return self._sam3

    def _publish_semantic_cloud(self, query: str, hits: List[Tuple[int, str, float]]) -> None:
        if not self.enable_3d:
            return
        sam3 = self._ensure_sam3()
        if sam3 is None or not hits:
            self._publish_cloud_msg(None)  # clear on no result
            return
        try:
            from localize3d import localize3d_multi

            ids = [int(iid) for iid, _, _ in hits]
            scores = {int(iid): float(s) for iid, _, s in hits}
            res = localize3d_multi(
                query,
                ids,
                self.map_dir,
                sam3,
                retrieval_scores=scores,
                top_k=self.topk,
                voxel=0.02,
                min_sam3_score=self.min_sam3_score,
            )
            self._publish_cloud_msg(res.get("point_cloud"))
            self.get_logger().info(
                f"cloud query={query!r} frames={res.get('num_frames')} points={res.get('num_points')} "
                f"centroid={None if res.get('centroid') is None else res['centroid'].round(3).tolist()}"
            )
            for pf in res.get("per_frame", []):
                self.get_logger().info(
                    f"  id={pf['frame_id']} rgb={int(pf.get('has_rgb', False))} "
                    f"depth={int(pf.get('has_depth', False))} sam3={pf['sam3_score']:.3f} "
                    f"mask_px={pf.get('mask_px', 0)} cloud={0 if pf['point_cloud'] is None else len(pf['point_cloud'])}"
                )
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"3D localization failed: {e}")

    def _publish_cloud_msg(self, cloud) -> None:
        if pc2 is None or self.cloud_pub is None:
            return
        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = self.world_frame
        pts = [] if cloud is None else cloud.astype(np.float32).tolist()
        msg = pc2.create_cloud_xyz32(hdr, pts)
        self.cloud_pub.publish(msg)

    def _periodic_save(self) -> None:
        self.index.save(self.index_dir)

    def shutdown(self) -> None:
        self.index.shutdown()
        self.index.save(self.index_dir, force=True)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="SigLIP2 semantic keyframe indexer")
    parser.add_argument("--map-root", default=os.environ.get("VSLAM_MAP_ROOT", "/tmp/vslam_map"))
    parser.add_argument(
        "--model",
        default=str(_default_local_siglip2()),
        help="Local SigLIP2 directory (must already exist; no runtime download)",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--query", default="", help="One-shot offline query then exit")
    parser.add_argument("--no-3d", dest="enable_3d", action="store_false", default=True,
                        help="disable SAM3 3D localization / point cloud publishing")
    parser.add_argument("--min-sam3-score", type=float, default=0.5,
                        help="reject masks with SAM3 confidence below this")
    parser.add_argument("--cloud-topic", default="semantic_cloud", help="PointCloud2 topic")
    parser.add_argument("--world-frame", default="global", help="PointCloud2 frame_id")
    args, _ = parser.parse_known_args(argv)

    if args.query:
        index = SigLIP2Index(args.model, device=args.device)
        index_dir = Path(args.map_root) / "map" / "semantic" / "index"
        index.load(index_dir)
        index.start()
        poller = StateDbPoller(
            Path(args.map_root) / "map" / "state" / "state.db",
            Path(args.map_root) / "map" / "images",
            index,
        )
        poller.reconcile()
        index.flush(timeout=60.0)
        hits = index.search_sync(args.query, topk=args.topk)
        index.shutdown()
        print(
            json.dumps(
                {
                    "query": args.query,
                    "results": [
                        {
                            "image_id": i,
                            "file_name": Path(p).name,
                            "rgb_path": p,
                            "score": s,
                        }
                        for i, p, s in hits
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if rclpy is None:
        raise SystemExit("rclpy not available; use --query for offline mode")

    rclpy.init(args=argv)
    node = SemanticIndexNode(
        args.map_root, args.model, args.topk, args.device,
        enable_3d=args.enable_3d,
        min_sam3_score=args.min_sam3_score,
        world_frame=args.world_frame,
        cloud_topic=args.cloud_topic,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
