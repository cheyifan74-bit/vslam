#!/usr/bin/env python3
"""Visualize COLMAP matches (including empty pairs) from database.db."""

import argparse
import os
import sqlite3
import struct

import cv2
import numpy as np

K_MAX_NUM_IMAGES = 2147483647  # int32 max, same as colmap::kMaxNumImages

CONFIG_NAME = {
    0: "UNDEFINED",
    1: "DEGENERATE",
    2: "CALIBRATED",
    3: "UNCALIBRATED",
    4: "PLANAR",
    5: "PANORAMIC",
    6: "PLANAR_OR_PANORAMIC",
    7: "WATERMARK",
    8: "MULTIPLE",
    9: "CALIBRATED_RIG",
}


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % K_MAX_NUM_IMAGES
    image_id1 = (pair_id - image_id2) // K_MAX_NUM_IMAGES
    return int(image_id1), int(image_id2)


def blob_to_array(blob, dtype, shape):
    if blob is None or len(blob) == 0:
        return np.zeros(shape, dtype=dtype)
    array = np.frombuffer(blob, dtype=dtype)
    return array.reshape(shape)


def load_images(con):
    rows = con.execute("SELECT image_id, name FROM images").fetchall()
    return {int(image_id): name for image_id, name in rows}


def load_keypoints(con):
    keypoints = {}
    for image_id, rows, cols, data in con.execute(
            "SELECT image_id, rows, cols, data FROM keypoints"):
        if rows == 0 or cols == 0 or data is None:
            keypoints[int(image_id)] = np.zeros((0, 2), np.float32)
            continue
        mat = blob_to_array(data, np.float32, (int(rows), int(cols)))
        keypoints[int(image_id)] = mat[:, :2].copy()
    return keypoints


def load_match_blob(data, rows, cols):
    if rows is None or rows == 0 or data is None:
        return np.zeros((0, 2), np.uint32)
    return blob_to_array(data, np.uint32, (int(rows), int(cols)))


def read_pairs(con):
    matches = {
        int(pair_id): load_match_blob(data, rows, cols)
        for pair_id, rows, cols, data in con.execute(
            "SELECT pair_id, rows, cols, data FROM matches")
    }
    geometries = {}
    for pair_id, rows, cols, data, config in con.execute(
            "SELECT pair_id, rows, cols, data, config FROM two_view_geometries"):
        geometries[int(pair_id)] = (
            load_match_blob(data, rows, cols),
            int(config),
        )

    pair_ids = sorted(set(matches) | set(geometries))
    pairs = []
    for pair_id in pair_ids:
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        raw = matches.get(pair_id, np.zeros((0, 2), np.uint32))
        inliers, config = geometries.get(
            pair_id, (np.zeros((0, 2), np.uint32), 0))
        pairs.append((image_id1, image_id2, raw, inliers, config))
    return pairs


def resolve_image_path(images_root, name):
    path = os.path.join(images_root, name)
    if os.path.isfile(path):
        return path
    return os.path.join(images_root, os.path.basename(name))


def draw_pair(img1, img2, kpts1, kpts2, raw, inliers, title):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    height = max(h1, h2)
    canvas = np.zeros((height, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    inlier_set = set((int(a), int(b)) for a, b in inliers)
    raw_only = [(int(a), int(b)) for a, b in raw if (int(a), int(b)) not in inlier_set]

    def pt1(idx):
        return (int(round(kpts1[idx, 0])), int(round(kpts1[idx, 1])))

    def pt2(idx):
        return (int(round(kpts2[idx, 0])) + w1, int(round(kpts2[idx, 1])))

    # All keypoints so empty pairs are still inspectable.
    for i in range(len(kpts1)):
        cv2.circle(canvas, pt1(i), 1, (180, 180, 180), -1, lineType=cv2.LINE_AA)
    for i in range(len(kpts2)):
        cv2.circle(canvas, pt2(i), 1, (180, 180, 180), -1, lineType=cv2.LINE_AA)

    for a, b in raw_only:
        if a < 0 or b < 0 or a >= len(kpts1) or b >= len(kpts2):
            continue
        cv2.line(canvas, pt1(a), pt2(b), (0, 165, 255), 1, lineType=cv2.LINE_AA)

    for a, b in inliers:
        a, b = int(a), int(b)
        if a < 0 or b < 0 or a >= len(kpts1) or b >= len(kpts2):
            continue
        p1, p2 = pt1(a), pt2(b)
        cv2.line(canvas, p1, p2, (0, 220, 0), 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, (0, 220, 0), 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p2, 3, (0, 220, 0), 1, lineType=cv2.LINE_AA)

    banner_h = 36
    banner = np.zeros((banner_h, canvas.shape[1], 3), dtype=np.uint8)
    color = (0, 220, 0) if len(inliers) > 0 else ((0, 165, 255) if len(raw) > 0 else (0, 0, 220))
    banner[:] = (30, 30, 30)
    cv2.putText(banner, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
                cv2.LINE_AA)
    return np.vstack([banner, canvas])


def write_index(out_dir, entries):
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>COLMAP matches</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee}",
        "img{max-width:100%;height:auto;border:1px solid #333;margin:8px 0}",
        "h2{margin-top:24px}.empty{opacity:0.85}</style></head><body>",
        "<h1>COLMAP match visualization</h1>",
        "<p>Green = TwoViewGeometry inliers. Orange = raw matches not in inliers. "
        "Gray dots = all keypoints. Empty pairs are still drawn.</p>",
    ]
    html.append("<h2>With inliers (%d)</h2>" % sum(1 for e in entries if e["ninl"] > 0))
    for e in entries:
        if e["ninl"] <= 0:
            continue
        html.append("<div><div>%s</div><img src='%s'></div>" % (e["title"], e["rel"]))
    html.append("<h2 class='empty'>No inliers (%d)</h2>" % sum(1 for e in entries if e["ninl"] <= 0))
    for e in entries:
        if e["ninl"] > 0:
            continue
        html.append("<div class='empty'><div>%s</div><img src='%s'></div>" % (e["title"], e["rel"]))
    html.append("</body></html>")
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write("\n".join(html))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        default="/tmp/vslam_map/map/database/database.db",
    )
    parser.add_argument(
        "--images",
        default="/tmp/vslam_map/map/images",
    )
    parser.add_argument(
        "--output",
        default="/tmp/vslam_map/map/match_vis",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    con = sqlite3.connect(args.database)
    names = load_images(con)
    keypoints = load_keypoints(con)
    pairs = read_pairs(con)
    con.close()

    image_cache = {}

    def get_image(image_id):
        if image_id in image_cache:
            return image_cache[image_id]
        name = names[image_id]
        path = resolve_image_path(args.images, name)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        image_cache[image_id] = img
        return img

    entries = []
    n_empty = 0
    n_inlier = 0
    for image_id1, image_id2, raw, inliers, config in pairs:
        cfg = CONFIG_NAME.get(config, str(config))
        title = "pair %d-%d  raw=%d  inliers=%d  config=%s" % (
            image_id1, image_id2, len(raw), len(inliers), cfg)
        img1 = get_image(image_id1)
        img2 = get_image(image_id2)
        kpts1 = keypoints.get(image_id1, np.zeros((0, 2), np.float32))
        kpts2 = keypoints.get(image_id2, np.zeros((0, 2), np.float32))
        vis = draw_pair(img1, img2, kpts1, kpts2, raw, inliers, title)

        sub = "inliers" if len(inliers) > 0 else "empty"
        os.makedirs(os.path.join(args.output, sub), exist_ok=True)
        fname = "%04d_%04d_raw%03d_inl%03d_cfg%d.jpg" % (
            image_id1, image_id2, len(raw), len(inliers), config)
        rel = os.path.join(sub, fname)
        cv2.imwrite(os.path.join(args.output, rel), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        entries.append({"title": title, "rel": rel, "ninl": int(len(inliers))})
        if len(inliers) > 0:
            n_inlier += 1
        else:
            n_empty += 1

    write_index(args.output, entries)
    print("pairs=%d inliers=%d empty=%d" % (len(pairs), n_inlier, n_empty))
    print("output=%s" % args.output)
    print("open %s" % os.path.join(args.output, "index.html"))


if __name__ == "__main__":
    main()
