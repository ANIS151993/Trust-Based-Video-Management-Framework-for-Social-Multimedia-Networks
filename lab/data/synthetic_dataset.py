"""Synthetic silhouette-proxy dataset generator.

IMPORTANT: this generates procedural line-art silhouettes standing in for the
five weapon categories used in the paper (AK-47, gun, knife, sickle, sword).
It contains NO real weapon photography. It exists so that anyone cloning the
public repository can reproduce the training/evaluation pipeline end-to-end
without needing a sensitive or license-restricted image dataset -- it is a
validation testbed for the framework's ML pipeline (transfer learning setup,
training loop, evaluation methodology), not a substitute for a real-world
weapon-imagery benchmark.

Each class is defined by a parametric "skeleton" (a set of line strokes and
filled shapes in a normalized [-1, 1] canvas). Every generated sample applies
random rotation, scale, translation, per-vertex jitter, stroke-width and
color jitter, a randomized background, distractor strokes, and mild
blur/noise -- enough visual variability that the classification task is
non-trivial rather than a trivial memorization exercise.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASSES, DATA_DIR, RANDOM_SEED  # noqa: E402

CANVAS = 160


def _skeleton_ak47():
    # Long body (barrel + receiver), angled stock, curved magazine.
    body = [(-0.85, 0.05), (0.55, -0.05)]
    stock = [(0.55, -0.05), (0.9, 0.35)]
    magazine = [(-0.05, 0.05), (0.05, 0.15), (0.15, 0.55), (0.05, 0.6)]
    grip = [(0.35, 0.0), (0.42, 0.35)]
    return {"lines": [body, stock, magazine, grip], "fill": [], "width": 0.05}


def _skeleton_gun():
    # L-shaped handgun: barrel + grip.
    barrel = [(-0.75, -0.15), (0.55, -0.15)]
    slide_top = [(-0.75, -0.28), (0.55, -0.28)]
    grip = [(0.35, -0.1), (0.55, 0.7), (0.75, 0.68), (0.55, -0.15)]
    trigger_guard = [(0.15, -0.1), (0.05, 0.25), (0.3, 0.3), (0.35, -0.05)]
    return {"lines": [barrel, slide_top, trigger_guard], "fill": [grip], "width": 0.06}


def _skeleton_knife():
    # Triangular blade + rectangular handle.
    blade = [(-0.85, 0.0), (0.35, -0.25), (0.35, 0.25), (-0.85, 0.0)]
    handle = [(0.35, -0.15), (0.85, -0.15), (0.85, 0.15), (0.35, 0.15)]
    return {"lines": [], "fill": [blade, handle], "width": 0.04}


def _skeleton_sickle():
    # Large curved arc (Bezier-ish via many points) + short handle.
    pts = []
    for t in np.linspace(0.15 * math.pi, 1.6 * math.pi, 24):
        r = 0.6
        pts.append((r * math.cos(t), r * math.sin(t) - 0.1))
    handle = [pts[0], (pts[0][0] - 0.15, pts[0][1] + 0.35)]
    return {"lines": [pts, handle], "fill": [], "width": 0.05}


def _skeleton_sword():
    # Long thin blade, crossguard, handle/pommel.
    blade = [(-0.9, 0.0), (0.55, -0.06), (0.55, 0.06), (-0.9, 0.0)]
    crossguard = [(0.5, -0.25), (0.62, 0.25)]
    handle = [(0.62, -0.06), (0.9, -0.06), (0.9, 0.06), (0.62, 0.06)]
    pommel_center = (0.92, 0.0)
    return {
        "lines": [crossguard],
        "fill": [blade, handle],
        "circles": [(pommel_center, 0.05)],
        "width": 0.045,
    }


SKELETONS = {
    "ak47": _skeleton_ak47,
    "gun": _skeleton_gun,
    "knife": _skeleton_knife,
    "sickle": _skeleton_sickle,
    "sword": _skeleton_sword,
}


def _jitter_point(p, jitter, rng):
    return (p[0] + rng.uniform(-jitter, jitter), p[1] + rng.uniform(-jitter, jitter))


def _transform(points, angle, scale, tx, ty):
    ca, sa = math.cos(angle), math.sin(angle)
    out = []
    for x, y in points:
        xr = x * ca - y * sa
        yr = x * sa + y * ca
        out.append((xr * scale + tx, yr * scale + ty))
    return out


def _to_canvas(points, size):
    c = size / 2
    return [(c + x * c * 0.85, c + y * c * 0.85) for x, y in points]


def _random_background(size, rng):
    base = rng.randint(200, 250)
    img = Image.new("RGB", (size, size), (base, base - rng.randint(0, 15), base - rng.randint(0, 10)))
    draw = ImageDraw.Draw(img)
    # A few faint distractor strokes so the model can't rely on "blank vs. shape" shortcuts.
    for _ in range(rng.randint(0, 3)):
        x1, y1, x2, y2 = [rng.randint(0, size) for _ in range(4)]
        shade = rng.randint(max(base - 60, 0), base - 10)
        draw.line((x1, y1, x2, y2), fill=(shade, shade, shade), width=rng.randint(1, 2))
    noise = (np.random.RandomState(rng.randint(0, 1 << 30)).normal(0, 6, (size, size, 3))).astype(np.int16)
    arr = np.clip(np.array(img).astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def render_sample(cls: str, seed: int, size: int = CANVAS) -> Image.Image:
    rng = random.Random(seed)
    skeleton = SKELETONS[cls]()
    angle = rng.uniform(-0.5, 0.5)
    scale = rng.uniform(0.75, 1.05)
    tx = rng.uniform(-0.12, 0.12)
    ty = rng.uniform(-0.12, 0.12)
    jitter = 0.02

    img = _random_background(size, rng)
    draw = ImageDraw.Draw(img)
    stroke_shade = rng.randint(20, 70)
    color = (stroke_shade, stroke_shade, stroke_shade)
    width_px = max(2, int(skeleton["width"] * size * rng.uniform(0.8, 1.2)))

    for line in skeleton["lines"]:
        jittered = [_jitter_point(p, jitter, rng) for p in line]
        transformed = _transform(jittered, angle, scale, tx, ty)
        canvas_pts = _to_canvas(transformed, size)
        draw.line(canvas_pts, fill=color, width=width_px, joint="curve")

    for poly in skeleton.get("fill", []):
        jittered = [_jitter_point(p, jitter, rng) for p in poly]
        transformed = _transform(jittered, angle, scale, tx, ty)
        canvas_pts = _to_canvas(transformed, size)
        draw.polygon(canvas_pts, fill=color)

    for center, radius in skeleton.get("circles", []):
        cx, cy = _transform([center], angle, scale, tx, ty)[0]
        cx, cy = _to_canvas([(cx, cy)], size)[0]
        r = radius * scale * size * 0.85
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)

    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.9)))

    return img


def generate_dataset(per_class: int = 120, size: int = CANVAS, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_counter = 0
    for cls in CLASSES:
        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            seed = RANDOM_SEED * 100003 + seed_counter
            seed_counter += 1
            img = render_sample(cls, seed, size)
            img.save(cls_dir / f"{cls}_{i:04d}.png")
    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the synthetic weapon-silhouette proxy dataset.")
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--size", type=int, default=CANVAS)
    args = parser.parse_args()
    path = generate_dataset(args.per_class, args.size)
    total = args.per_class * len(CLASSES)
    print(f"Generated {total} images ({args.per_class}/class) into {path}")
