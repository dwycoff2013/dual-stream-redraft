from __future__ import annotations

from collections import Counter, deque
import hashlib
from typing import Iterable

import numpy as np

GridArray = np.ndarray


def to_array(grid: list[list[int]] | GridArray) -> GridArray:
    if isinstance(grid, np.ndarray):
        return grid.astype(np.int64, copy=False)
    return np.asarray(grid, dtype=np.int64)


def to_grid(grid: list[list[int]] | GridArray) -> list[list[int]]:
    return to_array(grid).tolist()


def shape_of(grid: list[list[int]] | GridArray) -> tuple[int, int]:
    arr = to_array(grid)
    return int(arr.shape[0]), int(arr.shape[1])


def color_histogram(grid: list[list[int]] | GridArray) -> dict[int, int]:
    arr = to_array(grid)
    counts = Counter(int(v) for v in arr.flatten())
    return dict(sorted(counts.items()))


def bounding_box_of_color(grid: list[list[int]] | GridArray, color: int) -> tuple[int, int, int, int] | None:
    arr = to_array(grid)
    ys, xs = np.where(arr == color)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def crop(grid: list[list[int]] | GridArray, top: int, left: int, height: int, width: int) -> GridArray:
    arr = to_array(grid)
    return arr[top : top + height, left : left + width]


def pad(grid: list[list[int]] | GridArray, top: int, bottom: int, left: int, right: int, fill: int = 0) -> GridArray:
    arr = to_array(grid)
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant", constant_values=fill)


def paste(base: list[list[int]] | GridArray, patch: list[list[int]] | GridArray, top: int, left: int) -> GridArray:
    arr = to_array(base).copy()
    pat = to_array(patch)
    h, w = pat.shape
    arr[top : top + h, left : left + w] = pat
    return arr


def rotate90(grid: list[list[int]] | GridArray) -> GridArray:
    return np.rot90(to_array(grid), k=1)


def rotate180(grid: list[list[int]] | GridArray) -> GridArray:
    return np.rot90(to_array(grid), k=2)


def rotate270(grid: list[list[int]] | GridArray) -> GridArray:
    return np.rot90(to_array(grid), k=3)


def reflect_horizontal(grid: list[list[int]] | GridArray) -> GridArray:
    return np.fliplr(to_array(grid))


def reflect_vertical(grid: list[list[int]] | GridArray) -> GridArray:
    return np.flipud(to_array(grid))


def translate(grid: list[list[int]] | GridArray, dy: int, dx: int, fill: int = 0) -> GridArray:
    arr = to_array(grid)
    out = np.full_like(arr, fill)
    h, w = arr.shape

    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy) if dy >= 0 else h
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx) if dx >= 0 else w
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]
    return out


def tile_pattern(grid: list[list[int]] | GridArray, reps_y: int, reps_x: int) -> GridArray:
    return np.tile(to_array(grid), (reps_y, reps_x))


def grids_equal(a: list[list[int]] | GridArray, b: list[list[int]] | GridArray) -> bool:
    aa = to_array(a)
    bb = to_array(b)
    return aa.shape == bb.shape and bool(np.array_equal(aa, bb))


def pixel_accuracy(a: list[list[int]] | GridArray, b: list[list[int]] | GridArray) -> float:
    aa = to_array(a)
    bb = to_array(b)
    if aa.shape != bb.shape:
        return 0.0
    if aa.size == 0:
        return 1.0
    return float((aa == bb).sum() / aa.size)


def diff_pixels(a: list[list[int]] | GridArray, b: list[list[int]] | GridArray) -> int:
    aa = to_array(a)
    bb = to_array(b)
    if aa.shape != bb.shape:
        return max(aa.size, bb.size)
    return int((aa != bb).sum())


def hash_grid(grid: list[list[int]] | GridArray) -> str:
    arr = to_array(grid)
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode("utf-8")).hexdigest()


def _neighbors(y: int, x: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            yield ny, nx


def connected_components(grid: list[list[int]] | GridArray, *, background: int = 0) -> list[dict[str, object]]:
    arr = to_array(grid)
    h, w = arr.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for y in range(h):
        for x in range(w):
            if visited[y, x] or arr[y, x] == background:
                continue
            color = int(arr[y, x])
            q = deque([(y, x)])
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []

            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))
                for ny, nx in _neighbors(cy, cx, h, w):
                    if visited[ny, nx] or int(arr[ny, nx]) != color:
                        continue
                    visited[ny, nx] = True
                    q.append((ny, nx))

            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            components.append(
                {
                    "color": color,
                    "pixels": pixels,
                    "size": len(pixels),
                    "bbox": (min(ys), min(xs), max(ys), max(xs)),
                }
            )

    return components


def extract_color(grid: list[list[int]] | GridArray, color: int, *, background: int = 0) -> GridArray:
    arr = to_array(grid)
    out = np.full_like(arr, background)
    out[arr == color] = color
    return out


def keep_largest_object(grid: list[list[int]] | GridArray, *, background: int = 0) -> GridArray:
    arr = to_array(grid)
    comps = connected_components(arr, background=background)
    if not comps:
        return np.full_like(arr, background)
    best = max(comps, key=lambda c: int(c["size"]))
    out = np.full_like(arr, background)
    color = int(best["color"])
    for y, x in best["pixels"]:  # type: ignore[index]
        out[y, x] = color
    return out


def keep_smallest_object(grid: list[list[int]] | GridArray, *, background: int = 0) -> GridArray:
    arr = to_array(grid)
    comps = connected_components(arr, background=background)
    if not comps:
        return np.full_like(arr, background)
    best = min(comps, key=lambda c: int(c["size"]))
    out = np.full_like(arr, background)
    color = int(best["color"])
    for y, x in best["pixels"]:  # type: ignore[index]
        out[y, x] = color
    return out


def crop_to_non_background_bbox(grid: list[list[int]] | GridArray, *, background: int = 0) -> GridArray:
    arr = to_array(grid)
    ys, xs = np.where(arr != background)
    if len(ys) == 0:
        return arr.copy()
    return arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
