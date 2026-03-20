from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from . import arc_grid


class Program(Protocol):
    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        ...

    def to_dict(self) -> dict[str, object]:
        ...

    def stable_repr(self) -> str:
        ...

    def complexity_cost(self) -> float:
        ...


@dataclass(frozen=True)
class Identity:
    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        return arc_grid.to_grid(grid)

    def to_dict(self) -> dict[str, object]:
        return {"type": "Identity"}

    def stable_repr(self) -> str:
        return "Identity"

    def complexity_cost(self) -> float:
        return 0.0


@dataclass(frozen=True)
class RecolorMap:
    mapping: dict[int, int]

    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        arr = arc_grid.to_array(grid).copy()
        for src, dst in self.mapping.items():
            arr[arr == int(src)] = int(dst)
        return arr.tolist()

    def to_dict(self) -> dict[str, object]:
        return {"type": "RecolorMap", "mapping": dict(self.mapping)}

    def stable_repr(self) -> str:
        items = ",".join(f"{k}->{v}" for k, v in sorted(self.mapping.items()))
        return f"RecolorMap({items})"

    def complexity_cost(self) -> float:
        return 0.2 + 0.05 * len(self.mapping)


@dataclass(frozen=True)
class Rotate90:
    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        return arc_grid.rotate90(grid).tolist()

    def to_dict(self) -> dict[str, object]:
        return {"type": "Rotate90"}

    def stable_repr(self) -> str:
        return "Rotate90"

    def complexity_cost(self) -> float:
        return 0.1


@dataclass(frozen=True)
class Rotate180:
    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        return arc_grid.rotate180(grid).tolist()

    def to_dict(self) -> dict[str, object]:
        return {"type": "Rotate180"}

    def stable_repr(self) -> str:
        return "Rotate180"

    def complexity_cost(self) -> float:
        return 0.1


@dataclass(frozen=True)
class Rotate270:
    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        return arc_grid.rotate270(grid).tolist()

    def to_dict(self) -> dict[str, object]:
        return {"type": "Rotate270"}

    def stable_repr(self) -> str:
        return "Rotate270"

    def complexity_cost(self) -> float:
        return 0.1


@dataclass(frozen=True)
class ReflectHorizontal:
    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        return arc_grid.reflect_horizontal(grid).tolist()

    def to_dict(self) -> dict[str, object]:
        return {"type": "ReflectHorizontal"}

    def stable_repr(self) -> str:
        return "ReflectHorizontal"

    def complexity_cost(self) -> float:
        return 0.1


@dataclass(frozen=True)
class ReflectVertical:
    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        return arc_grid.reflect_vertical(grid).tolist()

    def to_dict(self) -> dict[str, object]:
        return {"type": "ReflectVertical"}

    def stable_repr(self) -> str:
        return "ReflectVertical"

    def complexity_cost(self) -> float:
        return 0.1


@dataclass(frozen=True)
class CropToBoundingBox:
    background: int = 0

    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        return arc_grid.crop_to_non_background_bbox(grid, background=self.background).tolist()

    def to_dict(self) -> dict[str, object]:
        return {"type": "CropToBoundingBox", "background": self.background}

    def stable_repr(self) -> str:
        return f"CropToBoundingBox(bg={self.background})"

    def complexity_cost(self) -> float:
        return 0.15


@dataclass(frozen=True)
class FilterColor:
    color: int
    background: int = 0

    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        return arc_grid.extract_color(grid, self.color, background=self.background).tolist()

    def to_dict(self) -> dict[str, object]:
        return {"type": "FilterColor", "color": self.color, "background": self.background}

    def stable_repr(self) -> str:
        return f"FilterColor({self.color})"

    def complexity_cost(self) -> float:
        return 0.2


@dataclass(frozen=True)
class KeepLargestObject:
    background: int = 0

    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        return arc_grid.keep_largest_object(grid, background=self.background).tolist()

    def to_dict(self) -> dict[str, object]:
        return {"type": "KeepLargestObject", "background": self.background}

    def stable_repr(self) -> str:
        return "KeepLargestObject"

    def complexity_cost(self) -> float:
        return 0.3


@dataclass(frozen=True)
class KeepSmallestObject:
    background: int = 0

    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        return arc_grid.keep_smallest_object(grid, background=self.background).tolist()

    def to_dict(self) -> dict[str, object]:
        return {"type": "KeepSmallestObject", "background": self.background}

    def stable_repr(self) -> str:
        return "KeepSmallestObject"

    def complexity_cost(self) -> float:
        return 0.3


@dataclass(frozen=True)
class TranslateWholeGrid:
    dy: int
    dx: int
    fill: int = 0

    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        return arc_grid.translate(grid, self.dy, self.dx, fill=self.fill).tolist()

    def to_dict(self) -> dict[str, object]:
        return {"type": "TranslateWholeGrid", "dy": self.dy, "dx": self.dx, "fill": self.fill}

    def stable_repr(self) -> str:
        return f"TranslateWholeGrid(dy={self.dy},dx={self.dx},fill={self.fill})"

    def complexity_cost(self) -> float:
        return 0.25


@dataclass(frozen=True)
class TilePattern:
    reps_y: int
    reps_x: int

    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        return arc_grid.tile_pattern(grid, self.reps_y, self.reps_x).tolist()

    def to_dict(self) -> dict[str, object]:
        return {"type": "TilePattern", "reps_y": self.reps_y, "reps_x": self.reps_x}

    def stable_repr(self) -> str:
        return f"TilePattern({self.reps_y}x{self.reps_x})"

    def complexity_cost(self) -> float:
        return 0.35


@dataclass(frozen=True)
class Compose:
    steps: tuple[Program, ...]

    def apply(self, grid: list[list[int]]) -> list[list[int]]:
        out = grid
        for step in self.steps:
            out = step.apply(out)
        return out

    def to_dict(self) -> dict[str, object]:
        return {"type": "Compose", "steps": [s.to_dict() for s in self.steps]}

    def stable_repr(self) -> str:
        return "Compose(" + " -> ".join(s.stable_repr() for s in self.steps) + ")"

    def complexity_cost(self) -> float:
        return 0.05 + sum(step.complexity_cost() for step in self.steps)
