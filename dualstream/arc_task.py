from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

Grid = list[list[int]]


@dataclass(frozen=True)
class TrainPair:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class ArcTask:
    task_id: str
    train: list[TrainPair]
    test: list[Grid]


@dataclass(frozen=True)
class PredictionAttempt:
    attempt_1: Grid
    attempt_2: Grid

    def to_dict(self) -> dict[str, Any]:
        return {"attempt_1": self.attempt_1, "attempt_2": self.attempt_2}


def validate_grid(grid: Grid) -> None:
    if not isinstance(grid, list) or not grid:
        raise ValueError("Grid must be a non-empty list of rows")
    if not all(isinstance(row, list) for row in grid):
        raise ValueError("Grid rows must be lists")
    width = len(grid[0])
    if width == 0:
        raise ValueError("Grid rows must be non-empty")
    for row in grid:
        if len(row) != width:
            raise ValueError("Grid rows must all have the same width")
        for value in row:
            if not isinstance(value, int) or value < 0 or value > 9:
                raise ValueError("Grid cells must be ints in [0, 9]")


def validate_task(task: ArcTask) -> None:
    if not task.task_id:
        raise ValueError("task_id must be non-empty")
    if not task.train:
        raise ValueError("ARC task requires at least one train pair")
    if not task.test:
        raise ValueError("ARC task requires at least one test input")
    for pair in task.train:
        validate_grid(pair.input)
        validate_grid(pair.output)
    for grid in task.test:
        validate_grid(grid)


def _task_id_from_path(path: Path) -> str:
    return path.stem


def load_task(path: str | Path) -> ArcTask:
    task_path = Path(path)
    payload = json.loads(task_path.read_text(encoding="utf-8"))

    train = [
        TrainPair(input=item["input"], output=item["output"])
        for item in payload.get("train", [])
    ]
    test = [item["input"] for item in payload.get("test", [])]

    task = ArcTask(task_id=_task_id_from_path(task_path), train=train, test=test)
    validate_task(task)
    return task


def load_tasks_from_dir(path: str | Path) -> list[ArcTask]:
    root = Path(path)
    tasks: list[ArcTask] = []
    for task_file in sorted(root.glob("*.json")):
        tasks.append(load_task(task_file))
    return tasks


def submission_dict_from_predictions(predictions: dict[str, list[PredictionAttempt]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task_id, attempts in predictions.items():
        out[task_id] = [item.to_dict() for item in attempts]
    return out
