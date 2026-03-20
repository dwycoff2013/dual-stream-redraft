import json

from dualstream.arc_task import ArcTask, PredictionAttempt, TrainPair, load_task, load_tasks_from_dir, submission_dict_from_predictions, validate_grid


def test_load_and_validate_task(tmp_path):
    payload = {
        "train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
        "test": [{"input": [[1, 0], [0, 1]]}],
    }
    path = tmp_path / "toy_task.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    task = load_task(path)
    assert task.task_id == "toy_task"
    assert len(task.train) == 1
    assert len(task.test) == 1


def test_load_tasks_from_dir_and_submission_dict(tmp_path):
    for idx in (1, 2):
        payload = {
            "train": [{"input": [[idx]], "output": [[idx]]}],
            "test": [{"input": [[idx]]}],
        }
        (tmp_path / f"task_{idx}.json").write_text(json.dumps(payload), encoding="utf-8")

    tasks = load_tasks_from_dir(tmp_path)
    assert len(tasks) == 2

    submission = submission_dict_from_predictions(
        {tasks[0].task_id: [PredictionAttempt(attempt_1=[[1]], attempt_2=[[1]])]}
    )
    assert list(submission.keys()) == [tasks[0].task_id]


def test_validate_grid_errors():
    try:
        validate_grid([[1, 2], [3]])
    except ValueError as exc:
        assert "same width" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
