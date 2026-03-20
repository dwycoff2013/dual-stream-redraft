from dualstream.arc_solver import ArcSolver, SolverConfig
from dualstream.arc_task import ArcTask, TrainPair


def _rotation_task() -> ArcTask:
    return ArcTask(
        task_id="rotate_task",
        train=[
            TrainPair(input=[[1, 0], [0, 0]], output=[[0, 0], [1, 0]]),
            TrainPair(input=[[0, 2], [0, 0]], output=[[2, 0], [0, 0]]),
        ],
        test=[[[3, 0], [0, 0]]],
    )


def test_solver_end_to_end_and_trace_emission():
    solver = ArcSolver(SolverConfig())
    result = solver.solve_task(_rotation_task())

    assert len(result.attempts) == 1
    assert "attempt_1" in result.attempts[0].to_dict()
    assert "attempt_2" in result.attempts[0].to_dict()
    assert len(result.frames) >= 3
    assert any(f.decision_type == "hypothesis_select" for f in result.frames)
    assert any(f.decision_type == "attempt_finalize" for f in result.frames)


def test_attempt_diversity_when_possible():
    solver = ArcSolver(SolverConfig())
    result = solver.solve_task(_rotation_task())
    a1 = result.attempts[0].attempt_1
    a2 = result.attempts[0].attempt_2
    assert a1 != [] and a2 != []
