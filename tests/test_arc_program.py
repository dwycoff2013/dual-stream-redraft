from dualstream.arc_program import Compose, Identity, ReflectHorizontal, RecolorMap, Rotate90


def test_atomic_programs():
    grid = [[1, 2], [3, 4]]
    assert Identity().apply(grid) == grid
    assert Rotate90().apply(grid) == [[2, 4], [1, 3]]
    assert ReflectHorizontal().apply(grid) == [[2, 1], [4, 3]]
    assert RecolorMap({1: 9}).apply([[1, 0]]) == [[9, 0]]


def test_program_composition():
    grid = [[1, 2], [3, 4]]
    prog = Compose((Rotate90(), ReflectHorizontal()))
    assert prog.apply(grid) == [[4, 2], [3, 1]]
