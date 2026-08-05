from dualstream import arc_grid


def test_grid_geometry_and_hashing():
    g = [[1, 0], [0, 1]]
    assert arc_grid.shape_of(g) == (2, 2)
    assert arc_grid.color_histogram(g) == {0: 2, 1: 2}
    assert arc_grid.hash_grid(g) == arc_grid.hash_grid(g)


def test_rotation_reflection_translate():
    g = [[1, 2], [3, 4]]
    assert arc_grid.rotate90(g).tolist() == [[2, 4], [1, 3]]
    assert arc_grid.rotate180(g).tolist() == [[4, 3], [2, 1]]
    assert arc_grid.reflect_horizontal(g).tolist() == [[2, 1], [4, 3]]
    assert arc_grid.translate(g, 1, 0, fill=0).tolist() == [[0, 0], [1, 2]]


def test_connected_components_and_objects():
    g = [
        [1, 0, 2],
        [1, 0, 0],
        [0, 0, 2],
    ]
    comps = arc_grid.connected_components(g, background=0)
    assert len(comps) == 3
    assert arc_grid.keep_largest_object(g, background=0).tolist() == [[1, 0, 0], [1, 0, 0], [0, 0, 0]]
