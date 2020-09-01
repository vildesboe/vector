from vector import Vector3D

def test_vector_str():
    v = Vector3D(1, 2, 3)
    assert str(v) == "(1, 2, 3"

def test_vector_repr():
    v = Vector3D(1, 2, 3)
    assert repr(v) == "Vector3D(1, 2, 3"


def test_vector_add():
    v = Vector3D(1, 2, 3)
    u = Vector3D(1, 1, 1)
    w = u+v
    assert w.x == 2
    assert w.y == 3
    assert w.z == 4
    assert w == Vector3D(2, 3, 4)