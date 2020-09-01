from vector import Vector3D

def test_vector_str():
    v = Vector3D(1, 2, 3)
    assert str(v) == "(1, 2, 3"