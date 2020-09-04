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
    w = u + v
    assert w == Vector3D(2, 3, 4)


def test_vector_add_integer_right():
    v = Vector3D(1, 2, 3)
    u = 1
    w = u + v
    assert w == Vector3D(2, 3, 4)


import pytest


def test_vector_add_str_raises_TypeError():
    v = Vector3D(1, 2, 3)
    u = "Hello"
    with pytest.raises(TypeError):
        u + v


def test_vector_dot_product():
    v = Vector3D(1, 2, 3)
    u = Vector3D(1, 1, 1)
    w = u.dot(v)
    assert isinstance(w, (int, float))
    assert abs(w - 6) < 1e-12


def test_vector_dot_produc_mul():
    v = Vector3D(1, 2, 3)
    u = Vector3D(1, 1, 1)
    w = u * v
    assert isinstance(w, (int, float))
    assert abs(w - 6) < 1e-12


@pytest.mark.parametrize(
    "u, v, exp",
    [
        (Vector3D(0, 1, 0), Vector3D(1, 0, 0), Vector3D(0, 0, -1)),
        (Vector3D(2, 0, -2), Vector3D(2, 4, 2), Vector3D(8, -8, 8)),
    ],
)
def test_vector_cross_product(u, v, exp):
    w = u.cross(v)
    assert w == exp


@pytest.mark.parametrize(
    "u, v, exp",
    [
        (Vector3D(0, 1, 0), Vector3D(1, 0, 0), Vector3D(0, 0, -1)),
        (Vector3D(2, 0, -2), Vector3D(2, 4, 2), Vector3D(8, -8, 8)),
    ],
)
def test_vector_cross_product_matmul(u, v, exp):
    w = u @ v
    assert w == exp