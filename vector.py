class Vector3D:
    def __init__(self, x, y, z):
        self.x=x
        self.y=y
        self.z=z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z}"

if __name__ == "__main__":
    v=Vector3D(1, 4, 1)
    print(v)
