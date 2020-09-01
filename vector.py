class Vector3D:
    def __init__(self, x, y, z):
        self.x=x
        self.y=y
        self.z=z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x}, {self.y}, {self.z}"


    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Vector3D(self.x+other, self.y+other, self.z+other)
        elif isinstance(other, Vector3D):
            w = Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
            return w
        else:
            raise TypeError(f"Cannot add Vector3D to object of type {type(other)}")

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z

if __name__ == "__main__":
    v=Vector3D(1, 4, 2)
    v.dot(1)
    #u = Vector3D(1, 1, 1)
    #w=v+u
    #w == Vector3D(2, 5, 4)
    
