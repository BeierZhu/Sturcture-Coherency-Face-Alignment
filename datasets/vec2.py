#!/usr/bin/python3
import math

class Vec2:
    def __init__(self, x=0., y=0.):
        self.x, self.y = x, y

    def __call__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __truediv__(self, other):
        thr = 1e-5
        if isinstance(other, Vec2):
            if abs(other.x) < thr: other.x = thr
            if abs(other.y) < thr: other.y = thr
            return Vec2(self.x / other.x, self.y / other.y)
        else:
            if abs(other) < thr: other = thr
            return Vec2(self.x / other, self.y / other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, Vec2):
            return Vec2(self.x * other.x, self.y * other.y)
        else:
            return Vec2(self.x * other, self.y * other)

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

def distance(p1, p2):
    assert isinstance(p1, Vec2) and isinstance(p2, Vec2), "argument should be Vec2"
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
