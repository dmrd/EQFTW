import math
import matplotlib.pylab as plt


def plot_poly(poly):
    x, y = zip(*(poly + [poly[0]]))
    return plt.plot(x, y)


def plot_polys(polys):
    for poly in polys:
        plot_poly(poly)


class Point(list):
    def __add__(a, b):
        return Point(a[i] + b[i] for i in range(len(a)))

    def __sub__(a, b):
        return Point(a[i] - b[i] for i in range(len(a)))

    def __mul__(a, b):
        if type(a) == type(b):
            return Point(a[i] * b[i] for i in range(len(a)))
        else:
            return Point(v * b for v in a)

    def __div__(a, b):
        if type(a) == type(b):
            return Point(a[i] / b[i] for i in range(len(a)))
        else:
            return Point(v / b for v in a)

    def __rmul__(a, b):
        return a.__mul__(b)

    def abs(a):
        return tuple(abs(v) for v in a)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]


def dot(a, b):
    return sum(a * b)


def length(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    return v / length(v)


def project(p, a, b):
    """ Project point p onto the line a-b """
    ab = b - a
    ap = p - a
    x = ((ap.x*ab.x + ap.y*ab.y) / (ab.x*ab.x + ab.y*ab.y)) * ab.x
    y = ((ap.x*ab.x + ap.y*ab.y) / (ab.x*ab.x + ab.y*ab.y)) * ab.y
    return Point([x, y]) + a


def rotate_polygon(poly, pivot, degree):
    """ Rotates given polygon around the pivot """
    r = math.radians(degree)
    new_positions = []
    for p in poly:
        p = p - pivot  # Make pivot origin

        pos = Point((math.cos(r) * p[0] + math.sin(r) * p[1],
                     math.cos(r) * p[1] - math.sin(r) * p[0]))
        new_positions.append(pos + pivot)
    return new_positions


def tri2rect(poly):
    assert(len(poly) == 3)
    # Midpoint along 2 edges
    h1 = (poly[0] + poly[1]) / 2.0
    h2 = (poly[2] + poly[1]) / 2.0

    # Create two parts from top of triangle
    mid = project(poly[1], h1, h2)
    left_tri = [h1, poly[1], mid]
    right_tri = [poly[1], h2, mid]
    # Rotate into corners of rectangle
    left_rot = rotate_polygon(left_tri, h1, 180)
    right_rot = rotate_polygon(right_tri, h2, 180)

    quad = [poly[0], h1, h2, poly[2]]
    rect = [poly[0], left_rot[2], right_rot[2], poly[2]],
    return {'rect': rect,
            'inner_quad': quad,
            'left_tri': left_tri,
            'right_tri': right_tri,
            'left_rot': left_rot,
            'right_rot': right_rot}


def rect2square(poly):
    assert(len(poly) == 4)
    pass


def combine_squares(squares):
    assert(all(len(poly) == 4 for poly in squares))
    pass


def triangulate(poly):
    pass

square = [Point(p) for p in [(0, 0), (0, 1), (1, 1), (1, 0)]]
tri = [Point(p) for p in [(-1, 0), (0, 1), (1, 0)]]
