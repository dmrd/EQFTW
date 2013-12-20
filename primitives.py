import math
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import ConvexHull
from itertools import chain

EPSILON = 1e-10

# Make all plots have square aspect ratio
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')


def random_color():
    return np.random.rand(3)


def plot_poly(poly, color=random_color()):
    x, y = zip(*(poly + [poly[0]]))
    ax.fill(x, y, color=color)


def plot_polys(polys):
    for poly in polys:
        plot_poly(poly)


def plot_line(a, b):
    plt.plot((a.x, b.x), (a.y, b.y))


class Point(np.ndarray):
    def __new__(self, x, y=None):
        """ Point(x, y) or Point([x, y]) """
        if y is None:
            assert(len(x) == 2)
            vals = x
        else:
            vals = [x, y]
        obj = np.asarray(vals).view(self)
        return obj

    def __eq__(a, b):
        return a.x == b.x and a.y == b.y

    def __ne__(a, b):
        return a.x != b.x or a.y != b.y

    def __lt__(a, b):
        return (a.x, a.y) < (b.x, b.y)

    def __gt__(a, b):
        return not(a < b)

    def __ge__(a, b):
        return a == b or a > b

    def __le__(a, b):
        return a == b or a < b

    def __repr__(self):
        return "({:0.2f}, {:0.2f})".format(self.x, self.y)
        #return str((self.x, self.y))

    def plot(self):
        plt.scatter(self.x, self.y)

    def project(p, a, b):
        """ Project point p onto the line a-b """
        ab = b - a
        ap = p - a
        x = ((ap.x*ab.x + ap.y*ab.y) / float((ab.x*ab.x + ab.y*ab.y)) * ab.x)
        y = ((ap.x*ab.x + ap.y*ab.y) / float((ab.x*ab.x + ab.y*ab.y)) * ab.y)
        return a + Point(x, y)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]


class Shape:
    def __init__(self, pieces):
        self.pieces = pieces

    def __iter__(self):
        """ Iterate over pieces of the shape """
        for piece in self.pieces:
            yield piece

    def __repr__(self):
        return str(self.pieces)

    def combine(self, *others):
        return merge_shapes(chain(others, [self]))

    def cut(self, a, b):
        """
        Cut shape (and all sub pieces) with given line.
        Return two shapes, one with all pieces to left
        """
        left = []
        right = []
        for piece in self.pieces:
            l, r = piece.cut(a, b)
            if l is not None:
                left.append(l)
            if r is not None:
                right.append(r)
        return Shape(left), Shape(right)

    def rotate(self, pivot, angle):
        """ Rotate shape around pivot"""
        for piece in self.pieces:
            piece.rotate(pivot, angle)

    def translate(self, vector):
        """ Move all points by given vector"""
        for piece in self.pieces:
            piece.translate(vector)

    def apply_transform(self, transform):
        """ Apply the given transform to all pieces """
        for piece in self.pieces:
            piece.apply_transform(transform)

    def bbox(self):
        min_x = float('inf')
        min_y = min_x
        max_x = float('-inf')
        max_y = max_x

        for piece in self.pieces:
            for point in piece:
                min_x = min(min_x, point.x)
                max_x = max(max_x, point.x)
                min_y = min(min_y, point.y)
                max_y = max(max_y, point.y)
        return (min_x, min_y, max_x, max_y)

    def hull(self):
        points = []
        for piece in self.pieces:
            points.extend(piece.poly)
        points = np.asarray(points)
        return make_poly(reversed(points[ConvexHull(points).vertices]))

    def plot(self, shift=None):
        for piece in self.pieces:
            piece.plot(shift=shift)

    def original_position(self):
        return Shape([piece.original_position() for piece in self.pieces])


class Piece:
    def __init__(self, poly, transform=None, color=None):
        if transform is None:
            transform = identity_transform()
        if color is None:
            color = random_color()
        self.transform = transform
        self.poly = make_poly(poly)
        self.color = color

    def __iter__(self):
        """ Iterate over points of the piece """
        for p in self.poly:
            yield p

    def __repr__(self):
        return str(self.poly)

    def plot(self, shift=None):
        if shift is not None:
            poly = [p + shift for p in self.poly]
        else:
            poly = self.poly
        return plot_poly(poly, color=self.color)

    def translate(self, vector):
        self.apply_transform(translation_matrix(vector))

    def rotate(self, pivot, radians):
        center = translation_matrix(-pivot)
        rotate = rotation_matrix(radians)
        uncenter = translation_matrix(pivot)
        self.apply_transform(np.dot(np.dot(uncenter, rotate), center))

    def apply_transform(self, transform):
        self.transform = np.dot(transform, self.transform)
        for i, point in enumerate(self.poly):
            self.poly[i] = transform_point(point, transform)

    def inverse_transform(self):
        return np.linalg.inv(self.transform)

    def cut(self, a, b):
        result = split_convex_poly(self.poly, a, b)
        if result is None:
            return self

        # Split pieces have same transform so far
        p1 = result[0] and Piece(result[0], self.transform)
        p2 = result[1] and Piece(result[1], self.transform)

        return p1, p2

    def original_position(self):
        inverse = self.inverse_transform()
        return Piece(make_poly(transform_point(p, inverse) for p in self.poly), color=self.color)


def merge_shapes(shapes):
    pieces = []
    for shape in shapes:
        if shape is not None:
            pieces.extend(shape.pieces)
    return Shape(pieces)


def distance(a, b):
    return np.linalg.norm(b - a)


def transform_point(p, transform):
    return Point(np.dot(transform, [p.x, p.y, 1])[:2])


def translation_matrix(vector):
    return np.array([[1, 0, vector[0]],
                     [0, 1, vector[1]],
                     [0, 0, 1]])


def rotation_matrix(radians):
    sa = math.sin(radians)
    ca = math.cos(radians)
    return np.array([[ca, sa, 0],
                     [-sa, ca, 0],
                     [0, 0, 1]])


def identity_transform():
    return np.identity(3)


def extend_line(a, b):
    """ Extend a line past both its endpoints """
    v = b - a
    return (a - v), (b + v)


def make_poly(points):
    return [Point(p) for p in points]


def dot(a, b):
    return sum(a * b)


def cross(a, b):
    return float(np.cross(a, b))


def length(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    return v / length(v)


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


def tri_area(a, b, c):
    return ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x))


def ccw(a, b, c):
    """
    Check if a,b are same point but c could be different
    Behavior unspecified when b == c and a != b, so
    using -2 since it is opposite of a == b != c.
    """
    if a == b == c:
        return 0
    elif a == b != c:
        return 2
    elif a != b == c:
        return -2

    # Cross product
    cross = tri_area(a, b, c)
    if cross > 0:
        # Left turn
        return -1
    elif cross < 0:
        # Right turn
        return 1
    else:
        # When sorted by (x,y) coordinates, the point in the middle
        # geometrically will always be in the middle of the list
        # Sorintg 3 numbers considered constant time
        s = sorted([a, b, c])
        if s[1] is c:
            # a -- c -- b
            # b -- c -- a
            return 0
        elif s[1] is a:
            # b -- a -- c
            # c -- a -- b
            return -2
        else:
            # a -- b -- c
            return 2


def polygon_ccw(poly, a, b):
    """
    Returns True if polygon is right turn from a->b, False otherwise

    Assumes no intersections (except at endpoints)
    """
    return all(ccw(a, b, p) >= 0 for p in poly)


def intersect(p, pr, q, qs):
    """
    Intersect two lines specified by endpoints (p, pe) and (q, qe).
    Returns a tuple of (parametric distance along p-pr, intersection point)

    Ignored endpoint intersections except with q
    http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    """
    r = pr - p
    s = qs - q

    rxs = cross(r, s)

    # Parallel, count as no intersection
    if rxs == 0:
        return None

    u = cross(q - p, r) / rxs
    t = cross(q - p, s) / rxs
    if u < 0 or u >= 1 or t <= 0 or t >= 1:
        return None

    return (t, p + t * r)


def split_convex_poly(poly, a, b):
    """
    Split the given convex polygon with a line a->b
    Returns two new polygons (left turn, right turn) from line

    If the line does not intersect, it returns tuple with polygon in correct
    ccw position and None in other position
    """
    hits = []
    for i in range(len(poly)):
        hit = intersect(a, b, poly[i], poly[(i+1) % len(poly)])
        if hit is not None:
            hits.append((i, hit[1], hit[0]))
    # At most 2 intersections for convex polygon and line
    if len(hits) > 2:
        # Numerical errors
        for i in range(len(hits)):
            if distance(hits[i][1], hits[(i+1) % len(hits)][1]) < EPSILON:
                print("Numerical error in split")
                del hits[i]
    assert(len(hits) <= 2)

    # Polygon lies on one side of the line
    if len(hits) <= 1:
        if polygon_ccw(poly, a, b):
            return None, poly
        return poly, None

    # Polygon split in two
    l1, h1, t1 = hits[0]
    l2, h2, t2 = hits[1]
    if poly[l2] == h2:
        p1_new = [h1]
    else:
        p1_new = [h2, h1]

    if poly[l1] == h1:
        p2_new = [h2]
    else:
        p2_new = [h1, h2]

    p1 = poly[l1 + 1:l2+1] + p1_new
    p2 = poly[l2 + 1:] + poly[:l1+1] + p2_new

    # Determine which is left and right
    if t1 < t2:
        return p1, p2
    else:
        return p2, p1
