import math
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from scipy.spatial import ConvexHull
from itertools import chain

EPSILON = 1e-9


def random_color():
    return np.random.rand(3)


def show(size=(12, 12)):
    # Make all plots have square aspect ratio
    plt.axis('equal')
    # Make high quality
    fig = plt.gcf()
    fig.set_size_inches(*size)
    plt.show()


def plot_poly(poly, color=random_color()):
    plt.axis('equal')
    x, y = zip(*(poly + [poly[0]]))
    #return plt.fill(x, y, color=color)
    return plt.plot(x, y, color=color)


def plot_polys(polys):
    for poly in polys:
        plot_poly(poly)


def plot_line(a, b):
    plt.plot((a.x, b.x), (a.y, b.y))


class inverse_draw:
    def __init__(self, shape, frames):
        self.shape = shape
        self.frames = frames

    def __call__(self, i):
        return self.shape.interpolate(i / float(self.frames)).plot()


def animate(shape, parts=30):
    fig = plt.figure()
    func = inverse_draw(shape, parts)
    frames = [func(i) for i in range(parts + 1)]
    return animation.ArtistAnimation(fig, frames, interval=100, repeat_delay=3000)


def side_by_side(shape):
    """ Plot shape and its original position side by side """
    orig = shape.original_position()
    p1 = shape.plot()
    _, _, hx, _ = shape.bbox()
    lx, ly, _, _ = orig.bbox()
    p2 = orig.translate((hx - lx, -ly)).plot()
    return p1, p2


class Point(np.ndarray):
    def __new__(self, x, y=None):
        """ Point(x, y) or Point([x, y]) """
        if y is None:
            if len(x) != 2:
                raise Exception("Point must have exactly x, y coordinates")
            vals = x
        else:
            vals = [x, y]
        obj = np.asarray(vals).view(self)
        return obj

    def __eq__(a, b):
        return distance(a, b) < EPSILON

    def __ne__(a, b):
        return not(a == b)

    def __lt__(a, b):
        return (a.x, a.y) < (b.x, b.y)

    def __gt__(a, b):
        return not(a < b)

    def __ge__(a, b):
        return a == b or a > b

    def __le__(a, b):
        return a == b or a < b

    def __repr__(self):
        #return "({:0.2f}, {:0.2f})".format(self.x, self.y)
        return str((self.x, self.y))

    def __hash__(self):
        return (self.x, self.y).__hash__()

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
        self._bbox = None
        self._hull = None
        self._inverse = None

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
        pieces = []
        for piece in self.pieces:
            pieces.append(piece.rotate(pivot, angle))
        return Shape(pieces)

    def translate(self, vector):
        """ Move all points by given vector"""
        pieces = []
        for piece in self.pieces:
            pieces.append(piece.translate(vector))
        return Shape(pieces)

    def apply_transform(self, transform):
        """ Apply the given transform to all pieces """
        pieces = []
        for piece in self.pieces:
            pieces.append(piece.apply_transform(transform))
        return Shape(pieces)

    def reset_transform(self):
        """ Return a copy of the shape with all transforms reset """
        return Shape([p.reset_transform() for p in self.pieces])

    def inverse(self):
        """ Return shape composed of original position for all pieces """
        if self._inverse is None:
            self._inverse = Shape([p.inverse() for p in self.pieces])
        return self._inverse

    def copy(self):
        return Shape(self.pieces)

    def bbox(self):
        if self._bbox:
            return self._bbox

        min_x = float('inf')
        min_y = min_x
        max_x = float('-inf')
        max_y = max_x

        for piece in self.pieces:
            pbbox = piece.bbox()
            min_x = min(min_x, pbbox[0])
            min_y = min(min_y, pbbox[1])
            max_x = max(max_x, pbbox[2])
            max_y = max(max_y, pbbox[3])
        self._bbox = (min_x, min_y, max_x, max_y)
        return self._bbox

    def hull(self):
        if self._hull:
            return self._hull

        points = []
        for piece in self.pieces:
            points.extend(piece.poly)
        self._hull = pointset_hull(points)
        return self._hull

    def plot(self, shift=None):
        res = tuple(piece.plot(shift=shift)[0] for piece in self.pieces)
        return res

    def original_position(self):
        return self.inverse()

    def interpolate(self, percent):
        inv = self.inverse()
        pieces = []
        for orig, cur in zip(inv.pieces, self.pieces):
            translation = orig.poly[0] - cur.poly[0]

            r1 = normalize(orig.poly[1] - orig.poly[0])
            r2 = normalize(cur.poly[1] - cur.poly[0])

            angle = vector_angle(r1, r2)
            #angle = np.arccos(dot(r1, r2))
            transformed = cur.rotate(cur.poly[0], angle * percent).translate(translation * percent)
            pieces.append(transformed)

        return Shape(pieces)

    def animate(self, n):
        for i in range(n):
            self.interpolate(i/float(n)).plot()


class Piece:
    def __init__(self, poly, transform=None, color=None, ccw_check=True):
        if transform is None:
            transform = identity_transform()
        if color is None:
            color = random_color()
        self.transform = transform
        self.poly = make_poly(poly)
        self.color = color

        self._bbox = None
        self._hull = None
        self._inverse = None

        if len(self.poly) <= 2:
            print(self.poly)
            raise Exception("Polygon must have at least 3 points")

        # Hull is convex
        if ccw_check and not(self.check_cw()):
            self.poly.reverse()
            if not(self.check_cw()):
                print(self)
                self.plot()
                show()
                raise Exception("Polygon not given in consistent cw order")

    def check_cw(self):
        """ Check that all turns are cw """
        p = self.poly
        l = len(p)
        i = 0
        while i < l:
            turn = ccw(p[i], p[(i + 1) % l], p[(i + 2) % l])
            if turn == 0:
                # Remove unnecessary points along line
                del p[(i + 1) % l]
                l -= 1
            elif not(turn == 1):
                return False
            else:
                i += 1

        return True

    def __iter__(self):
        """ Iterate over points of the piece """
        for p in self.poly:
            yield p

    def __repr__(self):
        return str(self.poly)

    def bbox(self):
        if self._bbox:
            return self._bbox

        min_x = float('inf')
        min_y = min_x
        max_x = float('-inf')
        max_y = max_x

        for point in self.poly:
            min_x = min(min_x, point.x)
            min_y = min(min_y, point.y)
            max_x = max(max_x, point.x)
            max_y = max(max_y, point.y)
        self._bbox = (min_x, min_y, max_x, max_y)
        return self._bbox

    def plot(self, shift=None):
        if shift is not None:
            poly = [p + shift for p in self.poly]
        else:
            poly = self.poly
        return plot_poly(poly, color=self.color)

    def translate(self, vector):
        return self.apply_transform(translation_matrix(vector))

    def rotate(self, pivot, radians):
        center = translation_matrix(-pivot)
        rotate = rotation_matrix(radians)
        uncenter = translation_matrix(pivot)
        return self.apply_transform(np.dot(np.dot(uncenter, rotate), center))

    def apply_transform(self, transform):
        result = np.dot(transform, self.transform)
        poly = []
        for point in self.poly:
            poly.append(transform_point(point, transform))
        return Piece(poly, transform=result, color=self.color, ccw_check=False)

    def reset_transform(self):
        return Piece(self.poly, color=self.color, ccw_check=False)

    def inverse(self):
        if self._inverse is None:
            self._inverse = self.apply_transform(np.linalg.inv(self.transform))
        return self._inverse

    def area(self):
        pass

    def cut(self, a, b):
        result = split_convex_poly(self.poly, a, b)
        if result is None:
            return self

        p1 = p2 = None
        # Split pieces have same transform so far
        if result[0]:
            poly1 = make_poly(result[0])
            if len(poly1) >= 3:
                p1 = Piece(poly1, self.transform, color=self.color)
        if result[1]:
            poly2 = make_poly(result[1])
            # Want one piece to keep same color
            if result[0]:
                set_color = None
            else:
                set_color = self.color
            if len(poly2) >= 3:
                p2 = Piece(poly2, self.transform, color=set_color)
        return p1, p2

    def original_position(self):
        return self.inverse(1)


def pointset_hull(points):
    points = np.asarray(list(set(points)))  # Get rid of duplicate points
    return make_poly(list(reversed(points[ConvexHull(points, qhull_options="Qs Pp").vertices])))


def axis_align(shape):
    """ Axis aligns shape """
    hull = shape.hull()
    i = longest_edge(hull)

    angle = vector_angle((hull[(i+1) % len(hull)] - hull[i]), Point(1, 0))
    shape = shape.rotate(hull[i], -angle)
    mx, my, _, _ = shape.bbox()
    return shape.translate(Point(-mx, -my))


def longest_edge(poly):
    """ Returns start index """
    m = (float('-inf'), None)
    for i in range(len(poly)):
        m = max(m, (length(poly[(i+1) % len(poly)] - poly[i]), i))
    return m[1]


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
    if len(points) == 0:
        return points
    poly = [Point(points[0])]
    for p in points[1:]:
        p = Point(p)
        if p != poly[-1]:
            poly.append(p)
    if len(poly) > 1 and poly[0] == poly[-1]:
        del poly[-1]
    return poly


def make_shape(points):
    poly = make_poly(points)
    return Shape([Piece(poly)])


def dot(a, b):
    return sum(a * b)


def cross(a, b):
    return float(np.cross(a, b))


def length(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    return v / length(v)


def vector_angle(a, b):
    return np.arctan2(b.y, b.x) - np.arctan2(a.y, a.x)


def tri_area(a, b, c):
    return ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x))


def ccw(a, b, c):
    """
    Return ccw of three points.  Has some error tolerance for collinear points
    """
    if a == b == c:
        return 0
    elif a == b != c or a != b == c:
        raise Exception("Duplicate points in ccw")

    # Cross product
    cross = tri_area(a, b, c)
    if abs(cross) < EPSILON:
        return 0
    elif cross > 0:
        # Left turn
        return -1
    elif cross < 0:
        # Right turn
        return 1


def polygon_ccw(poly, a, b):
    """
    Returns True if polygon is right turn from a->b, False otherwise

    Assumes no intersections (except at endpoints)
    """
    return all(ccw(a, b, p) >= 0 for p in poly)


def intersect(p, pr, q, qs):
    """
    Intersect two lines specified by endpoints (p, pe) and (q, qe).
    Returns (intersection point, parametric p-pe, parametric q-qe)

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
    if u < -EPSILON or u > (1 + EPSILON) or t < -EPSILON or t > (1 + EPSILON):
        return None

    return (p + t * r, t, u)


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
            # Don't count essentially parallel lines as intersecting
            if ccw(poly[0], poly[1], a) == 0 or ccw(poly[0], poly[1], b) == 0:
                continue
            loc, line_par, edge_par = hit
            if abs(edge_par) < EPSILON:
                # Reduce numerical errors when line goes through point
                hits.append((i, poly[i], line_par))
            # Don't consider intersection at end of edge
            elif distance(loc, poly[(i+1) % len(poly)]) > EPSILON:
                hits.append((i, loc, line_par))
    # At most 2 intersections for convex polygon and line
    if len(hits) >= 2:
        # Handle numerical errors where same point is found multiple times
        for i in range(len(hits)):
            if distance(hits[i][1], hits[(i+1) % len(hits)][1]) < EPSILON:
                print("Numerical error in split")
                del hits[i]
                break

    if len(hits) > 2:
        raise Exception("More than 2 line-polygon intersections")

    if len(hits) == 2:
        i1 = hits[0][0]
        i2 = hits[1][0]
        # Don't count intersections parallel to edge
        if ((i1 + 1) % len(poly) == i2
            and ccw(poly[i1], poly[i2], a) == 0
            and ccw(poly[i1], poly[i2], b) == 0):
            hits = []

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

    # Determine which hit was first and which part is on which side
    if t1 < t2:
        return p1, p2
    else:
        return p2, p1
