import math
from primitives import *


def tri2rect(shape):
    hull = shape.hull()
    assert(len(hull) == 3)

    # Midpoint along 2 edges
    h1 = (hull[0] + hull[1]) / 2.0
    h2 = (hull[2] + hull[1]) / 2.0
    c1, c2 = extend_line(h1, h2)
    top, bottom = shape.cut(c1, c2)

    mid = hull[1].project(h1, h2)
    v_bottom, v_top = extend_line(mid, hull[1])
    tl, tr = top.cut(v_bottom, v_top)

    if tl is not None:
        tl.rotate(h1, math.pi)
    if tr is not None:
        tr.rotate(h2, -math.pi)

    return bottom.combine(tl, tr)


def rect2square(poly):
    assert(len(poly) == 4)
    t1 = length(poly[1] - poly[0])
    t2 = length(poly[2] - poly[1])
    s1 = max(t1, t2)
    s2 = min(t1, t2)
    # Work on axis aligned rectangle
    np = make_poly([[0, 0], [0, s2], [s1, s2], [s1, 0]])
    area = s1 * s2
    s_side = math.sqrt(area)

    if max(s1, s2) > 2 * s_side:
        # split in two and stack
        s1 /= 2.0
        s2 *= 2.0
        np = make_poly([[0, 0], [0, s2], [s1, s2], [s1, 0]])


def combine_squares(squares):
    assert(all(len(poly) == 4 for poly in squares))
    pass


def triangulate(poly):
    pass

square = make_poly([(0, 0), (0, 1), (1, 1), (1, 0)])
tri = make_poly([(-1, 0), (0, 5), (1, 0)])
tri_shape = Shape([Piece(tri)])
square_shape = Shape([Piece(square)])

tri_cut = merge_shapes(tri_shape.cut(Point(0, -3), Point(1, 6)))
#print(tri_cut)
tri_cut = merge_shapes(tri_cut.cut(Point(-2, 1), Point(7, -2)))


def random_point(low, high):
    r = np.random.randint
    return Point(r(low, high), r(low, high))


def random_points(n, low, high):
    return [random_point(low, high) for i in range(n)]


def random_triangle(m, low=0, high=100):
    return Shape(Piece(random_points(3, low, high)))


def random_cut_shape(shape):
    lx, ly, hx, hy = shape.bbox()
    x = np.random.randint(lx, hx)
    x2 = np.random.randint(lx, hx)
    return merge_shapes(shape.cut(extend_line(Point(x, ly), Point(x2, hy))))
