import math
from primitives import *


def tri2rect(shape):
    hull = shape.hull()
    if len(hull) != 3:
        shape.plot()
        show()
        print(hull)
        raise Exception("Input shape must be a triangle")

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


def tridebug(shape):
    hull = shape.hull()
    if len(hull) != 3:
        shape.plot()
        show()
        print(hull)
        raise Exception("Input shape must be a triangle")
    # Midpoint along 2 edges
    h1 = (hull[0] + hull[1]) / 2.0
    h2 = (hull[2] + hull[1]) / 2.0
    c1, c2 = extend_line(h1, h2)
    top, bottom = shape.cut(c1, c2)
    mid = hull[1].project(h1, h2)
    v_bottom, v_top = extend_line(mid, hull[1])
    tl, tr = top.cut(v_bottom, v_top)
    #if tl is not None:
        #tl.rotate(h1, math.pi)
    #if tr is not None:
        #tr.rotate(h2, -math.pi)
    return {"result": bottom.combine(tl, tr),
            "bottom": bottom,
            "top": top,
            "tl": tl,
            "tr": tr,
            "v_bottom": v_bottom,
            "v_top": v_top}


def rect2square(shape):
    hull = shape.hull()
    if len(hull) != 4:
        raise Exception("Input shape must be a rectangle")
    #t1 = length(poly[1] - poly[0])
    #t2 = length(poly[2] - poly[1])
    #s1 = max(t1, t2)
    #s2 = min(t1, t2)
    ## Work on axis aligned rectangle
    #np = make_poly([[0, 0], [0, s2], [s1, s2], [s1, 0]])
    #area = s1 * s2
    #s_side = math.sqrt(area)

    #if max(s1, s2) > 2 * s_side:
    #    # split in two and stack
    #    s1 /= 2.0
    #    s2 *= 2.0
    #    np = make_poly([[0, 0], [0, s2], [s1, s2], [s1, 0]])


def combine_squares(squares):
    if any(len(poly) != 4 for poly in squares):
        raise Exception("All input shapes must be squares")
    pass


def triangulate(poly):
    pass

square = make_poly([(0, 0), (0, 1), (1, 1), (1, 0)])
tri = make_poly([(-1, 0), (0, 5), (1, 0)])
tri_shape = Shape([Piece(tri)])
square_shape = Shape([Piece(square)])


def random_point(low, high):
    r = np.random.randint
    return Point(r(low, high), r(low, high))


def random_points(n, low, high):
    return [random_point(low, high) for i in range(n)]


def random_triangle(low=0, high=100):
    p1, p2, p3 = random_points(3, low, high)
    while ccw(p1, p2, p3) == 0:
        p1, p2, p3 = random_points(3, low, high)

    return Shape([Piece([p1, p2, p3])])


def random_cut_shape(shape):
    lx, ly, hx, hy = shape.bbox()
    x1, x2 = lx + np.random.rand(2) * (hx - lx)
    start, end = extend_line(Point(x1, ly), Point(x2, hy))
    return merge_shapes(shape.cut(start, end))
