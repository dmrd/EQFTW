import math
from primitives import *


def tri2rect(shape, plot=False):
    hull = shape.hull()
    if len(hull) != 3:
        shape.plot()
        show()
        print(hull)
        raise Exception("Input shape must be a triangle")

    # Find the longest edge
    i = longest_edge(hull)
    top_v = hull[(i+2) % 3]
    # Midpoint along 2 edges
    h1 = (hull[(i+1) % 3] + top_v) / 2.0
    h2 = (hull[i] + top_v) / 2.0
    c1, c2 = extend_line(h1, h2)
    top, bottom = shape.cut(c1, c2)

    mid = top_v.project(h1, h2)
    v_bottom, v_top = extend_line(mid, top_v)
    tl, tr = top.cut(v_bottom, v_top)

    if tl is not None:
        tl.rotate(h1, math.pi)
    if tr is not None:
        tr.rotate(h2, -math.pi)

    if plot:
        hull[i].plot()
        mid.plot()
        top_v.plot()
        plot_line(c1, c2)
        plot_line(v_bottom, v_top)
        plot_line(hull[i], hull[(i+1) % 3])

    return bottom.combine(tl, tr)


def axis_align(shape):
    """ Axis aligns shape """
    hull = shape.hull()
    i = longest_edge(hull)

    angle = vector_angle((hull[(i+1) % len(hull)] - hull[i]), Point(1, 0))
    shape.rotate(hull[i], -angle)
    mx, my, _, _ = shape.bbox()
    shape.translate(Point(-mx, -my))


def rect2square(shape, plot=False):
    hull = shape.hull()
    if len(hull) != 4:
        raise Exception("Input shape must be a rectangle")
    axis_align(shape)

    # From here on, assume it is axis aligned and operate on bbox instead of
    # hull (should be the same, but may differ because of numerical errors)

    lx, ly, hx, hy = shape.bbox()
    assert(lx == 0 and ly == 0)
    while hx > 2 * hy:
        b_cut = Point(hx / 2, 0)
        t_cut = Point(hx / 2, hy)
        # Cut in half and stack right half on top of left half
        b_cut, t_cut = extend_line(b_cut, t_cut)
        bottom, top = shape.cut(b_cut, t_cut)
        top.translate(Point(-hx / 2, hy))
        shape = merge_shapes([bottom, top])
        lx, ly, hx, hy = shape.bbox()
    axis_align(shape)
    lx, ly, hx, hy = shape.bbox()
    area = hx * hy
    sides = math.sqrt(area)

    # Two lines - one splits from upper left of target square to lower right of triangle
    # Other splits vertically along right side of target square
    diag_begin = Point(0, sides)
    diag_end = Point(hy, 0)
    diag_begin, diag_end = extend_line(Point(0, sides), Point(hx, 0))
    split_begin, split_end = extend_line(Point(sides, 0), Point(sides, hy))
    upper_tri, bottom = shape.cut(diag_begin, diag_end)
    bottom, lower_tri = bottom.cut(split_begin, split_end)

    if show is True:
        plot_line(diag_begin, diag_end)
        plot_line(split_begin, split_end)
        bottom.plot((-1, 0))
        upper_tri.plot((0, 1))
        lower_tri.plot((1, -1))
        show()

    # Put lower tri in proper place
    lt_lx, lt_ly, _, _ = lower_tri.bbox()
    lower_tri.translate(Point(lx - lt_lx, hy - lt_ly))

    # Put upper tri in proper place
    ut_lx, _, _, ut_hy = upper_tri.bbox()
    upper_tri.translate(Point(-ut_lx, sides - ut_hy))

    if show is True:
        plot_line(diag_begin, diag_end)
        plot_line(split_begin, split_end)
        bottom.plot((-1, 0))
        upper_tri.plot((0, 1))
        lower_tri.plot((1, -1))
        show()

    return merge_shapes([bottom, lower_tri, upper_tri])


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
