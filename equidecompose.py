import math
from scipy.spatial import Delaunay, ConvexHull
from collections import deque
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
        tl = tl.rotate(h1, math.pi)
    if tr is not None:
        tr = tr.rotate(h2, -math.pi)

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
    shape = shape.rotate(hull[i], -angle)
    mx, my, _, _ = shape.bbox()
    return shape.translate(Point(-mx, -my))


def rect2square(shape, plot=False):
    hull = shape.hull()
    if len(hull) != 4:
        raise Exception("Input shape must be a rectangle")
    shape = axis_align(shape)
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
        top = top.translate(Point(-hx / 2, hy))
        shape = merge_shapes([bottom, top])
        lx, ly, hx, hy = shape.bbox()
    shape = axis_align(shape)
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

    if plot is True:
        bottom.plot((-1, 0))
        upper_tri.plot((0, 1))
        lower_tri.plot((1, -1))

    # Put lower tri in proper place
    lt_lx, lt_ly, _, _ = lower_tri.bbox()
    lower_tri = lower_tri.translate(Point(lx - lt_lx, hy - lt_ly))

    # Put upper tri in proper place
    ut_lx, _, _, ut_hy = upper_tri.bbox()
    upper_tri = upper_tri.translate(Point(-ut_lx, sides - ut_hy))

    result = merge_shapes([bottom, lower_tri, upper_tri])

    if plot is True:
        result.plot((sides * 2, 0))
        show()

    return result


def combine_two_squares(a, b, plot=False):
    """
    Takes two axis aligned squares, with lower left corners at (0, 0).
    Returns a single square composed of their pieces
    """
    #assert(len(a.hull()) == 4)
    #assert(len(b.hull()) == 4)
    a = axis_align(a)
    b = axis_align(b)

    _, _, ax, ay = a.bbox()
    _, _, bx, by = b.bbox()

    aa = ax * ay
    ba = bx * by

    if aa > ba:
        larger = a
        lx, ly = ax, ay
        smaller = b
        sx, sy = bx, by
    else:
        larger = b
        lx, ly = bx, by
        smaller = a
        sx, sy = ax, ay

    # Stack smaller on top right of large square
    smaller = smaller.translate(Point(lx - sx, ly))

    #st = Point(lx, sy)  # Vector representing top and bottom of new square
    ss = Point(sy,  -lx)  # Vector representing sides of new square

    # All cuts go down or to right
    s1_s, s1_e = Point(0, ly), Point(lx, sy + ly)  # top
    s2_s, s2_e = Point(0, ly), Point(0, ly) + ss  # left
    s3_s, s3_e = Point(lx, ly + sy), Point(lx, ly + sy) + ss  # right
    s4_s, s4_e = s2_e, s3_e  # bottom
    c1_s, c1_e = extend_line(s1_s, s1_e)
    c2_s, c2_e = extend_line(s2_s, s2_e)
    c3_s, c3_e = extend_line(s3_s, s3_e)
    c4_s, c4_e = extend_line(s4_s, s4_e)

    small_top, small_bot = smaller.cut(c1_s, c1_e)
    large_inner, large_t1 = larger.cut(c2_s, c2_e)
    large_inner, large_t2 = large_inner.cut(c4_s, c4_e)

    if plot:
        #smaller.plot()
        #larger.plot()
        small_top.plot()
        small_bot.plot()
        large_t1.plot()
        large_t2.plot()
        large_inner.plot()
        plot_line(s1_s, s1_e)
        plot_line(s2_s, s2_e)
        plot_line(s3_s, s3_e)
        plot_line(s4_s, s4_e)

    large_t1 = large_t1.translate(Point(lx, sy))

    x, _, _, _ = large_t2.bbox()
    large_t2 = large_t2.translate(Point(-x, ly))

    small_top = small_top.translate(s3_e - Point(lx, sy + ly))

    result = merge_shapes([small_top, small_bot, large_t1, large_t2, large_inner])

    result = axis_align(result)

    if plot:
        result.plot((2*lx, 0))
        show()

    return result


def combine_squares(squares):
    if any(len(shape.hull()) != 4 for shape in squares):
        raise Exception("All input shapes must be squares")
    q = deque([axis_align(square) for square in squares])

    # Combine in pairs of two
    while len(q) > 1:
        s1 = q.popleft()
        s2 = q.popleft()
        r = combine_two_squares(s1, s2)
        q.append(r)

    return q.popleft()


def triangulate(poly):
    """
    Takes a polygon and returns a shape representing it with triangular pieces
    For now just returns the delaunay triangulation for testing (until I get
    a real triangulation library)
    """
    points = np.array(poly)
    points = points[ConvexHull(points).vertices]
    triangles = Delaunay(points).simplices
    pieces = []
    for triangle in triangles:
        pieces.append(Piece(make_poly(points[triangle])))
    return Shape(pieces)


def equidecompose_to_square(polygon):
    triangulated = triangulate(polygon)
    triangulated_shapes = [Shape([piece]) for piece in triangulated.pieces]

    squares = []
    for triangle in triangulated_shapes:
        rect = tri2rect(triangle)
        squares.append(rect2square(rect))

    square = combine_squares(squares)
    return triangulated, square


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


def random_square(low=1, high=100):
    s = np.random.randint(low, high)
    return make_shape([(0, 0), (0, s), (s, s), (s, 0)])


def random_rect(low=1, high=100):
    s1 = np.random.randint(low, high)
    s2 = np.random.randint(low, high)
    return make_shape([(0, 0), (s2, 0), (s2, s1), (0, s1)])


def random_cut_shape(shape):
    lx, ly, hx, hy = shape.bbox()
    x1, x2 = lx + np.random.rand(2) * (hx - lx)
    start, end = extend_line(Point(x1, ly), Point(x2, hy))
    return merge_shapes(shape.cut(start, end))


def random_example(n):
    triangulated, r = equidecompose_to_square(random_points(n, 0, 50))
    return animate(r, 30)
