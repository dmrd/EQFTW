import math
from primitives import *


def tri2rect(poly):
    assert(len(poly) == 3)
    # Midpoint along 2 edges
    h1 = (poly[0] + poly[1]) / 2.0
    h2 = (poly[2] + poly[1]) / 2.0

    # Create two parts from top of triangle
    mid = project(poly[1], h1, h2)
    left_tri = [h1, poly[1], mid]
    right_tri = [poly[1], h2, mid]
    right_tri = [h2, mid, poly[1]]

    # Resulting triangules from rotating around midpoints
    # Do directly instead of using rotate_polygon to reduce numerical errors
    left_rot = [h1, 2 * h1 - mid, poly[0]]
    right_rot = [h2, 2 * h2 - mid, poly[2]]

    quad = [poly[0], h1, h2, poly[2]]
    rect = [poly[0], left_rot[2], right_rot[2], poly[2]]
    return {'rect': rect,
            'inner_quad': quad,
            'left_tri': left_tri,
            'right_tri': right_tri,
            'left_rot': left_rot,
            'right_rot': right_rot}


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
tri = make_poly([(-1, 0), (0, 1), (5, 0)])
