from equidecompose import *
from primitives import *


# Random tests
def test_tri2rect(cuts=100, show_flag=False):
    tri = random_triangle()
    #tri = tri_shape
    cut = tri
    for i in range(cuts):
        cut = random_cut_shape(cut)

    lx, ly, mx, my = cut.bbox()
    rect = tri2rect(cut)
    h = rect.hull()
    if (len(h) != 4 or
        abs(distance(h[0], h[1]) - distance(h[2], h[3])) > EPSILON or
        abs(distance(h[1], h[2]) - distance(h[3], h[0])) > EPSILON):
        raise Exception("tri2rect did not give a rectangle")
    if show_flag:
        cut.plot()
        rect.plot((mx - lx, 0))
        rect.original_position().plot((2 * (mx - lx), 0))
        show()
    return tri, cut, rect


def find_bad():
    while True:
        tri, cut, rect = test_tri2rect()
        if len(rect.hull()) > 4:
            return (tri, cut, rect)
