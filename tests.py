from equidecompose import *
from primitives import *


# Random tests
def test_tri2rect(cuts=1, show=False):
    print("Test")
    tri = random_triangle()
    #tri = tri_shape
    cut = tri
    for i in range(cuts):
        cut = random_cut_shape(cut)

    lx, ly, mx, my = cut.bbox()
    rect = tri2rect(cut)
    if show:
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
