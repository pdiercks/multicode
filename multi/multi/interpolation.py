import dolfinx
import numpy as np


# this code is part of the fenicsx-tutorial written by Jørgen S. Dokken
# see https://jorgensd.github.io/dolfinx-tutorial/chapter1/membrane_code.html#making-curve-plots-throughout-the-domain
def interpolate(u, points):
    """evaluate u at points"""
    V = u.function_space
    domain = V.mesh
    bb_tree = dolfinx.geometry.BoundingBoxTree(domain, domain.topology.dim)

    cells = []
    points_on_proc = []

    if not points.shape[1] == 3:
        points = points.T
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        domain, cell_candidates, points
    )
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = u.eval(points_on_proc, cells)
    return u_values
