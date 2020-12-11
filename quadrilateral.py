"""quadrilateral to be used in multiscale dofmap"""


class Quad:
    """
    3-----2
    |     |
    |     |
    0-----1
    """
    def __init__(self):
        self.verts = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        self.edges = {0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 1)}
        self.faces = {0: (0, 1, 2, 3)}
        self.topology = {0: {0: (0,), 1: (1,), 2: (2,), 3: (3,)},
                1: self.edges, 2: self.faces}

    def set_entity_dofs(self, dofs_per_vert, dofs_per_edge, dofs_per_face):
        n_coarse_dofs = len(self.verts) * dofs_per_vert
        n_fine_dofs = len(self.edges) * dofs_per_edge
        self.entity_dofs = {0: {}, 1: {}, 2: {}}
        for v in range(len(self.verts)):
            self.entity_dofs[0][v] = [dofs_per_vert * v + i for i in range(dofs_per_vert)]
        for e in range(len(self.edges)):
            self.entity_dofs[1][e] = [n_coarse_dofs + dofs_per_edge * e + i for i in range(dofs_per_edge)]
        for f in range(len(self.faces)):
            self.entity_dofs[2][f] = [n_fine_dofs + dofs_per_face * f + i for i in range(dofs_per_face)]


def main():
    q = Quad()
    q.set_entity_dofs(2, 3, 0)

    """example

    v2----e2----v4----e6----v5
    |           |           |
    e3    c0    e1    c1    e5
    |           |           |
    v0----e0----v1----e4----v3
    """
    # each cell has vertices and edges numbered globally
    # TODO has to be constructed from gmsh
    cells = [{0: [0, 1, 4, 2], 1: [0, 1, 2, 3]},
            {0: [1, 3, 5, 4], 1: [4, 5, 6, 1]}]
    dimension = [0, 1]
    entities = [0, 1, 2]

    dofmap = {0: {}, 1: {}}

    num_edges = 7
    num_verts = 6
    N = num_edges * 3 + num_verts * 2

    # FIXME Maybe something like this?
    global_dof = 0
    for cell in cells:
        for dim in dimension:
            entities = cell[dim]
            for local_ent, ent in enumerate(entities):
                if ent not in dofmap[dim].keys():
                    dofmap[dim][ent] = []
                    dofs = q.entity_dofs[dim][local_ent]
                    for dof in dofs:
                        dofmap[dim][ent].append(global_dof)
                        global_dof += 1
    print(dofmap)

    # TODO
    # use gmsh "quad8" or "quad" and define edge graph based on class "Quad"
    # test sparsity pattern


if __name__ == "__main__":
    main()
