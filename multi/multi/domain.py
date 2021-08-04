import os
from pathlib import Path

import dolfin as df
import numpy as np


class Domain:
    """Class to represent a subdomain Ω_i in the context of a multiscale method

    Parameters
    ----------
    mesh : str, dolfin.cpp.mesh.Mesh
        The discretization of the subdomain given as XMDF file (incl. ext) to load or
        instance of dolfin.cpp.mesh.Mesh.
    id_ : int
        The identification number of the subdomain.
    subdomains : optional
        Set to True if Ω_i has subdomains represented by a df.MeshFunction stored in
        the XDMF file given as `mesh` or provide df.MeshFunction directly.
    edges : bool, optional
        If True load mesh for each edge of the mesh.
    translate : optional
        A dolfin.Point by which the mesh is translated. If `edges=True` the edge
        meshes are translated as well.

    """

    def __init__(self, mesh, id_=0, subdomains=None, edges=False, translate=None):
        if isinstance(mesh, df.cpp.mesh.Mesh):
            self.mesh = mesh
            self.subdomains = subdomains
        else:
            self.xdmf_file = Path(mesh)
            self.mesh = df.Mesh()
            mvc = df.MeshValueCollection("size_t", self.mesh, dim=None)
            # note that mvc.dim() will be overwritten with
            # self.mesh.topology().dim() by f.read(mvc, ...) below

            with df.XDMFFile(self.xdmf_file.as_posix()) as f:
                f.read(self.mesh)
                if subdomains:
                    f.read(mvc, "gmsh:physical")

            if subdomains:
                self.subdomains = df.MeshFunction("size_t", self.mesh, mvc)
            else:
                self.subdomains = subdomains

        if translate:
            self.mesh.translate(translate)

        # ### domain coordinates
        coord = self.mesh.coordinates()
        self.xmin = np.amin(coord[:, 0])
        self.xmax = np.amax(coord[:, 0])
        self.ymin = np.amin(coord[:, 1])
        self.ymax = np.amax(coord[:, 1])

        if edges:
            self._load_edges(translate)
        else:
            self.edges = False
        self.id = int(id_)

    def translate(self, point):
        """translate the domain in space

        Parameters
        ----------
        point : dolfin.Point
            The point by which to translate.

        """
        self.mesh.translate(point)
        # update coordinates
        coord = self.mesh.coordinates()
        self.xmin = np.amin(coord[:, 0])
        self.xmax = np.amax(coord[:, 0])
        self.ymin = np.amin(coord[:, 1])
        self.ymax = np.amax(coord[:, 1])
        # update edges if True
        if self.edges:
            for edge in self.edges:
                edge.translate(point)

    def get_nodes(self, n=4):
        """get nodes to define shape functions

        Parameters
        ----------
        n : int, optional
            Number of nodes.

        Note
        ----
        This only makes sense for square shaped domains.
        """

        def midpoint(a, b):
            return a + (b - a) / 2

        nodes = np.array(
            [
                [self.xmin, self.ymin],
                [self.xmax, self.ymin],
                [self.xmax, self.ymax],
                [self.xmin, self.ymax],
                [midpoint(self.xmin, self.xmax), self.ymin],
                [self.xmax, midpoint(self.ymin, self.ymax)],
                [midpoint(self.xmin, self.xmax), self.ymax],
                [self.xmin, midpoint(self.ymin, self.ymax)],
                [midpoint(self.xmin, self.xmax), midpoint(self.ymin, self.ymax)],
            ]
        )
        return nodes[:n]

    def _load_edges(self, translate=None):
        path = os.path.dirname(os.path.abspath(self.xdmf_file))
        base = os.path.splitext(os.path.basename(self.xdmf_file))[0]
        ext = os.path.splitext(os.path.basename(self.xdmf_file))[1]

        def read(xdmf):
            mesh = df.Mesh()
            with df.XDMFFile(xdmf) as f:
                f.read(mesh)
            return mesh

        edge_meshes = []
        boundary = ["bottom", "right", "top", "left"]
        for b in boundary:
            edge = path + "/" + base + f"_{b}" + ext
            mesh = read(edge)
            if translate:
                mesh.translate(translate)
            edge_meshes.append(mesh)
        self.edges = tuple(edge_meshes)
