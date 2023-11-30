from typing import Union, Callable, Optional
from dolfinx import fem, mesh
from dolfinx.fem.petsc import set_bc
import ufl
import numpy as np
import numpy.typing as npt
from petsc4py.PETSc import InsertMode, ScatterMode


def get_boundary_dofs(V: fem.FunctionSpaceBase, marker: Optional[Callable] = None) -> npt.ArrayLike:
    """Returns dofs on the boundary"""
    domain = V.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1
    if marker is not None:
        entities = mesh.locate_entities_boundary(domain, fdim, marker)
    else:
        everywhere = lambda x: np.full(x.shape[1], True, dtype=bool)
        entities = mesh.locate_entities_boundary(domain, fdim, everywhere)
    dofs = fem.locate_dofs_topological(V, fdim, entities)
    return dofs


class BoundaryDataFactory(object):
    """handles creation of Functions for extension into a domain Ω

    the Functions are constructed such that
        u=g on Γ and u=0 on ∂Ω \\ Γ

    (for the extension also the latter condition is important).
    Usually, we have a bc known as {value, boundary(=Γ)} and
    can create a bc and apply it to a function (dolfinx.fem.petsc.set_bc)
    to achieve the above.
    However, for the extension we require a dirichletbc object that
    applies to the entire boundary ∂Ω.
    """

    def __init__(self, domain, V):
        """

        Parameters
        ----------
        domain : dolfinx.mesh.Mesh
            The computational domain.
        V : dolfinx.fem.FunctionSpace
            The FE space.

        """

        self.domain = domain
        self.V = V

        # bc handler
        self.bch = BoundaryConditions(domain, V)

        # boundary facets and dofs (entire boundary)
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(domain.topology)
        self.boundary_dofs = fem.locate_dofs_topological(
            V, fdim, boundary_facets
        )

    def create_function_values(self, values, boundary_dofs):
        """create a function and set values

        Parameters
        ----------
        values : np.ndarray
            The values to set.
        boundary_dofs : np.ndarray
            The dof indices of the vector.

        Returns
        -------
        u : dolfinx.fem.Function
        """
        u = fem.Function(self.V)
        u.vector.zeroEntries()
        u.vector.setValues(boundary_dofs, values, addv=InsertMode.INSERT)
        u.vector.assemblyBegin()
        u.vector.assemblyEnd()
        return u

    def create_function_bc(self, bc):
        """create a function and set given bc

        Parameters
        ----------
        bc : dict
            A suitable definition of a Dirichlet bc.
            See multi.bcs.BoundaryConditions.add_dirichlet_bc.

        Returns
        -------
        u : dolfinx.fem.Function
        """
        self.bch.clear()
        self.bch.add_dirichlet_bc(**bc)
        bcs = self.bch.bcs
        u = fem.Function(self.V)
        set_bc(u.vector, bcs)
        u.vector.ghostUpdate(
            addv=InsertMode.INSERT_VALUES, mode=ScatterMode.FORWARD
        )
        return u

    def create_bc(self, function):
        """create a bc with value function for the entire boundary

        Parameters
        ----------
        function : dolfinx.fem.Function
            The function to prescribe.

        Returns
        -------
        bc : dolfinx.fem.dirichletbc
        """
        dofs = self.boundary_dofs
        bc = fem.dirichletbc(function, dofs)
        return bc


# adapted version of MechanicsBCs by Thomas Titscher
class BoundaryConditions(object):
    """Handles Dirichlet and Neumann boundary conditions.

    Attributes:
        domain: The computational domain.
        V: The finite element space.
    """

    def __init__(
        self,
        domain: mesh.Mesh,
        space: fem.FunctionSpaceBase,
        facet_tags: Union[mesh.MeshTags, None] = None,
    ) -> None:
        """Initializes the instance based on domain and FE space.

        It sets up lists to hold the Dirichlet and Neumann BCs
        as well as the required `ufl` objects to define Neumann
        BCs if `facet_tags` is not None.

        Args:
            domain: The computational domain.
            space: The finite element space.
            facet_tags: The mesh tags defining boundaries.
        """

        self.domain = domain
        self.V = space

        # create connectivity
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)

        # list of dirichlet boundary conditions
        self._bcs = []

        # handle facets and measure for neumann bcs
        self._neumann_bcs = []
        self._facet_tags = facet_tags
        self._ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
        self._v = ufl.TestFunction(space)

    def add_dirichlet_bc(
        self,
        value: (
            Union[fem.Function, fem.Constant, fem.DirichletBC, np.ndarray, Callable]
        ),
        boundary: Union[int, np.ndarray, Callable, None] = None,
        sub: Union[int, None] = None,
        method: str = "topological",
        entity_dim: Union[int, None] = None,
    ) -> None:
        """Adds a Dirichlet bc.

        Args:
            value: Anything that *might* be used to define the Dirichlet function.
                    It can be a `Function`, a `Callable` which is then interpolated
                    or an already existing Dirichlet BC, or ... (see type hint).
            boundary: The part of the boundary whose dofs should be constrained.
                    This can be a callable defining the boundary geometrically or
                    an array of entity tags or an integer marking the boundary if
                    `facet_tags` is not None.
            sub: If `sub` is not None the subspace `V.sub(sub)` will be
                    constrained.
            method: A hint which method should be used to locate the dofs.
                    Choices: 'topological' or 'geometrical'.
            entity_dim: The dimension of the entities to be located
                        topologically. Note that `entity_dim` is required if `sub`
                        is not None and `method=geometrical`.
        """
        if isinstance(value, fem.DirichletBC):
            self._bcs.append(value)
        else:
            assert method in ("topological", "geometrical")
            V = self.V.sub(sub) if sub is not None else self.V

            # if sub is not None and method=="geometrical"
            # dolfinx.fem.locate_dofs_geometrical(V, boundary) will raise a RuntimeError
            # because dofs of a subspace cannot be tabulated
            topological = method == "topological" or sub is not None

            if topological:
                assert entity_dim is not None

                if isinstance(boundary, int):
                    if self._facet_tags is None:
                        raise AttributeError("There are no facet tags defined!")
                    else:
                        facets = self._facet_tags.find(boundary)
                    if facets.size < 1:
                        raise ValueError(f"Not able to find facets tagged with value {boundary=}.")
                elif isinstance(boundary, np.ndarray):
                    facets = boundary
                else:
                    facets = mesh.locate_entities_boundary(self.domain, entity_dim, boundary)

                dofs = fem.locate_dofs_topological(V, entity_dim, facets)
            else:
                dofs = fem.locate_dofs_geometrical(V, boundary)

            try:
                bc = fem.dirichletbc(value, dofs, V)
            except TypeError:
                # value is Function and V cannot be passed
                # TODO understand 4th constructor
                # see dolfinx/fem/bcs.py line 127
                bc = fem.dirichletbc(value, dofs)
            except AttributeError:
                # value has no Attribute `dtype`
                f = fem.Function(V)
                f.interpolate(value)
                bc = fem.dirichletbc(f, dofs)

            self._bcs.append(bc)

    def add_neumann_bc(self, marker: Union[str, int, list[int]], value: fem.Constant) -> None:
        """Adds a Neumann BC.

        Args:
            marker: The id of the boundary where Neumann BC should be applied.
                This can be the str 'everywhere' or int or list[int], see `ufl.Measure`.
            value: The Neumann data, e.g. a traction vector. This has
              to be a valid `ufl` object.
        """
        if self._facet_tags is not None:
            if marker not in self._facet_tags.values:
                raise ValueError(f"No facet tags defined for {marker=}.")

        self._neumann_bcs.append([value, marker])

    @property
    def has_neumann(self) -> bool:
        """check if Neumann BCs are defined

        Returns:
            True or False
        """
        return len(self._neumann_bcs) > 0

    @property
    def has_dirichlet(self) -> bool:
        """check if Dirichlet BCs are defined

        Returns:
            True or False
        """
        return len(self._bcs) > 0

    @property
    def bcs(self) -> list[fem.DirichletBC]:
        """returns the list of Dirichlet BCs

        Returns:
            The list of Dirichlet BCs.
        """

        return self._bcs

    def clear(self, dirichlet: bool = True, neumann: bool = True) -> None:
        """Clears list of Dirichlet and/or Neumann BCs.

        Args:
            dirichlet: flag for Dirichlet Bcs (if true will clear those)
            neumann: flag for Neumann Bcs (if true will clear those)

        """

        if dirichlet:
            self._bcs.clear()
        if neumann:
            self._neumann_bcs.clear()

    @property
    def neumann_bcs(self) -> ufl.Form:
        """creates the ufl form of (sum of) Neumann BCs

        Returns:
            A ufl object representing Neumann BCs
        """

        r = 0
        for expression, marker in self._neumann_bcs:
            r += ufl.inner(expression, self._v) * self._ds(marker)
        return r


def apply_bcs(lhs, rhs, bc_indices, bc_values):
    """
    Applies dirichlet bcs (in-place) using the algorithm described here
    http://www.math.colostate.edu/~bangerth/videos/676/slides.21.65.pdf

    Parameters
    ----------
    lhs
        The left hand side of the system.
    rhs
        The right hand side of the system.
    bc_indices
        DOF indices where bcs should be applied.
    bc_values
        The boundary data.

    Returns
    -------
    None
    """
    assert isinstance(lhs, np.ndarray)
    assert isinstance(rhs, np.ndarray)
    assert isinstance(bc_indices, (list, np.ndarray))
    if isinstance(bc_indices, list):
        bc_indices = np.array(bc_indices)
    assert isinstance(bc_values, (list, np.ndarray))
    if isinstance(bc_values, list):
        bc_values = np.array(bc_values)

    rhs.shape = (rhs.size,)
    values = np.zeros(rhs.size)
    values[bc_indices] = bc_values
    # substract bc values from right hand side
    rhs -= np.dot(lhs[:, bc_indices], values[bc_indices])
    # set columns to zero
    lhs[:, bc_indices] = np.zeros((rhs.size, bc_indices.size))
    # set rows to zero
    lhs[bc_indices, :] = np.zeros((bc_indices.size, rhs.size))
    # set diagonal entries to 1.
    lhs[bc_indices, bc_indices] = np.ones(bc_indices.size)
    # set bc_values on right hand side
    rhs[bc_indices] = values[bc_indices]
