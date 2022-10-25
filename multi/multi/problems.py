import pathlib
import yaml
import dolfinx
from dolfinx.io import gmshio
import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from multi.bcs import BoundaryConditions
from multi.domain import StructuredQuadGrid
from multi.interpolation import make_mapping
from multi.materials import LinearElasticMaterial
from multi.misc import x_dofs_VectorFunctionSpace
from multi.product import InnerProduct
from multi.projection import orthogonal_part
from multi.sampling import correlation_matrix, _create_random_values
from multi.solver import build_nullspace

from pymor.core.logger import getLogger
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from pymor.tools.random import get_random_state
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator

from scipy.sparse import csc_matrix


def _solver_options(
    solver=PETSc.KSP.Type.PREONLY, preconditioner=PETSc.PC.Type.LU, keep_solver=True
):
    return {
        "solver": solver,
        "preconditioner": preconditioner,
        "keep_solver": keep_solver,
    }


class LinearProblem(object):

    """Docstring for LinearProblem."""

    def __init__(self, domain, V, solver_options=None):
        """TODO: to be defined.

        Parameters
        ----------
        domain : multi.Domain
            The computational domain.
        V : dolfinx.fem.FunctionSpace
            The FE space.
        solver_options : optional
            The solver options.

        """
        self.logger = getLogger("multi.problems.LinearProblem")
        self.domain = domain
        self.V = V
        self.u = ufl.TrialFunction(V)
        self.v = ufl.TestFunction(V)
        self._bc_handler = BoundaryConditions(domain.mesh, V, domain.facet_markers)
        self._solver_options = solver_options or _solver_options()

    def add_dirichlet_bc(
        self, value, boundary=None, sub=None, method="topological", entity_dim=None
    ):
        self._bc_handler.add_dirichlet_bc(
            value, boundary=boundary, sub=sub, method=method, entity_dim=entity_dim
        )

    def add_neumann_bc(self, marker, value):
        self._bc_handler.add_neumann_bc(marker, value)

    def clear_bcs(self, dirichlet=True, neumann=True):
        """remove all dirichlet and neumann bcs"""
        self._bc_handler.clear(dirichlet=dirichlet, neumann=neumann)

    def get_dirichlet_bcs(self):
        return self._bc_handler.bcs

    def discretize_product(self, product, bcs=False, product_name=None):
        """discretize inner product

        Parameters
        ----------
        product : str or ufl.form.Form
            Either a string (see multi.product.InnerProduct) or
            a ufl form defining the inner product.
        bcs : bool, optional
            If True, apply BCs to inner product matrix.

        Returns
        -------
        product : dolfin.Matrix or None
            Returns a dolfin.Matrix or None in case of euclidean
            inner product.
        """
        if bcs:
            bcs = self.get_dirichlet_bcs()
            if not len(bcs) > 0:
                raise Warning("Forgot to apply BCs?")
        else:
            bcs = ()
        product = InnerProduct(self.V, product, bcs=bcs, name=product_name)
        return product.assemble_matrix()  # returns Matrix or None

    def get_form_lhs(self):
        pass

    def get_form_rhs(self):
        pass

    def setup_solver(self):
        """create matrix and vector objects, and setup solver"""
        a = self.get_form_lhs()
        L = self.get_form_rhs()
        self._A = dolfinx.fem.form(a)
        self._b = dolfinx.fem.form(L)
        self._matrix = dolfinx.fem.petsc.create_matrix(self._A)
        self._vector = dolfinx.fem.petsc.create_vector(self._b)

        options = self._solver_options
        method = options.get("solver")
        preconditioner = options.get("preconditioner")
        solver = PETSc.KSP().create(self.V.mesh.comm)
        solver.setOperators(self._matrix)
        solver.setType(method)
        solver.getPC().setType(preconditioner)

        return solver

    def assemble_matrix(self, bcs=()):
        """assemble matrix and apply boundary conditions"""
        self._matrix.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self._matrix, self._A, bcs=bcs)
        self._matrix.assemble()

        return self._matrix

    def assemble_vector(self, bcs=()):
        """assemble vector and apply boundary conditions"""
        self._vector.zeroEntries()
        dolfinx.fem.petsc.assemble_vector(self._vector, self._b)
        self._vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )

        # Compute b - J(u_D-u_(i-1))
        dolfinx.fem.petsc.apply_lifting(self._vector, [self._A], [bcs])
        # Set dx|_bc = u_{i-1}-u_D
        dolfinx.fem.petsc.set_bc(self._vector, bcs, scale=1.0)
        self._vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
        )

        return self._vector

    def solve(self, u=None):
        """performs single solve"""
        try:
            solver = self._solver
        except AttributeError:
            self.logger.info("Setting up the solver ...")
            solver = self.setup_solver()

        bcs = self.get_dirichlet_bcs()

        _ = self.assemble_matrix(bcs)
        rhs = self.assemble_vector(bcs)
        self.logger.info("Solving linear variational problem ...")

        if u is None:
            u = dolfinx.fem.Function(self.V)

        solver.solve(rhs, u.vector)

        if self._solver_options["keep_solver"]:
            self._solver = solver

        return u


class LinearElasticityProblem(LinearProblem):
    """class representing a linear elastic problem

    Parameters
    ----------
    domain : multi.domain.Domain
        The computational domain.
    V : dolfin.FunctionSpace
        The finite element space.
    E : float or tuple of float
        Young's modulus of the linear elastic materials.
    NU : float or tuple of float
        Poisson ratio of the linear elastic materials.
    plane_stress : bool, optional
        2d constraint.
    solver_options : optional
        The solver options.

    """

    def __init__(
        self, domain, V, E=210e3, NU=0.3, plane_stress=False, solver_options=None
    ):
        super().__init__(domain, V, solver_options)
        assert all(
            [isinstance(E, (float, tuple, list)), isinstance(NU, (float, tuple, list))]
        )
        if isinstance(E, float) and isinstance(NU, float):
            E = (E,)
            NU = (NU,)
            assert domain.cell_markers is None
            self.dx = ufl.dx
        else:
            if len(E) > 1 and domain.cell_markers is None:
                raise KeyError("You need to define mesh tags for multiple materials")
            assert all(
                [
                    len(E) == len(NU),
                    len(E) == np.unique(domain.cell_markers.values).size,
                ]
            )
            # FIXME double check if gmsh cell data starts at 1
            assert np.amin(domain.cell_markers.values) > 0
            mesh = domain.mesh
            subdomains = domain.cell_markers
            self.dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
        self.gdim = int(V.element.value_shape)
        assert self.gdim in (1, 2, 3)
        self.materials = [
            LinearElasticMaterial(self.gdim, E=e, NU=nu, plane_stress=plane_stress)
            for e, nu in zip(E, NU)
        ]
        if hasattr(domain, "edges"):
            if domain.edges:
                self._init_edge_spaces()

    def _init_edge_spaces(self):
        edge_meshes = self.domain.edges
        V = self.V
        ufl_element = V.ufl_element()
        family_name = ufl_element.family_name
        degree = ufl_element.degree()

        # FIXME dolfinx:nightly now returns basix.ufl_wrapper.VectorElement
        # instead of ufl.VectorElement --> .reconstruct is not implemened
        # Therefore, _init_edge_spaces was moved to here and VectorFunctionSpace
        # is used to create edge spaces.

        V_to_L = {}
        Lambda = {}
        for key, facet_mesh in edge_meshes.items():
            # facet_element = ufl_element.reconstruct(cell=facet_mesh.ufl_cell())
            L = dolfinx.fem.VectorFunctionSpace(facet_mesh, (family_name, degree))
            Lambda[key] = L
            V_to_L[key] = make_mapping(L, V)
        self.V_to_L = V_to_L
        self.edge_spaces = Lambda

    def get_form_lhs(self):
        """get bilinear form a(u, v) of the problem"""
        u = self.u
        v = self.v
        if len(self.materials) > 1:
            return sum(
                [
                    ufl.inner(mat.sigma(u), mat.eps(v)) * self.dx(i + 1)
                    for (i, mat) in enumerate(self.materials)
                ]
            )
        else:
            mat = self.materials[0]
            return ufl.inner(mat.sigma(u), mat.eps(v)) * self.dx

    def get_form_rhs(self, body_forces=None):
        """get linear form f(v) of the problem"""
        domain = self.V.mesh
        v = self.v
        zero = dolfinx.fem.Constant(domain, (PETSc.ScalarType(0.0),) * self.gdim)
        rhs = ufl.inner(zero, v) * ufl.dx
        if body_forces is not None:
            if len(self.materials) > 1:
                assert isinstance(body_forces, (list, tuple))
                assert len(body_forces) == len(self.materials)
                for i in range(len(self.materials)):
                    rhs += ufl.inner(body_forces[i], v) * self.dx(i + 1)
            else:
                rhs += ufl.dot(body_forces, v) * self.dx

        if self._bc_handler.has_neumann:
            rhs += self._bc_handler.neumann_bcs

        return rhs


class TransferProblem(object):
    """General class for transfer problems.

    This class aids the solution of transfer problems given by:

        A(u) = 0 in Ω,
        with homogeneous Dirichlet bcs on Γ_D,
        with homogeneous Neumann bcs on Γ_N,
        with inhomogeneous Neumann bcs on Γ_N_inhom,
        with arbitrary Dirichlet boundary conditions on Γ_out.

    Here, we are only interested in the solution u restricted to the
    space defined on a target subdomain Ω_in ⊂ Ω.
    The Dirichlet bcs prescribed on Γ_out are transferred to
    the target subdomain, hence the name.

    The boundaries Γ_D, Γ_N and Γ_N_inhom are the part of ∂Ω that
    intersects with the (respective dirichlet or neumann) boundary
    of the global domain Ω_gl ( Ω ⊂ Ω_gl).
    The boundary Γ_out is the part of ∂Ω that does not intersect with ∂Ω_gl.

    Note: The above problem can be formulated as a transfer operator which
    maps the boundary data on Γ_out (source space) to the solution u in
    the (range) space defined on Ω_in. Since FEniCS does
    not allow to define function spaces on some part of the boundary
    of a domain (yet), the full space is defined as the source space.
    The range space is the space defined on Ω_in.

    Parameters
    ----------
    problem : multi.problems.LinearProblem
        The problem defined on the oversampling domain Ω.
    subdomain_space : dolfinx.fem.FunctionSpace
        The range space defined on the target subdomain Ω_in.
    gamma_out : callable
        A function that defines facets of Γ_out geometrically.
        `dolfinx.mesh.locate_entities_boundary` is used to determine the facets.
        Γ_out is the part of the boundary of the oversampling domain that does
        not intersect with the boundary of the global domain.
    dirichlet : list of dict or dict, optional
        Homogeneous dirichlet boundary conditions.
        See multi.bcs.BoundaryConditions.add_dirichlet_bc for suitable values.
    neumann : list of dict or dict, optional
        Inhomogeneous neumann boundary conditions or source terms.
        See multi.bcs.BoundaryConditions.add_neumann_bc for suitable values.
    source_product : dict, optional
        The inner product to use for the source space. The dictionary should define
        the key `product` and optionally `bcs` and `product_name`.
    range_product : dict, optional
        The inner product to use for the range space. The dictionary should define
        the key `product` and optionally `bcs` and `product_name`.
    remove_kernel : bool, optional
        If True, remove kernel (rigid body modes) from solution.
    solver_options : dict, optional
        The user is required to use pymor style options ({'inverse': options}),
        because the value of solver_options is directly passed to FenicsxMatrixOperator.
        See https://github.com/pymor/pymor/blob/main/src/pymor/operators/interface.py#L32-#L41

    """

    def __init__(
        self,
        problem,
        subdomain_space,
        gamma_out,
        dirichlet=None,
        neumann=None,
        source_product=None,
        range_product=None,
        remove_kernel=False,
        solver_options=None,
    ):
        self.logger = getLogger("multi.problems.TransferProblem")
        self.problem = problem
        self.subdomain_space = subdomain_space
        self.source = FenicsxVectorSpace(problem.V)
        self.range = FenicsxVectorSpace(subdomain_space)
        self.gamma_out = gamma_out
        self.neumann = neumann
        self.remove_kernel = remove_kernel
        self.solver_options = solver_options or {"inverse": _solver_options()}

        # initialize commonly used quantities
        self._init_bc_gamma_out()
        self._S_to_R = self._make_mapping()

        # initialize fixed set of dirichlet boundary conditions on Γ_D (using self.problem)
        if dirichlet is not None:
            if isinstance(dirichlet, (list, tuple)):
                for dirichlet_bc in dirichlet:
                    problem.add_dirichlet_bc(**dirichlet_bc)
            else:
                problem.add_dirichlet_bc(**dirichlet)
            dirichlet = problem.get_dirichlet_bcs()
        self.dirichlet_bcs = dirichlet or []

        # ### inner products
        default_product = {"product": None, "bcs": (), "product_name": None}
        source_prod = source_product or default_product
        range_prod = range_product or default_product
        self.source_product = self._get_source_product(**source_prod)
        self.range_product = self._get_range_product(**range_prod)

        # initialize kernel
        if remove_kernel:
            # build null space
            l2_product = self._get_range_product(product="l2")
            self.range_l2_product = l2_product
            self.kernel = build_nullspace(self.range, product=l2_product, gdim=2)

    def _init_bc_gamma_out(self):
        """define bc on gamma out"""
        V = self.source.V
        tdim = V.mesh.topology.dim
        fdim = tdim - 1
        dummy = dolfinx.fem.Function(V)

        # determine boundary facets of Γ_out
        boundary_facets = dolfinx.mesh.locate_entities_boundary(
            V.mesh, fdim, self.gamma_out
        )
        # determine dofs on Γ_out
        _dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
        bc = dolfinx.fem.dirichletbc(dummy, _dofs)
        dofs = bc.dof_indices()[0]
        self._bc_gamma_out = bc
        self._bc_dofs_gamma_out = dofs
        # source space restricted to Γ_out
        self._source_gamma = NumpyVectorSpace(dofs.size)

    def _make_mapping(self):
        """builds map from source space to range space"""
        return make_mapping(self.range.V, self.source.V)

    def discretize_operator(self):
        """discretize the operator"""
        V = self.source.V
        Vdim = V.dofmap.bs * V.dofmap.index_map.size_global
        self.logger.info(f"Discretizing left hand side of the problem (size={Vdim}).")
        ufl_lhs = self.problem.get_form_lhs()
        compiled_form = dolfinx.fem.form(ufl_lhs)

        # A refers to full operator without bcs applied
        # which is used to construct rhs (apply lifting)
        A = dolfinx.fem.petsc.create_matrix(compiled_form)
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, compiled_form)
        A.assemble()
        self._A = FenicsxMatrixOperator(
            A, V, V, solver_options=self.solver_options, name="A"
        )

        # A_0 refers to operator with bcs applied
        bcs = [self._bc_gamma_out] + self.dirichlet_bcs  # list of dirichletbc
        A_0 = dolfinx.fem.petsc.create_matrix(compiled_form)
        A_0.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A_0, compiled_form, bcs=bcs)
        A_0.assemble()

        self.operator = FenicsxMatrixOperator(
            A_0, V, V, solver_options=self.solver_options, name="A_0"
        )

    def discretize_neumann(self):
        """discretize inhomogeneous neumann bc(s)"""
        self.problem.clear_bcs(dirichlet=False)
        if self.neumann is not None:
            if isinstance(self.neumann, (list, tuple)):
                for force in self.neumann:
                    # force is dict
                    self.problem.add_neumann_bc(**force)
            else:
                self.problem.add_neumann_bc(**self.neumann)

        ufl_rhs = self.problem.get_form_rhs()
        compiled_form = dolfinx.fem.form(ufl_rhs)
        rhs_vector = dolfinx.fem.petsc.create_vector(compiled_form)
        rhs_vector.zeroEntries()
        dolfinx.fem.petsc.assemble_vector(rhs_vector, compiled_form)
        rhs_vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        rhs = self.source.make_array([rhs_vector])
        self._f_ext = rhs

    def discretize_rhs(self, boundary_data):
        """discretize the right hand side"""
        R = boundary_data
        assert R in self.source

        if not hasattr(self, "_A"):
            self.discretize_operator()

        if not hasattr(self, "_f_ext"):
            # assemble external force only once
            self.discretize_neumann()
        rhs = self._f_ext

        # subtract g(x_i) times the i-th column of A from the rhs
        # rhs = rhs - self._A.apply(R)
        AR = self._A.apply(R)
        AR.axpy(-1, rhs)
        rhs = -AR

        # set g(x_i) for i-th dof in rhs
        bcs = [self._bc_gamma_out] + self.dirichlet_bcs
        bc_dofs = np.array([], dtype=np.intc)
        for bc in bcs:
            dofs = bc.dof_indices()[0]
            bc_dofs = np.append(bc_dofs, dofs)
        bc_vals = R.dofs(bc_dofs)
        # FIXME workaround
        rhs_array = rhs.to_numpy()
        rhs_array[:, bc_dofs] = bc_vals

        return self.source.from_numpy(rhs_array)

    def get_correlation_matrix(self, clength, mean, metric="euclidean"):
        """get the correlation matrix"""
        V = self.source.V
        x = x_dofs_VectorFunctionSpace(V)
        sigma = correlation_matrix(x, clength, mean, distance_metric=metric)
        return sigma

    def generate_boundary_data(self, values):
        """generate boundary data g in V with ``values`` on Γ_out and zero elsewhere"""
        bc_dofs = self._bc_dofs_gamma_out
        assert values.shape[1] == len(bc_dofs)
        D = np.zeros((len(values), self.source.dim))
        D[:, bc_dofs] = values
        return self.source.from_numpy(D)

    def generate_random_boundary_data(
        self, count, distribution="normal", random_state=None, seed=None, **kwargs
    ):
        """generate random boundary data g in V with random values on Γ_out and zero elsewhere"""
        # initialize
        D = np.zeros((count, self.source.dim))  # source.dim is the full space

        # FIXME
        # there is self._source_gamma = NumpyVectorSpace()
        # but it cannot create random values for distribution='multivariate_normal'
        bc_dofs = self._bc_dofs_gamma_out  # actual size of the source space
        assert random_state is None or seed is None
        random_state = get_random_state(random_state, seed)
        values = _create_random_values(
            (count, bc_dofs.size), distribution, random_state, **kwargs
        )

        # set random data at boundary dofs
        D[:, bc_dofs] = values
        return self.source.from_numpy(D)

    def solve(self, boundary_data):
        """solve the problem for boundary_data

        Parameters
        ----------
        boundary_data : VectorArray
            Vectors in FenicsxVectorSpace(problem.V) with DoF entries holding
            values of boundary data on Γ_out and zero elsewhere.

        Returns
        -------
        U_in : VectorArray
            The solutions in the range space.
        """
        if not hasattr(self, "operator"):
            self.discretize_operator()
        # construct rhs from boundary data
        rhs = self.discretize_rhs(boundary_data)
        self.logger.info(f"Solving TransferProblem for {len(rhs)} vectors.")
        U = self.operator.apply_inverse(rhs)
        U_in = self.range.from_numpy(U.dofs(self._S_to_R))
        if self.remove_kernel:
            return orthogonal_part(self.kernel, U_in, self.range_l2_product, orth=True)
        else:
            return U_in

    def _get_source_product(self, product=None, bcs=(), product_name=None):
        """get source product

        Parameters
        ----------
        product : str, optional
            The inner product to use.
        bcs : list of df.DirichletBC, optional
            The bcs to be applied to the product matrix.
        product_name : str, optional
            Name of the NumpyMatrixOperator.

        Returns
        -------
        source_product : NumpyMatrixOperator or None
        """
        if product is not None:
            inner_product = InnerProduct(
                self.problem.V, product, bcs=bcs, name=product_name
            )
            M = inner_product.assemble_matrix()
            # FIXME figure out how to do this with PETSc.Mat and
            # use FenicsxMatrixOperator instead of NumpyMatrixOperator
            full_matrix = csc_matrix(M.getValuesCSR()[::-1], shape=M.size)
            dofs = self._bc_dofs_gamma_out
            source_matrix = full_matrix[dofs, :][:, dofs]
            source_product = NumpyMatrixOperator(source_matrix, name=product_name)
            return source_product
        else:
            return None

    def _get_range_product(self, product=None, bcs=(), product_name=None):
        """discretize range product

        Parameters
        ----------
        product : str, optional
            The inner product to use.
        bcs : list of df.DirichletBC, optional
            The bcs to be applied to the product matrix.
        product_name : str, optional
            Name of the FenicsMatrixOperator.

        Returns
        -------
        range_product : FenicsMatrixOperator or None
        """
        range_product = InnerProduct(self.range.V, product, bcs=bcs, name=product_name)
        matrix = range_product.assemble_matrix()
        if product is None:
            return None
        else:
            return FenicsxMatrixOperator(matrix, self.range.V, self.range.V)


class MultiscaleProblem(object):
    def __init__(self, mshfile):
        self.mshfile = mshfile
        domain, ct, ft = gmshio.read_from_msh(mshfile, MPI.COMM_WORLD, gdim=2)
        self.grid = StructuredQuadGrid(domain, ct, ft)
        self.grid_dir = pathlib.Path(mshfile).parent

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, yaml_file):
        with open(yaml_file, "r") as instream:
            mat = yaml.safe_load(instream)
        self._material = mat

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, degree):
        self._degree = int(degree)

    @property
    def cell_sets(self):
        raise NotImplementedError

    @property
    def boundaries(self):
        raise NotImplementedError

    def get_dirichlet(self):
        raise NotImplementedError

    def get_neumann(self):
        raise NotImplementedError

    def get_gamma_out(self):
        raise NotImplementedError

    def get_remove_kernel(self):
        raise NotImplementedError

    def get_pod_config(self):
        raise NotImplementedError
