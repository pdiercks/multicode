import dolfin as df
import numpy as np
from multi.bcs import BoundaryConditions
from multi.materials import LinearElasticMaterial
from multi.misc import make_mapping
from multi.product import InnerProduct
from multi.solver import create_solver

from pymor.core.logger import getLogger
from pymor.bindings.fenics import FenicsMatrixOperator, FenicsVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.numpy import NumpyMatrixOperator
from scipy.sparse import csc_matrix


class LinearProblemBase(object):

    """Docstring for LinearProblemBase."""

    def __init__(self, domain, V):
        """TODO: to be defined.

        Parameters
        ----------
        domain : TODO
        V : TODO

        """
        self.logger = getLogger("multi.problems.LinearProblemBase")
        self.domain = domain
        self.V = V
        self._bc_handler = BoundaryConditions(domain, V)

    def add_dirichlet_bc(
        self, boundary, value, degree=0, sub=None, method="topological"
    ):
        self._bc_handler.add_dirichlet(boundary, value, degree, sub, method)

    def add_neumann_bc(self, boundary, value, degree=0):
        self._bc_handler.add_neumann(boundary, value, degree)

    def clear_bcs(self, dirichlet=True, neumann=True):
        """remove all dirichlet and neumann bcs"""
        self._bc_handler.clear(dirichlet=dirichlet, neumann=neumann)

    def dirichlet_bcs(self):
        return self._bc_handler.bcs()

    def discretize_product(self, product, bcs=False):
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
            bcs = self._bc_handler.bcs()
            if not len(bcs) > 0:
                raise Warning("Forgot to apply BCs?")
        else:
            bcs = ()
        if isinstance(product, str):
            product = InnerProduct(self.V, name=product, bcs=bcs, form=None)
        else:
            product = InnerProduct(self.V, name=None, bcs=bcs, form=product)
        return product.assemble() if product else None

    def get_form_lhs(self):
        # to be implemented by child
        raise NotImplementedError

    def get_form_rhs(self):
        # to be implemented by child
        raise NotImplementedError

    def solve(self, x=None, solver_options=None):
        """performs single solve"""
        matrix = df.assemble(self.get_form_lhs())
        rhs_vector = df.assemble(self.get_form_rhs())
        bcs = self.dirichlet_bcs()
        if len(bcs) < 1:
            self.logger.warning("No dirichlet bcs defined for this problem...")
        for bc in bcs:
            bc.zero_columns(matrix, rhs_vector, 1.0)

        solver = create_solver(matrix, solver_options=solver_options)
        if x:
            solver.solve(x.vector(), rhs_vector)
        else:
            x = df.Function(self.V)
            solver.solve(x.vector(), rhs_vector)
            return x


class LinearElasticityProblem(LinearProblemBase):
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

    """

    def __init__(self, domain, V, E=210e3, NU=0.3, plane_stress=False):
        super().__init__(domain, V)
        assert all(
            [isinstance(E, (float, tuple, list)), isinstance(NU, (float, tuple, list))]
        )
        if isinstance(E, float) and isinstance(NU, float):
            E = (E,)
            NU = (NU,)
            assert not domain.subdomains
            self.dx = df.dx
        else:
            if not domain.subdomains and len(E) > 1:
                raise KeyError(
                    "You need to define a df.MeshFunction for multiple materials"
                )
            assert all(
                [len(E) == len(NU), len(E) == np.unique(domain.subdomains.array()).size]
            )
            # pygmsh version 6.1.1 convention
            assert np.amin(domain.subdomains.array()) > 0
            mesh = domain.mesh
            subdomains = domain.subdomains
            self.dx = df.Measure("dx", domain=mesh, subdomain_data=subdomains)
        self.u = df.TrialFunction(V)
        self.v = df.TestFunction(V)
        self.gdim = V.element().geometric_dimension()
        self.materials = [
            LinearElasticMaterial(self.gdim, E=e, NU=nu, plane_stress=plane_stress)
            for e, nu in zip(E, NU)
        ]

    def get_form_lhs(self):
        """get bilinear form a(u, v) of the problem"""
        u = self.u
        v = self.v
        if len(self.materials) > 1:
            return sum(
                [
                    df.inner(mat.sigma(u), mat.eps(v)) * self.dx(i + 1)
                    for (i, mat) in enumerate(self.materials)
                ]
            )
        else:
            mat = self.materials[0]
            return df.inner(mat.sigma(u), mat.eps(v)) * self.dx

    def get_form_rhs(self, body_forces=None):
        """get linear form f(v) of the problem"""
        v = self.v
        zero = (0.0,) * self.gdim
        rhs = df.dot(df.Constant(zero), v) * df.dx
        if body_forces is not None:
            if len(self.materials) > 1:
                assert isinstance(body_forces, (list, tuple))
                assert len(body_forces) == len(self.materials)
                for i in range(len(self.materials)):
                    rhs += df.dot(body_forces[i], v) * self.dx(i + 1)
            else:
                rhs += df.dot(body_forces, v) * self.dx

        if self._bc_handler.has_neumann():
            rhs += self._bc_handler.neumann_bcs()

        return rhs


class OversamplingProblem(object):
    """General class for oversampling problems.

    This class aids the solution of oversampling problems given by:

        A(u) = 0 in Ω,
        with homogeneous dirichlet bcs on Γ_D,
        with homogeneous neumann bcs on Γ_N,
        with inhomogeneous neumann bcs on Γ_N_inhom,
        with arbitrary dirichlet boundary conditions on Γ_out.

    Here, we are only interested in the solution u restricted to the
    space defined on a target subdomain Ω_in ⊂ Ω.

    The boundaries Γ_D, Γ_N and Γ_N_inhom are the part of ∂Ω that
    intersects with the (respective dirichlet or neumann) boundary
    of the global domain Ω_gl ( Ω ⊂ Ω_gl).
    The boundary Γ_out is the part of ∂Ω that does not intersect with ∂Ω_gl.

    Note: The above problem can be formulated as a transfer operator which
    maps the boundary data on Γ_out (source space) to the solution u in
    the (range) space defined on Ω_in. Since fenics (version 2019.1.0) does
    not allow to define function spaces on some part of the boundary of a domain,
    the full space is defined as the source space. The range space is the
    space defined on Ω_in.

    Parameters
    ----------
    problem : multi.problems.LinearProblemBase
        The problem defined on the oversampling domain Ω.
    subdomain_problem : multi.problems.LinearProblemBase
        The problem defined on the target subdomain Ω_in.
    gamma_out : df.SubDomain
        The part of the boundary of the oversampling domain that does
        not intersect with the boundary of the global domain.
    dirichlet : list of dict or dict, optional
        Homogeneous or inhomogeneous dirichlet boundary conditions.
        See multi.bcs.BoundaryConditinos.add_dirichlet_bc for suitable values.
    neumann : list of dict or dict, optional
        Inhomogeneous neumann boundary conditions.
        See multi.bcs.BoundaryConditions.add_neumann_bc for suitable values.
    solver_options : dict, optional
        The user is required to use pymor style options ({'inverse': options}),
        because the value of solver_options is directly passed to FenicsMatrixOperator.
        See https://github.com/pymor/pymor/blob/main/src/pymor/operators/interface.py#L32-#L41

    """

    def __init__(
        self,
        problem,
        subdomain_problem,
        gamma_out,
        dirichlet=None,
        neumann=None,
        solver_options=None,
    ):
        self.logger = getLogger("multi.problems.OversamplingProblem")
        self.problem = problem
        self.subdomain_problem = subdomain_problem
        self.source = FenicsVectorSpace(problem.V)
        self.range = FenicsVectorSpace(subdomain_problem.V)
        self.gamma_out = gamma_out
        self.dirichlet = dirichlet
        self.neumann = neumann
        self.solver_options = solver_options
        # initialize commonly used quantities
        self._init_bc_gamma_out()
        self._S_to_R = self._make_mapping()

    def _init_bc_gamma_out(self):
        """define bc on gamma out"""
        V = self.source.V
        dummy = df.Function(V)
        self._bc_gamma_out = df.DirichletBC(V, dummy, self.gamma_out)
        self._bc_dofs_gamma_out = list(self._bc_gamma_out.get_boundary_values().keys())
        # source space restricted to Γ_out
        self._source_gamma = NumpyVectorSpace(len(self._bc_dofs_gamma_out))

    def _make_mapping(self):
        """builds map from source space to range space"""
        return make_mapping(self.range.V, self.source.V)

    def discretize_operator(self):
        """discretize the operator"""
        self.logger.info(
            f"Discretizing left hand side of the problem (size={self.problem.V.dim()})."
        )
        matrix = df.PETScMatrix()
        df.assemble(self.problem.get_form_lhs(), tensor=matrix)
        # make copy and wrap as FenicsMatrixOperator for construction of rhs
        V = self.source.V
        # A refers to full operator without bcs applied
        self._A = FenicsMatrixOperator(
            matrix.copy(), V, V, solver_options=self.solver_options, name="A"
        )

        # ### apply bcs to operator
        # make sure there are no unwanted bcs present
        self.problem.clear_bcs(neumann=False)
        if self.dirichlet is not None:
            if isinstance(self.dirichlet, (list, tuple)):
                for dirichlet_bc in self.dirichlet:
                    self.problem.add_dirichlet_bc(**dirichlet_bc)
            else:
                self.problem.add_dirichlet_bc(**self.dirichlet)
        bcs = [self._bc_gamma_out] + self.problem.dirichlet_bcs()
        dummy = df.Function(V)
        for bc in bcs:
            bc.zero_columns(matrix, dummy.vector(), 1.0)

        # A_0 refers to operator with bcs applied
        self.operator = FenicsMatrixOperator(
            matrix, V, V, solver_options=self.solver_options, name="A_0"
        )

    def discretize_neumann(self):
        """discretize inhomogeneous neumann bc(s)"""
        self.problem.clear_bcs(dirichlet=False)
        if self.neumann is not None:
            if isinstance(self.neumann, (list, tuple)):
                for force in self.neumann:
                    self.problem.add_neumann_bc(**force)
            else:
                self.problem.add_neumann_bc(**self.neumann)

        # will always be null vector in case neumann is None see LinearElasticityProblem.get_rhs
        rhs = self.source.make_array([df.assemble(self.problem.get_form_rhs())])
        self._f_ext = rhs

    def discretize_rhs(self, boundary_data):
        """discretize the right hand side"""
        R = boundary_data
        assert R in self.source

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
        bc_dofs = list(self._bc_gamma_out.get_boundary_values().keys())
        bc_vals = R.dofs(bc_dofs)
        # workaround
        rhs_array = rhs.to_numpy()
        rhs_array[:, bc_dofs] = bc_vals

        return self.source.from_numpy(rhs_array)

    def generate_random_boundary_data(
        self, count, distribution="normal", random_state=None, seed=None
    ):
        """generate random boundary data"""
        # initialize
        D = np.zeros((count, self.source.dim))

        bc_dofs = self._bc_dofs_gamma_out
        random_values = self._source_gamma.random(
            count, distribution=distribution, random_state=random_state, seed=seed
        )
        # set random data at boundary dofs
        D[:, bc_dofs] = random_values.to_numpy()
        return self.source.from_numpy(D)

    def solve(self, boundary_data):
        """solve the problem for boundary_data

        Parameters
        ----------
        boundary_data : VectorArray
            Vectors in FenicsVectorSpace(problem.V) with DoF entries holding
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
        self.logger.info(f"Solving OversamplingProblem for {len(rhs)} vectors.")
        U = self.operator.apply_inverse(rhs)
        return self.range.from_numpy(U.dofs(self._S_to_R))

    def get_source_product(self, product=None):
        """get source product

        Parameters
        ----------
        product : str, optional
            The inner product to use.

        Returns
        -------
        source_product : NumpyMatrixOperator or None
        """
        if product is not None:
            # compute source product
            matrix = self.problem.discretize_product(name=product, bcs=False)
            M = df.as_backend_type(matrix).mat()
            # FIXME figure out how to use (take slice of) dolfin matrix directly?
            full_matrix = csc_matrix(M.getValuesCSR()[::-1], shape=M.size)
            dofs = self._bc_dofs_gamma_out
            source_matrix = full_matrix[dofs, :][:, dofs]
            source_product = NumpyMatrixOperator(
                source_matrix, name=f"{product}_product"
            )
            return source_product
        else:
            return None

    def get_range_product(self, product=None):
        """get range product

        Parameters
        ----------
        product : str, optional
            The inner product to use.

        Returns
        -------
        range_product : FenicsMatrixOperator or None
        """
        if product is not None:
            # compute range product
            matrix = self.subdomain_problem.get_product(name=product, bcs=False)
            V = self.range.V
            range_product = FenicsMatrixOperator(
                matrix, V, V, name=f"{product}_product"
            )
            return range_product
        else:
            return None
