import numpy as np
import dolfin as df
from multi.solver import create_solver, _solver_options
from pymor.core.logger import getLogger
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.bindings.fenics import FenicsMatrixOperator, FenicsVectorSpace

# discretize oversampling problem and return FenicsMatrixOperator
# for LHS (with bcs applied) and to construct later RHS
# --> LHS can make use of all that solver stuff implemented for FenicsMatrixOperator
# then only need to create appropriate VectorArray that contains all
# right hand side vectors for which the problem should be solved


class OversamplingProblem(object):

    """Docstring for OversamplingProblem."""

    def __init__(
        self, problem, gamma_out, dirichlet=None, neumann=None, solver_options=None
    ):
        """TODO: to be defined.

        Parameters
        ----------
        problem : multi.linear_elasticity.LinearElasticityProblem
            The problem defined on the oversampling domain.
        gamma_out : df.SubDomain
            The part of the boundary of the oversampling domain that does
            not intersect with the boundary of the global domain.
        dirichlet : list of dict or dict, optional
            See multi.bcs.MechanicsBCs.add_bc for suitable values.
        neumann : list of dict or dict, optional
            See multi.bcs.MechanicsBCs.add_force for suitable values.
        solver_options : dict, optional
            The user is required to use pymor style options ({'inverse': options}),
            because the value of solver_options is directly passed to FenicsMatrixOperator.


        """
        self._logger = getLogger("multi.oversampling.OversamplingProblem")
        self.problem = problem
        self.source = FenicsVectorSpace(problem.V)
        self.gamma_out = gamma_out
        self.dirichlet = dirichlet
        self.neumann = neumann
        self.solver_options = solver_options

    def _init_bc_gamma_out(self):
        """define bc on gamma out"""
        V = self.problem.V
        dummy = df.Function(V)
        self._bc_gamma_out = df.DirichletBC(V, dummy, self.gamma_out)
        self._bc_dofs_gamma_out = list(self._bc_gamma_out.get_boundary_values().keys())

    def _discretize_operator(self):
        """discretize the operator"""
        self._logger.info(
            f"Discretizing left hand side of the problem (size={self.problem.V.dim()})."
        )
        matrix = df.PETScMatrix()
        df.assemble(self.problem.get_lhs(), tensor=matrix)
        # make copy and wrap as FenicsMatrixOperator for construction of rhs
        V = self.problem.V
        # A refers to full operator without bcs applied
        self._A = FenicsMatrixOperator(
            matrix.copy(), V, V, solver_options=self.solver_options, name="A"
        )

        # ### apply bcs to operator
        if not hasattr(self, "_bc_gamma_out"):
            self._init_bc_gamma_out()

        # make sure there are no unwanted bcs present
        self.problem.bc_handler.remove_bcs()
        if self.dirichlet is not None:
            if isinstance(self.dirichlet, list):
                for dirichlet_bc in self.dirichlet:
                    self.problem.bc_handler.add_bc(**dirichlet_bc)
            else:
                self.problem.bc_handler.add_bc(**self.dirichlet)
        bcs = [self._bc_gamma_out] + self.problem.bc_handler.bcs()
        dummy = df.Function(V)
        for bc in bcs:
            bc.zero_columns(matrix, dummy.vector(), 1.0)

        # A_0 refers to operator with bcs applied
        self.operator = FenicsMatrixOperator(
            matrix, V, V, solver_options=self.solver_options, name="A_0"
        )

    def _discretize_rhs(self, boundary_data):
        """discretize the right hand side"""
        # R = self.source.make_array(boundary_data)
        R = boundary_data
        assert R in self.source

        # inhomogeneous neumann bc(s)
        self.problem.bc_handler.remove_forces()
        if self.neumann is not None:
            if isinstance(self.neumann, list):
                for force in self.neumann:
                    self.problem.bc_handler.add_force(**force)
            else:
                self.problem.bc_handler.add_force(**self.neumann)

        # will always be zero in case neumann is None see LinearElasticityProblem.get_rhs
        rhs = self.source.make_array([df.assemble(self.problem.get_rhs())])

        # subtract g(x_i) times the i-th column of A from the rhs
        # rhs = rhs - self._A.apply(R)
        AR = self._A.apply(R)
        AR.axpy(-1, rhs)
        rhs = -AR
        # ensure len(rhs) == len(R)

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

        if not hasattr(self, "_bc_gamma_out"):
            self._init_bc_gamma_out()

        # use NumpyVectorSpace to create random values
        bc_dofs = self._bc_dofs_gamma_out
        space = NumpyVectorSpace(len(bc_dofs))
        random_values = space.random(
            count, distribution=distribution, random_state=random_state, seed=seed
        )
        # set random data at boundary dofs
        D[:, bc_dofs] = random_values.to_numpy()
        return self.source.from_numpy(D)

    # TODO how is the boundary data best created?
    # 1. data comes from projection of SoI data into full space
    # 2. random data on Γ_out
    def solve(self, boundary_data):
        """solve the problem for boundary_data

        Parameters
        ----------
        boundary_data : VectorArray
            Vectors in FenicsVectorSpace(problem.V) with DoF entries holding
            values of boundary data on Γ_out and zero elsewhere.

        Returns
        -------
        U : VectorArray
            The solutions in the full space.
        """
        if not hasattr(self, "operator"):
            self._discretize_operator()
        # construct rhs from boundary data
        rhs = self._discretize_rhs(boundary_data)
        self._logger.info(f"Solving OversamplingProblem for {len(rhs)} vectors.")
        U = self.operator.apply_inverse(rhs)
        return U
