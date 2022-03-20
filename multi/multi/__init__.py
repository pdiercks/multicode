name = "multi"
from multi.bcs import BoundaryConditions, apply_bcs
from multi.dofmap import DofMap
from multi.domain import Domain, RectangularDomain
from multi.io import ResultFile
from multi.linear_elasticity import LinearElasticMaterial, LinearElasticityProblem
from multi.misc import make_mapping
from multi.product import InnerProduct
