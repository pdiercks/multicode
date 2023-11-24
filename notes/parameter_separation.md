# Parameter separation

## Initial design

```python
class ParameterSeparatedLinearElasticProblem(LinearProblem):
    """Represents parameter separated linear problems.
    This class should be used to create |Operators| such that
    these can be used in conjunction with a |StationaryProblem|.

    """

    def __init__(self, domain: Domain, space: fem.FunctionSpaceBase, phases: tuple[LinearElasticMaterial]):
        """Initializes a parameter separated problem."""
        super().__init__(domain, space)
        self.gdim = domain.grid.ufl_cell().geometric_dimension()
        self.phases = phases

    # FIXME if the material handles partly the affine decomposition
    # then I could simply add the below methods to LinearElasticityProblem ...

    def assemble_operators(self) -> list[Operator]:
        """Assembles operators in an affine expansion of the left hand side.
        Can be used to define a `pymor.operators.constructions.LincombOperator`."""
        raise NotImplementedError

    def assemble_rhs(self) -> list[Operator]:
        raise NotImplementedError

```

## Conclusion

This did not add much value, because each parameter separated form is problem dependent
and would require a new implementation.
