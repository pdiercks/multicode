import dolfin as df


class DisplacementSolution(df.UserExpression):
    def __init__(self, solution, **kwargs):
        super().__init__(**kwargs)
        self.solution = solution

    def eval(self, value, x):
        ux, uy = self.solution.displacement(x)
        value[0] = ux
        value[1] = uy

    def value_shape(self):
        return (2,)


class StrainSolution(df.UserExpression):
    def __init__(self, solution, **kwargs):
        super().__init__(**kwargs)
        self.solution = solution

    def eval(self, value, x):
        exx, eyy, exy = self.solution.strain(x)

        value[0] = exx
        value[1] = eyy
        value[2] = exy

    def value_shape(self):
        return (3,)


class StressSolution(df.UserExpression):
    def __init__(self, solution, **kwargs):
        super().__init__(**kwargs)
        self.solution = solution

    def eval(self, value, x):
        sxx, syy, sxy = self.solution.stress(x)

        value[0] = sxx
        value[1] = syy
        value[2] = sxy

    def value_shape(self):
        return (3,)
