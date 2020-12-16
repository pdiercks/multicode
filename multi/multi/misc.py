"""miscellaneous helpers"""
import dolfin as df


def make_mapping(sub_space, super_space):
    """get map from sub to super space

    Parameters
    ----------
    sub_space
        A dolfin.FunctionSpace
    super_space
        A dolfin.FunctionSpace

    Returns
    -------
    The dofs of super_space located at entities in sub_space.

    Note: This only works for conforming meshes.
    """
    f = df.Function(super_space)
    f.vector().set_local(super_space.dofmap().dofs())
    If = df.interpolate(f, sub_space)
    return (If.vector().get_local() + 0.5).astype(int)
