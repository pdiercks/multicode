"""common helpers"""


def to_floats(values):
    floats = []
    try:
        for v in values:
            floats.append(float(v))
    except TypeError as e:
        floats = [float(values)]

    return floats
