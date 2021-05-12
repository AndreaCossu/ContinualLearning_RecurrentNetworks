import enum


@enum.unique
class OUTPUT_TYPE(enum.Enum):
    """
    Define what kind of output each model produces
    NOTHING -> return without arguments
    ALL_OUTS = 1 -> return entire output tensor
    LAST_OUT = 2 -> return last vector in the second dimension of output tensor
    ALL_HS = 3 -> return entire hidden state tensor
    LAST_H = 4 -> return last vector in the second dimension of hidden state
    ALL_OUTH = 5 -> return both 1 and 3
    LAST_OUTH = 6 -> return both 2 and 4

    Note that feedforward models are not compatible with LAST_OUTH, LAST_H.
    """

    NOTHING = 0
    ALL_OUTS = 1
    LAST_OUT = 2
    ALL_HS = 3
    LAST_H = 4
    ALL_OUTH = 5
    LAST_OUTH = 6

def choose_output(outs, hs, mode):
    if mode == OUTPUT_TYPE.NOTHING:
        return
    if mode == OUTPUT_TYPE.ALL_OUTS:
        return outs
    if mode == OUTPUT_TYPE.LAST_OUT:
        return outs[:, -1]
    if mode == OUTPUT_TYPE.ALL_HS:
        return hs
    if mode == OUTPUT_TYPE.LAST_H:
        return hs[:, -1]
    if mode == OUTPUT_TYPE.ALL_OUTH:
        return outs, hs
    if mode == OUTPUT_TYPE.LAST_OUTH:
        return outs[:, -1], hs[:, -1]
    
    raise ValueError(f"INVALID OUTPUT_TYPE CONSTANT: {mode}.")