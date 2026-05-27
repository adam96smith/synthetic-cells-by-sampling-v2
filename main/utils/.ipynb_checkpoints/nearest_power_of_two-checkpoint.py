import math

def nearest_power_of_two(x: float, prefer_higher_on_tie: bool = False):
    
    assert x > 0
    
    # log2
    z_floor = math.floor(math.log2(x))
    if z_floor < 1:
        z_floor = 1
    z_ceil = z_floor if 2**z_floor >= x else z_floor + 1

    y_floor = 2**z_floor
    y_ceil = 2**z_ceil

    # distances
    d_floor = abs(x - y_floor)
    d_ceil = abs(y_ceil - x)

    if d_floor < d_ceil:
        return y_floor
    elif d_ceil < d_floor:
        return y_ceil
    else:
        # tie
        if prefer_higher_on_tie:
            return y_ceil
        else:
            return y_floor