from gp import build_gp

## Loss function
def loss_func(params, X, y):
    gp = build_gp(params, X)
    return -gp.log_probability(y)