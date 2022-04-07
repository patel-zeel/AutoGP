import jax
import jax.numpy as jnp
from itertools import product
from tinygp import GaussianProcess
from functools import partial
from multiprocessing import set_start_method
from loss import loss_func
from gp import build_gp
from jaxopt import ScipyMinimize
set_start_method("spawn")
from multiprocessing import Pool
import matplotlib.pyplot as plt
import optax


def plot_posterior(params, X_train, y_train, X_test, y_test, save_name):
    gp = build_gp(params, X_train)
    noise = params[-1]
    post_gp = gp.condition(y_train, X_test, diag=jnp.exp(noise)).gp
    mean = post_gp.mean
    std = post_gp.variance**0.5

    plt.figure(figsize=(10, 4))
    plt.scatter(X_train, y_train, label="Train", s=5)
    plt.scatter(X_test, y_test, label="Test", s=5)
    plt.plot(X_test, mean, label="Pred")
    plt.fill_between(X_test.flatten(), mean - 2 * std, mean + 2 * std, label="$\mu \pm 2\sigma$", alpha=0.5)
    plt.title(parse_kernel(params[0]))
    plt.legend()
    # plt.savefig(save_name+'.jpg')


## Pretty print kernel combination
def parse_kernel(kernel):
    if kernel.__class__.__name__ == "Product":
        if kernel.kernel1.__class__.__name__ == "Constant":
            return kernel.kernel2.__class__.__name__
        return f"({parse_kernel(kernel.kernel1)} x {parse_kernel(kernel.kernel2)})"
    elif kernel.__class__.__name__ == "Sum":
        return f"({parse_kernel(kernel.kernel1)} + {parse_kernel(kernel.kernel2)})"
    else:
        return kernel.__class__.__name__

def mul_op(k1, k2): return k1 * k2
def sum_op(k1, k2): return k1 + k2

def forward_search_single(experiment, old_kernel, x, y, initializer, n_iter, re_init):
    key, new_kernel, operation = experiment
    if old_kernel is None:
        kernel = new_kernel
    else:
        kernel = operation(old_kernel, new_kernel)
    
    params = (kernel, 0.1, 0.1)
    len_params = len(jax.tree_util.tree_leaves(params))
    random_params = jnp.log(jax.random.truncated_normal(key, lower=0.01, upper=10.0, shape=(len_params, )))
    tree_def = jax.tree_util.tree_structure(params)
    params = jax.tree_util.tree_unflatten(tree_def, random_params)

    tx = optax.adam(learning_rate=0.1)
    state = tx.init(params)
    loss_and_grad_func = jax.jit(jax.value_and_grad(loss_func))
    for _ in range(n_iter):
        value, grads = loss_and_grad_func(params, x, y)
        updates, state = tx.update(grads, state)
        params = optax.apply_updates(params, updates)

    value, _ = loss_and_grad_func(params, x, y)
    return value, params
    # optimizer = ScipyMinimize(method="L-BFGS-B", fun=loss_func)
    # solution = optimizer.run(params, x, y)
    # return solution

def forward_search(x, y, x_test, y_test, kernel_list, depth=1, n_iter=100, n_restarts=5, key_start=0, n_jobs=1, save_name="old"):
    keys = [jax.random.PRNGKey(key_start + i) for i in range(n_restarts)]
    operations = [mul_op, sum_op]
    
    old_kernel = None
    for depth in range(3):
        print(f"Experiment started at depth {depth}")
        loggers = []
        for experiment in product(keys, kernel_list, operations):
            loggers.append(forward_search_single(experiment, old_kernel, x, y, 
            initializer=None, n_iter=100, re_init=False))
        
        # return loggers
        # losses = jnp.array([solution.state.fun_val for solution in loggers])
        # params = [solution.params for solution in loggers]
        losses = jnp.array([solution[0] for solution in loggers])
        params = [solution[1] for solution in loggers]
        best_idx = jnp.argmin(losses)
        best_params = params[best_idx]
        old_kernel = best_params[0]
        print(f"Best kernel at depth {depth} is {parse_kernel(old_kernel)}")
        
        plot_posterior(best_params, x, y, x_test, y_test, save_name+f"_depth_{depth}")