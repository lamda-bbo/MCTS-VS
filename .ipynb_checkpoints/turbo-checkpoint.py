from benchmark_problem import hartmann6, hartmann6_50, hartmann6_100, levy10, levy10_50, levy20, levy20_50, FunctionBenchmark
from synthetic_function import Hartmann, Levy
from baseline import Turbo1


# f = Hartmann(6, False)
# f = FunctionBenchmark(f, 50, list(range(6)))
f = Levy(10, False)
f = FunctionBenchmark(f, 50, list(range(10)))

turbo1 = Turbo1(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=20,  # Number of initial bounds from an Latin hypercube design
    max_evals = 1000,  # Maximum number of evaluations
    batch_size=10,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)

turbo1.optimize()