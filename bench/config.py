"""Configuration constants for benchmark defaults and validation thresholds."""

DEFAULT_BATCH_SIZES = (1, 32, 128)
DEFAULT_WARMUP = 20
DEFAULT_ITERS = 100
DEFAULT_SEED = 42
DEFAULT_INPUT_SHAPE = (1, 32, 32)
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "float32"

ABS_ERR_TOL = 1e-4
REL_ERR_TOL = 1e-3

DEFAULT_RESULTS_DIR = "results"
