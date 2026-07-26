"""``python -m chimeraboost`` — pre-compile the numba kernels."""

from .warmup import main

# Guarded so that merely importing this module (module enumeration, tooling)
# does not run the command.
if __name__ == "__main__":
    raise SystemExit(main())
