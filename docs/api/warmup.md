# warmup

Compile the numba kernels up front, so the first `fit` or `predict` in a fresh process
is not slow. See [Deployment](../deployment.md#the-first-call-pays-the-numba-compile).

::: chimeraboost.warmup
    options:
      show_root_heading: false
      show_root_toc_entry: false
