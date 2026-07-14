"""SynthGen: prior-sampled synthetic benchmark suite (decision tier 1).

Deterministic numpy-only SCM-prior datasets, calibrated to harvested public
dataset metadata (TabArena excluded -- sealed holdout). See the plan notes in
docs/PROJECT_STATUS.md and the /experiment skill for how verdicts are used.
"""
from .api import (build_dataset, hash_dataset, key_for, make_builder,  # noqa: F401
                  parse_key, recipe_meta, sample_recipe, task_of)
from .recipe import VERSION  # noqa: F401
from .suites import SUITES, all_frozen_keys, frozen_keys  # noqa: F401
