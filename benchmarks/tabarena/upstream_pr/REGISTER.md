# Wiring ChimeraBoost into a tabarena fork

These files are the model package for a `[New Model]` PR to
`github.com/autogluon/tabarena` (the OrionMSP PR #303 is the template).

## 1. Copy the package

Copy the `chimeraboost/` dir to `tabarena/tabarena/models/chimeraboost/` in your
fork. Auto-discovery (`tabarena/models/_registry.py::discover_models`) imports
each `models/<name>/info.py` and registers its `<name>_info`, so the package is
picked up without further wiring for the benchmark.

## 2. Register the public class export

For the public API surface (mirrors every other model), add `ChimeraBoostModel`
to `tabarena/tabarena/models/__init__.py`:

- In the `TYPE_CHECKING` block:
  ```python
  from tabarena.models.chimeraboost.model import ChimeraBoostModel
  ```
- In `_LAZY_CLASSES`:
  ```python
  "ChimeraBoostModel": "tabarena.models.chimeraboost.model",
  ```

## 3. Declare the pip dependency

ChimeraBoost is pip-installable (`pip install chimeraboost`); `pip_extra` in
`info.py` already declares it. Add it wherever the repo lists optional model
extras (e.g. the `pyproject.toml` model-extras group), matching how other
config models are listed.

## 4. Sanity-check in the fork

```python
from tabarena.models import discover_models
reg = discover_models()
assert "ChimeraBoost" in {m.method_metadata.method for m in reg.values()}

# default model unit test (AutoGluon FitHelper), all 3 problem types:
#   see benchmarks/tabarena/model_unittest.py in the chimeraboost repo
```

## Notes
* `model.py` here is identical to `benchmarks/tabarena/chimeraboost_tabarena_model.py`'s
  `ChimeraBoostModel` (the E10 / tuned subclasses and run-script config helpers
  are not part of the upstream package).
* `hpo.py` carries the tuned search space (for the later tuned row); the default
  row uses only `manual_configs=[{}]`.
* Metadata fields left unset (`has_raw/has_processed/has_results`, `s3_*`,
  `cache_type`) are filled by maintainers when result artifacts are hosted in the
  official pool.
