"""Compatibility package that forwards notebook-style imports to ``src.nqs``."""

from importlib import import_module

import src.nqs as _impl

# Reuse src.nqs as the package search root so nqs.<module> imports resolve
# without maintaining a second set of wrapper files.
__path__ = list(_impl.__path__)

_FORWARDED_SUBMODULES = (
    "driver",
    "expectation",
    "exact_diag",
    "graph",
    "hilbert",
    "models",
    "observables",
    "operator",
    "optimizer",
    "sampler",
    "vmc_setup",
    "vqs",
)

for export_name in _impl.__all__:
    globals()[export_name] = getattr(_impl, export_name)

driver = import_module("src.nqs.driver")
expectation = import_module("src.nqs.expectation")
exact_diag = import_module("src.nqs.exact_diag")
graph = import_module("src.nqs.graph")
hilbert = import_module("src.nqs.hilbert")
models = import_module("src.nqs.models")
observables = import_module("src.nqs.observables")
operator = import_module("src.nqs.operator")
optimizer = import_module("src.nqs.optimizer")
sampler = import_module("src.nqs.sampler")
vmc_setup = import_module("src.nqs.vmc_setup")
vqs = import_module("src.nqs.vqs")

__all__ = [*_impl.__all__, *_FORWARDED_SUBMODULES]
