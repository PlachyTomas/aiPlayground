import importlib
import pkgutil
import sys

import visionsuite_core


def test_core_imports_no_web_framework():
    for mod in pkgutil.walk_packages(visionsuite_core.__path__, "visionsuite_core."):
        importlib.import_module(mod.name)
    forbidden = {"fastapi", "starlette", "uvicorn"}
    assert forbidden.isdisjoint(sys.modules.keys())
