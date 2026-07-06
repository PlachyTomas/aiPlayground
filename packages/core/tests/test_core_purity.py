import subprocess
import sys


def test_core_imports_no_web_framework():
    # Fresh interpreter: independent of what the rest of the session imported.
    code = (
        "import importlib, pkgutil, sys\n"
        "import visionsuite_core\n"
        "for mod in pkgutil.walk_packages(visionsuite_core.__path__, 'visionsuite_core.'):\n"
        "    importlib.import_module(mod.name)\n"
        "forbidden = {'fastapi', 'starlette', 'uvicorn', 'label_studio_sdk'}\n"
        "loaded = sorted(forbidden & sys.modules.keys())\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
