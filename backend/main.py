import sys
import os
import importlib.util

base_dir = os.path.dirname(os.path.abspath(__file__))
real_backend = os.path.join(base_dir, "..", "generative-ai-lab", "backend")
real_backend = os.path.abspath(real_backend)

if real_backend not in sys.path:
    sys.path.insert(0, real_backend)

spec = importlib.util.spec_from_file_location("real_main_module", os.path.join(real_backend, "main.py"))
real_main_module = importlib.util.module_from_spec(spec)
sys.modules["real_main_module"] = real_main_module
spec.loader.exec_module(real_main_module)

app = real_main_module.app
