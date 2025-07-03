import sys
from importlib import reload

def reload_modules(
  project_root: str,
):
  print("Reloading modules to ensure the latest code is used.")
  
  for module_name, module in list(sys.modules.items()):
    if module_name in ["__main__", "__mp_main__"]:
      continue
    if hasattr(module, '__file__') is False or module.__file__ is None:
      continue

    if module.__file__.startswith(project_root):
      print(f"Reloading module: {module_name}")
      reload(module)