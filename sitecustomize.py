import sys
import os

workspace_root = os.path.dirname(os.path.abspath(__file__))
venv_site = os.path.join(workspace_root, ".venv", "Lib", "site-packages")
backend_dir = os.path.join(workspace_root, "backend")

if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if venv_site not in sys.path:
    # Append after uv python's site-packages so updated pandas/fastapi takes precedence
    sys.path.append(venv_site)

# Set SUMO_HOME if not already set
sumo_dir = os.path.join(venv_site, "sumo")
if os.path.isdir(sumo_dir) and "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = sumo_dir
    os.environ["PATH"] = os.path.join(sumo_dir, "bin") + os.pathsep + os.environ.get("PATH", "")
