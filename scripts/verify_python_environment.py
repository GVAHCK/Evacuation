"""Production Environment and Dependency Verification Script.
Validates all required production dependencies and ML/Backend runtimes.
"""

import sys

DEPENDENCIES = [
    ("torch", "PyTorch"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("pandas", "Pandas"),
    ("gymnasium", "Gymnasium"),
    ("stable_baselines3", "Stable-Baselines3"),
    ("networkx", "NetworkX"),
    ("osmnx", "OSMnx"),
    ("shapely", "Shapely"),
    ("pyproj", "PyProj"),
    ("geopandas", "GeoPandas"),
    ("traci", "TraCI"),
    ("sumolib", "SUMOLib"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("sqlalchemy", "SQLAlchemy"),
    ("psycopg2", "psycopg2"),
    ("pydantic", "Pydantic"),
    ("jwt", "PyJWT"),
    ("websockets", "WebSockets"),
]

def verify_all():
    print("=" * 60, flush=True)
    print("VERIFYING PYTHON PRODUCTION DEPENDENCIES", flush=True)
    print("=" * 60, flush=True)
    
    all_ok = True
    for mod_name, label in DEPENDENCIES:
        print(f"Checking {label} ({mod_name})...", flush=True)
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"[PASS] {label:<20} ({mod_name:<18}) version: {ver}", flush=True)
        except ImportError as e:
            print(f"[FAIL] {label:<20} ({mod_name:<18}) ERROR: {e}", flush=True)
            all_ok = False
        except Exception as e:
            print(f"[WARN] {label:<20} ({mod_name:<18}) LOAD WARNING: {e}", flush=True)

    print("-" * 60, flush=True)
    if all_ok:
        print("ALL PRODUCTION DEPENDENCIES IMPORTED SUCCESSFULLY!", flush=True)
    else:
        print("SOME DEPENDENCIES FAILED TO IMPORT!", flush=True)
    print("-" * 60, flush=True)
    return all_ok

if __name__ == "__main__":
    success = verify_all()
    sys.exit(0 if success else 1)
