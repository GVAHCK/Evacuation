"""SUMO realistic traffic demand generator across multiple density regimes."""

import os
import sys
import subprocess
import yaml
from typing import Dict, Any, Optional
from src.utils.logger import setup_logger

logger = setup_logger("TrafficGenerator")


class TrafficGenerator:
    """Generates reproducible multi-density traffic scenarios (low, normal, peak, heavy)

    using SUMO randomTrips.py and configurable flow parameters.
    """

    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.net_file = self.config.get("network", {}).get("net_file", "data/processed/network.net.xml")
        self.traffic_cfg = self.config.get("traffic", {})
        self.densities = self.traffic_cfg.get("densities", {})
        self.sim_duration = self.traffic_cfg.get("sim_duration", 3600)
        self.output_dir = "data/generated/traffic"

        # Locate randomTrips.py
        try:
            import sumolib
            self.random_trips_py = os.path.abspath(
                os.path.join(os.path.dirname(sumolib.__file__), "..", "sumo", "tools", "randomTrips.py")
            )
            if not os.path.isfile(self.random_trips_py):
                # Check virtualenv tools location
                self.random_trips_py = os.path.abspath(".venv/Lib/site-packages/sumo/tools/randomTrips.py")
        except Exception:
            self.random_trips_py = os.path.abspath(".venv/Lib/site-packages/sumo/tools/randomTrips.py")

    def generate_scenario(
        self,
        traffic_level: str = "normal",
        seed: int = 42,
        duration: Optional[int] = None
    ) -> str:
        """Generates a SUMO route XML file for the specified density level and seed."""
        os.makedirs(self.output_dir, exist_ok=True)
        dur = duration if duration is not None else self.sim_duration
        route_file = os.path.join(self.output_dir, f"traffic_{traffic_level}_seed{seed}.rou.xml")

        # Check if already generated
        if os.path.isfile(route_file) and os.path.getsize(route_file) > 100:
            return route_file

        density_info = self.densities.get(traffic_level, {"period": 7.2})
        period = density_info.get("period", 7.2)

        python_exe = sys.executable
        cmd = [
            python_exe,
            self.random_trips_py,
            "-n", os.path.abspath(self.net_file),
            "-r", os.path.abspath(route_file),
            "-e", str(dur),
            "-p", str(period),
            "--seed", str(seed),
            "--fringe-factor", "5",
            "--validate",
            "--min-distance", "50",
            "--vehicle-class", "passenger"
        ]

        logger.info(f"Generating {traffic_level} traffic demand (seed={seed}, period={period}s)...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error(f"randomTrips failed: {res.stderr}")
            raise RuntimeError(f"Failed to generate traffic: {res.stderr}")

        logger.info(f"Generated route file: {route_file} ({os.path.getsize(route_file)} bytes)")
        return route_file

    def generate_all_traffic_regimes(self, seed: int = 42) -> Dict[str, str]:
        """Generates route files for all configured traffic regimes."""
        results = {}
        for level in ["low", "normal", "morning_peak", "evening_peak", "heavy"]:
            results[level] = self.generate_scenario(traffic_level=level, seed=seed)
        return results
