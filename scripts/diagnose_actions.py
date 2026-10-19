import json
import math
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from src.network.osm_loader import OSMLoader
from src.environment.evacuation_env import EvacuationEnv
from src.baselines.dijkstra import DijkstraBaseline
from src.baselines.astar import AStarBaseline

def test_baselines():
    graph = OSMLoader.load_graph_json()
    scenarios = json.load(open("data/generated/test_scenarios.json"))[:20]

    dijkstra = DijkstraBaseline()
    astar = AStarBaseline()

    for name, agent in [("Dijkstra", dijkstra), ("A*", astar)]:
        successes = []
        returns = []
        steps_list = []

        for scen in scenarios:
            env = EvacuationEnv(scenarios=[scen], use_traci=False, curriculum_stage=1)
            obs, info = env.reset(options={"scenario": scen})
            done = False
            ep_ret = 0.0
            ep_steps = 0

            while not done:
                action = agent.select_action(env)
                obs, r, term, trunc, step_info = env.step(action)
                ep_ret += r
                ep_steps += 1
                done = term or trunc

            succ = step_info.get("success", False)
            successes.append(succ)
            returns.append(ep_ret)
            steps_list.append(ep_steps)

        print(f"=== {name} Baseline on 20 test scenarios ===")
        print(f"Success Rate: {np.mean(successes)*100:.1f}%")
        print(f"Mean Return: {np.mean(returns):.2f}")
        print(f"Mean Steps: {np.mean(steps_list):.1f}")
        assert np.mean(successes) == 1.0, f"{name} did not achieve 100% success!"

    print("\nALL BASELINES VERIFIED WITH 100% SUCCESS!")

if __name__ == "__main__":
    test_baselines()
