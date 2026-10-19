"""Audit script for the Offline-to-Online MARL model and checkpoint."""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.marl_six import OfflineOnlineMARLModel

def audit_model():
    ckpt_path = os.path.join("models", "marl_six", "offline_online", "seed_42", "best_model.pt")
    print(f"Checking checkpoint path: {ckpt_path}", flush=True)
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: Checkpoint file does not exist!", flush=True)
        return False
    
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint keys: {list(ckpt_data.keys())}", flush=True)
    
    model = OfflineOnlineMARLModel(n_agents=4, obs_dim=34, act_dim=4, device="cpu")
    model.load(ckpt_path)
    
    # Calculate total parameters across all modules
    total_params = 0
    param_details = {}
    for name, module in [
        ("encoder", getattr(model, "encoder", None)),
        ("critic", getattr(model, "critic", None)),
        ("actor", getattr(model, "actor", None)),
    ]:
        if module is not None and hasattr(module, "parameters"):
            count = sum(p.numel() for p in module.parameters())
            param_details[name] = count
            total_params += count
            
    # Or inspect submodules in model
    named_children_params = {}
    for cname, child in model.__dict__.items():
        if isinstance(child, torch.nn.Module):
            c_count = sum(p.numel() for p in child.parameters())
            named_children_params[cname] = c_count

    print(f"Named module counts: {named_children_params}", flush=True)
    all_net_params = sum(p.numel() for p in model.actor.parameters()) if hasattr(model, 'actor') else 0
    print(f"Actor parameters: {all_net_params}", flush=True)
    
    # Also check backend/app/ml/model_manager.py to see how it calculates parameters
    from backend.app.ml.model_manager import model_manager
    manager_meta = model_manager.get_metadata()
    print(f"ModelManager metadata: {manager_meta}", flush=True)
    return True

if __name__ == "__main__":
    audit_model()
