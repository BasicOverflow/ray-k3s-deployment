"""Measure actual VRAM usage of a model by deploying a single replica."""
import os
import sys
import ray
import time

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

import ray_utils
import vram_allocator
from ray import serve
from vllm_model_actor import VLLMModel

get_vram_allocator = vram_allocator.get_vram_allocator

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
ESTIMATED_VRAM_GB = 3.5

def main():
    ray_utils.init_ray(suppress_logging=True)
    allocator = get_vram_allocator()
    time.sleep(1)
    
    # Find GPU with enough free VRAM
    state = ray.get(allocator.get_all_gpus.remote())
    target_gpu = None
    initial_used = 0
    
    for gpu_key, info in state.items():
        if len(gpu_key) > 50 or gpu_key.startswith('c'):
            continue
        total = info.get("total", 0)
        free = info.get("free", 0)
        if free >= ESTIMATED_VRAM_GB:
            target_gpu = gpu_key
            initial_used = total - free
            break
    
    if not target_gpu:
        print(f"No GPU has {ESTIMATED_VRAM_GB}GB free VRAM")
        return
    
    print(f"Deploying {MODEL_NAME} on {target_gpu}...")
    
    # Deploy
    node_name, gpu_id_str = target_gpu.split(":")
    gpu_id = gpu_id_str.replace("gpu", "")
    resource_name = f"{node_name}_gpu{gpu_id}"
    deployment_name = f"measure-{node_name}-gpu{gpu_id}"
    app_name = f"measure-{node_name}-gpu{gpu_id}"
    
    serve.run(
        VLLMModel.options(
            name=deployment_name,
            ray_actor_options={
                "num_gpus": 0.01,
                "memory": 2 * 1024 * 1024 * 1024,
                "resources": {resource_name: 1}
            },
            autoscaling_config={"min_replicas": 1, "max_replicas": 1}
        ).bind(model_id="measure", model_name=MODEL_NAME, required_vram_gb=ESTIMATED_VRAM_GB),
        name=app_name,
        route_prefix=f"/{deployment_name}"
    )
    
    print("Deployment started, waiting for VRAM to stabilize...")
    time.sleep(30)
    
    # Measure VRAM after deployment
    final_state = ray.get(allocator.get_all_gpus.remote())
    for gpu_key, info in final_state.items():
        if gpu_key == target_gpu:
            total = info.get("total", 0)
            free = info.get("free", 0)
            final_used = total - free
            actual_vram = final_used - initial_used
            break
    
    print(f"\nResults:")
    print(f"  Actual VRAM used: {actual_vram:.2f}GB")
    print(f"  Estimated: {ESTIMATED_VRAM_GB:.2f}GB")
    print(f"  Use ACTUAL_VRAM_GB = {actual_vram:.2f}")
    
    # Cleanup - same approach as 0_shutdown_models.py
    print(f"\nCleaning up...")
    apps = serve.status().applications
    if app_name in apps:
        serve.delete(name=app_name)
        cleared = ray.get(allocator.clear_reservations_by_prefix.remote("measure-"))
        print(f"✅ Shut down {app_name}")
        if cleared > 0:
            print(f"✅ Cleared {cleared} VRAM reservations")
    else:
        print(f"⚠️  Application '{app_name}' not found")
        cleared = ray.get(allocator.clear_reservations_by_prefix.remote("measure-"))
        if cleared > 0:
            print(f"✅ Cleared {cleared} VRAM reservations")

if __name__ == "__main__":
    main()

