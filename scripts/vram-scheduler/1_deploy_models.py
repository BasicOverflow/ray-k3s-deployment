"""Deploy multiple models using the VRAM scheduler."""
import sys
import os
import ray

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

import ray_utils
import model_orchestrator
import vram_allocator

ModelOrchestrator = model_orchestrator.ModelOrchestrator
get_vram_allocator = vram_allocator.get_vram_allocator

SUPPRESS_LOGGING = False

# Model configuration
MODELS = {
    # "tinyllama": {
    #     "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     "vram_gb": 2.0,
    #     "replicas": 1 
    # },
    "qwen": {
        "name": "Qwen/Qwen2-0.5B-Instruct",
        "vram_gb": 1.5,  # Model weights ~0.92GB, need room for KV cache
        "replicas": 15  # Multiple replicas can share GPUs (fractional GPU allocation)
    },
}

def main():
    ray_utils.init_ray(suppress_logging=SUPPRESS_LOGGING)
    
    get_vram_allocator()
    orchestrator = ModelOrchestrator.remote()
    ray.get(orchestrator.apply.remote(MODELS))
    
    allocator = ray.get_actor("vram_allocator", namespace="system")
    state = ray.get(allocator.get_all_gpus.remote())
    
    print("\nVRAM State (Per-GPU):")
    for gpu_key, info in state.items():
        active_count = info.get("active_count", 0) if "active_count" in info else 0
        print(f"  {gpu_key}: {info['free']:.2f}GB free / {info['total']:.2f}GB total, {active_count} replicas loaded")

if __name__ == "__main__":
    main()

