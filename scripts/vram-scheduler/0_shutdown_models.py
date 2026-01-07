"""Shutdown Ray Serve deployments."""
import os
import sys
import argparse
import time
import ray
from ray import serve

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

import ray_utils
import vram_allocator

SUPPRESS_LOGGING = False

def kill_vram_allocator():
    """Kill the VRAM allocator actor so it gets recreated fresh."""
    try:
        allocator = ray.get_actor("vram_allocator", namespace="system")
        ray.kill(allocator)
        time.sleep(0.5)  # Give it time to fully terminate
        print("✅ Killed VRAM allocator actor (will be recreated fresh)")
    except ValueError:
        # Actor doesn't exist, nothing to kill
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=str, help="Specific app name to shutdown (default: all)")
    args = parser.parse_args()
    
    ray_utils.init_ray(suppress_logging=SUPPRESS_LOGGING)
    
    # Kill VRAM allocator first so it gets recreated fresh
    kill_vram_allocator()
    
    apps = serve.status().applications
    if not apps:
        print("No Ray Serve applications to shutdown.")
        # Get fresh allocator and clear any stale reservations
        allocator = vram_allocator.get_vram_allocator()
        cleared = ray.get(allocator.clear_all_reservations.remote())
        if cleared > 0:
            print(f"✅ Cleared {cleared} stale VRAM reservations")
        else:
            print("✅ No stale VRAM reservations to clear")
        return
    
    # Get fresh allocator
    allocator = vram_allocator.get_vram_allocator()
    
    if args.app:
        if args.app in apps:
            serve.delete(name=args.app)
            # Clear reservations for this app
            cleared = ray.get(allocator.clear_reservations_by_prefix.remote(f"{args.app}-"))
            print(f"✅ Shut down {args.app}")
            if cleared > 0:
                print(f"✅ Cleared {cleared} stale VRAM reservations for {args.app}")
        else:
            print(f"❌ Application '{args.app}' not found. Available: {list(apps.keys())}")
    else:
        print(f"Shutting down {len(apps)} application(s): {list(apps.keys())}")
        serve.shutdown()
        # Clear all reservations
        cleared = ray.get(allocator.clear_all_reservations.remote())
        print("✅ All applications shut down")
        if cleared > 0:
            print(f"✅ Cleared {cleared} stale VRAM reservations")

if __name__ == "__main__":
    main()

