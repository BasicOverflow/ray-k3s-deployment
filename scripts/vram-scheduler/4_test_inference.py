"""Test inference on deployed models and load balancing."""
import os
import sys
import time

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

from ray import serve
import ray_utils

SUPPRESS_LOGGING = True

PROMPT = "Write be a nice long poem about beer."

ray_utils.init_ray(suppress_logging=SUPPRESS_LOGGING)

# Get deployed models
serve_status = serve.status()
apps = serve_status.applications

print(f"Found {len(apps)} model(s): {list[str](apps.keys())}\n")

# Test each model
for model_id in apps.keys():
    print(f"Testing {model_id}...", end=" ", flush=True)
    try:
        handle = serve.get_deployment_handle(model_id, app_name=model_id)
        # Warmup request to reduce cold start latency
        _ = handle.generate.remote("warmup").result()
        
        start = time.time()
        response = handle.generate.remote(PROMPT)
        result = response.result()
        elapsed = time.time() - start
        
        # Extract text from list if needed
        if isinstance(result, list) and len(result) > 0:
            text = result[0] if isinstance(result[0], str) else result[0]
        else:
            text = result
        print(f"({elapsed:.2f}s) Response: {text[:100]}...\n")
    except Exception as e:
        print(f"Error: {e}\n")



# Load balancing test
if apps:
    test_model = list(apps.keys())[0]
    print(f"Load balancing test: sending 50 parallel requests to {test_model}")
    
    try:
        handle = serve.get_deployment_handle(test_model, app_name=test_model)
        start = time.time()
        
        # Send parallel requests
        responses = [handle.generate.remote(f"{PROMPT} Request {i}") for i in range(50)]
        results = [r.result() for r in responses]
        
        elapsed = time.time() - start
        print(f"Completed 50 requests in {elapsed:.2f}s ({50/elapsed:.1f} req/s)")
        print(f"Received {len(results)} responses")
    except Exception as e:
        print(f"Error in load balancing test: {e}")

