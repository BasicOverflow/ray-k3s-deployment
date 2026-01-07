"""Test inference through the unified router endpoint with 100 concurrent requests."""
import os
import sys
import ray
from ray import serve
import time
import asyncio

vram_scheduler_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, vram_scheduler_dir)

import ray_utils

SUPPRESS_LOGGING = False
MODEL_ID = "max-llm"
NUM_REQUESTS = 100

def main():
    ray_utils.init_ray(suppress_logging=SUPPRESS_LOGGING)
    
    print(f"\n{'='*80}")
    print(f"Testing {MODEL_ID} with {NUM_REQUESTS} concurrent inference requests")
    print(f"{'='*80}\n")
    
    try:
        # Check status to find the router deployment name
        status = serve.status()
        router_deployment_name = None
        
        if MODEL_ID in status.applications:
            app_info = status.applications[MODEL_ID]
            if hasattr(app_info, 'deployments'):
                # Find the Router deployment (it's the only one in the app)
                for dep_name in app_info.deployments.keys():
                    router_deployment_name = dep_name
                    break
        
        if not router_deployment_name:
            print(f"Available applications: {list(status.applications.keys())}")
            if MODEL_ID in status.applications:
                app_info = status.applications[MODEL_ID]
                print(f"Application {MODEL_ID} deployments: {list(app_info.deployments.keys()) if hasattr(app_info, 'deployments') else 'N/A'}")
            raise RuntimeError(f"Could not find router deployment in application {MODEL_ID}")
        
        # Get handle to the unified router deployment
        handle = serve.get_deployment_handle(router_deployment_name, app_name=MODEL_ID)
        print(f"Using deployment handle: {router_deployment_name} in app {MODEL_ID}")
        
        # Test prompt
        test_prompt = "Write a long poem about beer"
        
        print(f"Prompt: {test_prompt}")
        print(f"\nSending {NUM_REQUESTS} concurrent requests...")
        
        # Create request payload
        # max_tokens can be up to 8192 (max_model_len), but we use 8000 to account for prompt tokens
        request = {
            "prompt": test_prompt,
            "max_tokens": 8000,
            "temperature": 0.7
        }
        
        # Send all requests concurrently
        # Router uses async __call__ method, handle.remote() returns DeploymentResponse
        # We can await them or use ray.get() on the responses
        async def run_requests():
            responses = [handle.remote(request) for _ in range(NUM_REQUESTS)]
            # DeploymentResponse objects can be awaited directly
            return [await resp for resp in responses]
        
        print(f"All {NUM_REQUESTS} requests submitted, waiting for responses...")
        start_time = time.time()
        results = asyncio.run(run_requests())
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"Results Summary")
        print(f"{'='*80}")
        print(f"Total requests: {NUM_REQUESTS}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Average time per request: {elapsed/NUM_REQUESTS:.2f}s")
        print(f"Throughput: {NUM_REQUESTS/elapsed:.2f} requests/second")
        
        # Count successes/failures
        successes = 0
        failures = 0
        for result in results:
            if isinstance(result, dict) and "error" in result:
                failures += 1
            else:
                successes += 1
        
        print(f"\nSuccesses: {successes}")
        print(f"Failures: {failures}")
        
        # Write all results to file
        output_file = f"inference_results_{int(time.time())}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"{'='*80}\n")
            f.write(f"Inference Test Results\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Model: {MODEL_ID}\n")
            f.write(f"Prompt: {test_prompt}\n")
            f.write(f"Total requests: {NUM_REQUESTS}\n")
            f.write(f"Total time: {elapsed:.2f}s\n")
            f.write(f"Average time per request: {elapsed/NUM_REQUESTS:.2f}s\n")
            f.write(f"Throughput: {NUM_REQUESTS/elapsed:.2f} requests/second\n")
            f.write(f"Successes: {successes}\n")
            f.write(f"Failures: {failures}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"All Responses\n")
            f.write(f"{'='*80}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"Response #{i}\n")
                f.write(f"{'='*80}\n")
                if isinstance(result, dict):
                    if "error" in result:
                        f.write(f"❌ Error: {result['error']}\n")
                    elif "text" in result:
                        f.write(f"{result['text']}\n")
                    else:
                        f.write(f"{result}\n")
                else:
                    f.write(f"{result}\n")
        
        print(f"\n✅ All results written to: {output_file}")
        
        # Show first response as sample
        print(f"\n{'='*80}")
        print("Sample Response (first request):")
        print(f"{'='*80}")
        result = results[0]
        if isinstance(result, dict):
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            elif "text" in result:
                print(result["text"])
            else:
                print(result)
        else:
            print(result)
            
    except Exception as e:
        print(f"❌ Failed to test {MODEL_ID}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("Inference testing complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

