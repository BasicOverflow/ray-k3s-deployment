"""Test inference features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import time
from pydantic import BaseModel
from ray_hive.inference import inference, a_inference, inference_batch, a_inference_batch

MODEL_ID = "qwen-custom"
prompt = "Write a short poem about beer"
amount = 10_000

# Warmup
print("Warming up...")
_ = inference(prompt, model_id=MODEL_ID, max_tokens=10, temperature=0.0)

# Synchronous inference
print("\n=== Synchronous Inference ===")
start = time.time()
result = inference(prompt, model_id=MODEL_ID, max_tokens=100, temperature=0.7)
elapsed = time.time() - start
print(f"Time: {elapsed:.3f}s")
print(f"Sample: {result}")

# Async inference
print("\n=== Async Inference ===")
async def test_async():
    start = time.time()
    result = await a_inference(prompt, model_id=MODEL_ID, max_tokens=100, temperature=0.8)
    elapsed = time.time() - start
    print(f"Time: {elapsed:.3f}s")
    print(f"Sample: {result}")
    return result
asyncio.run(test_async())

# Batch inference
print("\n=== Batch Inference ===")
prompts = [f"{prompt} {i}" for i in range(amount)]
start = time.time()
results = inference_batch(prompts, model_id=MODEL_ID, max_tokens=100, temperature=0.0)
elapsed = time.time() - start
print(f"Processed {len(results)} prompts in {elapsed:.3f}s ({len(results)/elapsed:.2f} req/s)")
print(f"Sample: {results[0]}")

# Async batch inference
print("\n=== Async Batch Inference ===")
async def test_async_batch():
    prompts = [prompt] * amount
    start = time.time()
    results = await a_inference_batch(prompts, model_id=MODEL_ID, max_tokens=100, temperature=0.7, stop=["\n\n"])
    elapsed = time.time() - start
    print(f"Processed {len(results)} prompts in {elapsed:.3f}s ({len(results)/elapsed:.2f} req/s)")
    print(f"Sample: {results[0]}")
    return results
asyncio.run(test_async_batch())

# Structured output
print("\n=== Structured Output ===")
class MathResponse(BaseModel):
    answer: str
    explanation: str

start = time.time()
structured = inference("What is 2+2? Explain.", model_id=MODEL_ID, structured_output=MathResponse, max_tokens=150)
elapsed = time.time() - start
print(f"Time: {elapsed:.3f}s")
print(f"\n\n{structured}")