# Ray K3s Deployment

Infrastructure as Code for deploying a production-ready Ray cluster on k3s using KubeRay. Supports vLLM inference serving, REST API job submission, and dynamic VRAM-based GPU scheduling.

## Overview

This repository contains all the manifests, scripts, and documentation needed to deploy a Ray cluster on k3s with:

- **KubeRay Operator**: Manages RayCluster lifecycle
- **Dynamic GPU Pool**: Automatic GPU allocation per node (no per-node configuration needed)
- **VRAM-Based Scheduling**: Tasks request VRAM, not GPU counts - Ray handles GPU assignment
- **vLLM Inference**: Deploy models via Ray Serve with automatic replica placement
- **REST API**: Submit jobs remotely via REST API (no CLI required)
- **Heterogeneous GPU Support**: Treats all GPUs on a node as a VRAM pool

## Architecture

```
Remote Devices/Scripts
    ↓
MetalLB LoadBalancer (Ray Dashboard + REST API)
    ↓
Ray Head Node (1 pod)
    ├── Ray Dashboard (port 8265)
    └── Ray REST API (/api/jobs)
    ↓
Ray Worker Pods
    ├── CPU Workers (on CPU-only nodes)
    └── GPU Workers (on GPU nodes)
        ├── All GPUs allocated per pod (dynamic)
        ├── VRAM reported as custom resource
        └── vLLM replicas scheduled by VRAM availability
```

## Key Features

### VRAM-Aware Scheduling
- **Dynamic VRAM tracking**: Custom DaemonSet monitors VRAM on each GPU node every 0.5s
- **Global allocator actor**: Singleton actor maintains VRAM state across all nodes
- **Exact VRAM requirements**: Models declare exact VRAM needs, no overcommit
- **Automatic placement**: Ray Serve places replicas based on available VRAM

### vLLM Model Deployment
- Deploy models via Ray Serve with VRAM reservation
- **Multiple models per GPU**: Fractional GPU allocation enables multiple replicas to share a single GPU
- Declarative model configuration
- Automatic scaling and placement
- Zero OOM guarantees through hard reservations

### Cluster Testing
- Basic connectivity and resource tests
- VRAM allocator verification
- CPU/GPU stress testing

## Repository Structure

```
ray-k3s-deployment/
├── manifests/                          # Kubernetes manifests
│   ├── raycluster.yaml                 # Main RayCluster deployment
│   ├── ray-vram-monitor-daemonset.yaml # VRAM monitoring DaemonSet
│   ├── vram-scheduler-configmap.yaml   # VRAM scheduler scripts
│   └── helm/                           # KubeRay operator Helm config
├── ray_hive/                           # Ray Hive Python module
│   ├── __init__.py                     # Main API exports
│   ├── client.py                       # RayHive main client class
│   ├── inference.py                    # Standalone inference functions
│   ├── shutdown.py                     # Shutdown functionality
│   ├── openai_compat.py                # OpenAI API compatibility (TODO)
│   ├── langchain.py                    # LangChain compatibility (TODO)
│   ├── core/                           # Core components
│   │   ├── vram_allocator.py          # VRAM allocator actor
│   │   ├── vllm_model_actor.py       # vLLM model actor
│   │   ├── model_orchestrator.py     # Model orchestrator
│   │   └── model_router.py           # Load balancer router
│   └── utils/                          # Utilities
│       └── ray_utils.py                # Ray utilities
├── examples/                           # Example scripts
│   ├── 0_shutdown_models.py          # Shutdown deployments
│   ├── 1_deploy_models.py            # Deploy models
│   └── 2_test_inference.py           # Test inference
├── basic_ray_tests/                    # Cluster testing scripts
│   ├── 1_test_basic_connection.py    # Basic connectivity test
│   ├── 2_test_rest_api.py            # REST API test
│   ├── 3_test_vram_resource.py       # VRAM allocator test
│   └── 4_test_cpu_stress.py          # CPU stress test
├── setup.py                            # Package setup
├── pyproject.toml                     # Modern package config
└── requirements.txt                   # Dependencies
```

## Quick Start

### Deploy Ray Cluster

```bash
# Deploy KubeRay operator (if not already installed)
kubectl apply -f manifests/helm/kuberay-operator-values.yaml

# Deploy Ray cluster
kubectl apply -f manifests/raycluster.yaml

# Deploy VRAM monitor
kubectl apply -f manifests/vram-scheduler-configmap.yaml
kubectl apply -f manifests/ray-vram-monitor-daemonset.yaml
```

### Install Ray Hive Module

```bash
# Install from local source
pip install -e .

# Or install from GitLab (update URL with your project)
pip install ray-hive --extra-index-url https://gitlab.com/api/v4/projects/.../packages/pypi/simple
```

### Deploy Models

**Using the Ray Hive Module:**

```python
from ray_hive import RayHive

scheduler = RayHive()

# Deploy model with specific number of replicas
scheduler.deploy_model(
    model_id="qwen",
    model_name="Qwen/Qwen3-0.6B-GPTQ-Int8",
    vram_weights_gb=0.763,  # Model weights only (KV cache calculated separately)
    replicas=6,
    max_model_len=2048,
    enforce_eager=True,
    kv_cache_dtype="fp8"
)

# Deploy in test mode (single replica on GPU with most VRAM)
scheduler.deploy_model(
    model_id="qwen-test",
    model_name="Qwen/Qwen3-0.6B-GPTQ-Int8",
    vram_weights_gb=0.763,
    test_mode=True,  # Deploy only on GPU with most VRAM
    max_model_len=2048
)

# Display VRAM state
scheduler.display_vram_state()
```

**Using Example Scripts:**

```bash
# Deploy models
python examples/1_deploy_models.py

# Shutdown all models
python examples/0_shutdown_models.py
```

### Run Inference

**Using Standalone Inference Functions:**

```python
from ray_hive.inference import inference, a_inference, inference_batch

# Synchronous inference
result = inference("Hello!", model_id="my-model")

# Async inference
result = await a_inference("Hello!", model_id="my-model")

# Batch inference
results = inference_batch(
    ["Prompt 1", "Prompt 2", "Prompt 3"],
    model_id="my-model"
)

# Structured output
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

result = inference(
    "What is 2+2?",
    model_id="my-model",
    structured_output=Response
)
```

**Using Example Script:**

```bash
python examples/2_test_inference.py
```

## How It Works

### Ray Hive Architecture

1. **VRAM Allocator**: Global singleton actor (detached, HA-safe) that tracks VRAM state per GPU. Maintains pending and active reservations to prevent OOM.

2. **VRAM Monitoring**: DaemonSet runs on each GPU node, queries VRAM via `nvidia-smi` every 0.5s and updates the allocator with current free/total VRAM per GPU.

3. **Model Deployment Flow**:
   - `RayHive.deploy_model()` calls `ModelOrchestrator` to deploy models
   - Orchestrator queries allocator for available GPUs with sufficient VRAM
   - Creates one `VLLMModel` deployment per GPU (one replica per GPU)
   - Each deployment targets a specific GPU via `CUDA_VISIBLE_DEVICES`
   - `ModelRouter` load balances requests across all GPU deployments

4. **VRAM Reservation**: Each `VLLMModel` replica:
   - Reserves VRAM in the allocator before loading (moves from pending to active after init)
   - Uses fractional GPU allocation (`num_gpus: 0.01`) for Ray scheduling
   - Hard-limits VRAM usage via `torch.cuda.set_per_process_memory_fraction()` before loading
   - Calculates KV cache VRAM requirements from model architecture (auto-detected or provided)

5. **Multiple Models Per GPU**: Enabled through fractional GPU allocation and CUDA memory slicing. Each replica uses `gpu_memory_utilization` in vLLM and hard memory limits to prevent OOM.

### Components

- **`VRAMAllocator`** (`core/vram_allocator.py`): Global VRAM state actor (singleton, detached, HA-safe). Tracks free/available VRAM per GPU and manages reservations.
- **`ModelOrchestrator`** (`core/model_orchestrator.py`): Deploys models via Ray Serve. Queries allocator for available GPUs, creates one `VLLMModel` deployment per GPU, and sets up routing.
- **`VLLMModel`** (`core/vllm_model_actor.py`): vLLM deployment actor. Reserves VRAM before loading, targets specific GPU via `CUDA_VISIBLE_DEVICES`, calculates KV cache requirements, and hard-limits memory usage.
- **`ModelRouter`** (`core/model_router.py`): Load balances inference requests across all GPU deployments for a model.

## Configuration

- **Ray Address**: `ray://10.0.1.53:10001` (LoadBalancer IP)
- **Dashboard**: `http://10.0.1.53:8265`
- **Cluster**: 6 worker nodes (3 CPU + 3 GPU), 6 GPUs total
- **VRAM Tracking**: Per-node tracking via K8s node names

## Troubleshooting

When deploying multiple replicas that share a GPU, transient memory errors during initialization are expected. Ray Serve automatically retries failed deployments until models successfully load. If models consistently fail, verify VRAM requirements include a 70% buffer for overhead and that total VRAM doesn't exceed available GPU memory.

## Ray Hive Module API

### RayHive Class

Main client for distributed LLM serving.

**Methods:**

- `deploy_model(model_id, model_name, vram_weights_gb, replicas, ...)` - Deploy a model
  - `vram_weights_gb` - Model weights VRAM requirement in GB (required, KV cache calculated separately)
  - `replicas` - Number of replicas to deploy (optional, defaults to one per GPU)
  - `test_mode=True` - Deploy single replica on GPU with most VRAM (useful for testing)
  - `max_num_seqs` - Max concurrent sequences per instance (optional, auto-calculated if not provided)
  - `max_model_len=8192` - Maximum prompt length
  - `max_tokens=256` - Maximum tokens to generate
  - `hidden_dim`, `num_layers`, `dtype` - Architecture params (auto-detected if not provided)
  - `**vllm_kwargs` - Pass through to vLLM (e.g., `enforce_eager`, `kv_cache_dtype`, etc.)

- `shutdown(model_id=None)` - Shutdown models (None = all)
- `get_vram_state()` - Get VRAM state dict
- `display_vram_state()` - Display VRAM state

### Inference Functions

Standalone functions that work independently of the scheduler:

- `inference(prompt, model_id, structured_output=None, max_tokens=None, ...)` - Synchronous inference
- `a_inference(prompt, model_id, ...)` - Async inference
- `inference_batch(prompts, model_id, batch_size=None, ...)` - Batch inference (auto-batches based on max_num_seqs)
- `a_inference_batch(prompts, model_id, batch_size=None, ...)` - Async batch inference
- `streaming_batch(prompts, model_id, ...)` - Streaming (TODO)

All inference functions:
- Auto-discover deployments from Ray Serve status
- Support structured output (Pydantic classes)
- Support max_tokens and input truncation
- Accept all vLLM sampling parameters

## Future Enhancements

- **LangChain Compatibility**: LangChain LLM wrapper (TODO)
- **Vision/Audio Support**: Support for vision and audio models (TODO)
- **OpenAI API Compatibility**: OpenAI-compatible endpoints (TODO)
- **Streaming**: Full streaming support (TODO)

## Related Repositories

- [rayify](https://github.com/BasicOverflow/rayify) - Ray script conversion tool

