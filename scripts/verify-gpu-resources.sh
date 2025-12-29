#!/bin/bash

# Verify GPU resources are exposed in Kubernetes
# This script checks if nvidia.com/gpu resources are available on GPU nodes
# Exact commands used during deployment

set -e

echo "=== Checking NVIDIA Device Plugin Status ==="
echo ""

# Check if device plugin pods are running (namespace: nvidia-device-plugin)
echo "1. Checking NVIDIA device plugin pods..."
kubectl get pods -n nvidia-device-plugin

echo ""
echo "=== Checking GPU Resources on Nodes ==="
echo ""

# Check GPU capacity on each GPU node (exact commands we used)
echo "2. Checking nvidia.com/gpu capacity on GPU nodes..."
echo ""
echo "ergos-02-nv:"
kubectl get node ergos-02-nv -o jsonpath='{.status.capacity.nvidia\.com/gpu}{"\n"}'

echo "ergos-04-nv:"
kubectl get node ergos-04-nv -o jsonpath='{.status.capacity.nvidia\.com/gpu}{"\n"}'

echo "ergos-06-nv:"
kubectl get node ergos-06-nv -o jsonpath='{.status.capacity.nvidia\.com/gpu}{"\n"}'

echo ""
echo "=== All GPU Nodes Summary ==="
echo ""

# Check all nodes at once (without jq dependency)
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUS:.status.capacity.nvidia\.com/gpu | grep -v "<none>"

echo ""
echo "=== Detailed Node Information ==="
echo ""

# Show detailed info for each GPU node
echo "3. Node: ergos-02-nv"
kubectl describe node ergos-02-nv | grep -A 2 "nvidia.com/gpu"
echo ""

echo "4. Node: ergos-04-nv"
kubectl describe node ergos-04-nv | grep -A 2 "nvidia.com/gpu"
echo ""

echo "5. Node: ergos-06-nv"
kubectl describe node ergos-06-nv | grep -A 2 "nvidia.com/gpu"
echo ""

echo "=== Verification Complete ==="
echo ""
echo "Expected GPU counts:"
echo "  - ergos-02-nv: 3 GPUs"
echo "  - ergos-04-nv: 2 GPUs"
echo "  - ergos-06-nv: 1 GPU"
echo ""
echo "If GPUs are visible, you can proceed to Step 2: KubeRay Operator Installation"
