#!/bin/bash

# Check if NVIDIA Container Toolkit is installed and configured on GPU nodes
# This must be done BEFORE deploying the device plugin
# Exact commands used during deployment

echo "=== Checking NVIDIA Container Toolkit on GPU Nodes ==="
echo ""

GPU_NODES=("ergos-02-nv" "ergos-04-nv" "ergos-06-nv")
GPU_IPS=("10.0.1.4" "10.0.1.6" "10.0.1.10")

for i in "${!GPU_NODES[@]}"; do
    node="${GPU_NODES[$i]}"
    ip="${GPU_IPS[$i]}"
    
    echo "Checking $node ($ip)..."
    
    # Check if nvidia-container-runtime exists (exact command we used)
    echo -n "  - nvidia-container-runtime: "
    if ssh peter@$ip "which nvidia-container-runtime" 2>/dev/null | grep -q nvidia-container-runtime; then
        echo "✓ Found"
    else
        echo "✗ NOT FOUND - Container Toolkit not installed"
    fi
    
    # Check containerd config for NVIDIA runtime (exact command we used)
    echo -n "  - containerd NVIDIA runtime config: "
    if ssh peter@$ip "sudo grep nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml" 2>/dev/null | grep -q nvidia; then
        echo "✓ Configured"
        echo "    Config:"
        ssh peter@$ip "sudo grep -A 2 nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml" 2>/dev/null | sed 's/^/      /'
    else
        echo "✗ NOT CONFIGURED - Need to configure containerd"
    fi
    
    echo ""
done

echo "=== Prerequisites Check Complete ==="
echo ""
echo "If Container Toolkit is missing, install it on GPU nodes:"
echo ""
echo "On each GPU node, run:"
echo "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
echo "  curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -"
echo "  curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | \\"
echo "    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
echo "  sudo apt-get update"
echo "  sudo apt-get install -y nvidia-container-toolkit"
echo "  sudo nvidia-ctk runtime configure --runtime=containerd"
echo "  sudo systemctl restart k3s"
echo ""
echo "Then verify with:"
echo "  which nvidia-container-runtime"
echo "  sudo grep nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml"
