"""
Global VRAM Allocator Actor - singleton, HA-safe, persistent.

Per-GPU tracking: Tracks VRAM state per GPU device (node_id:gpu_id format).
"""
import ray
from typing import Dict, Optional

@ray.remote(num_cpus=0)
class VRAMAllocator:
    """Global VRAM allocator - singleton, HA-safe, persistent."""
    
    def __init__(self):
        # gpu_key (node_id:gpu_id) -> {total_gb, free_gb (from nvidia-smi), pending: {replica_id: gb}, active: {replica_id: gb}}
        self.gpus: Dict[str, Dict] = {}
        # Per-GPU initialization locks - tracks which replica is currently initializing on each GPU
        # gpu_key -> replica_id (or None if no one is initializing)
        self.gpu_locks: Dict[str, Optional[str]] = {}
    
    def update_gpu(self, node_id: str, gpu_id: int, free_gb: float, total_gb: float):
        """Update VRAM state for a GPU (called by DaemonSet).
        
        free_gb from nvidia-smi is the actual free memory on the GPU.
        """
        gpu_key = f"{node_id}:gpu{gpu_id}"
        if gpu_key not in self.gpus:
            self.gpus[gpu_key] = {
                "total": total_gb, 
                "free": free_gb,  # Actual free from nvidia-smi
                "pending": {},  # Reservations not yet initialized
                "active": {}  # Successfully initialized replicas
            }
        else:
            # Update free/total from nvidia-smi, preserve reservations
            self.gpus[gpu_key]["total"] = total_gb
            self.gpus[gpu_key]["free"] = free_gb  # Ground truth from nvidia-smi
    
    def get_available_vram(self, gpu_key: str) -> float:
        """Get truly available VRAM: nvidia-smi free minus pending reservations.
        
        Note: nvidia-smi 'free' already accounts for active replicas using VRAM.
        We subtract pending reservations because they will soon start using VRAM.
        """
        if gpu_key not in self.gpus:
            return 0.0
        gpu = self.gpus[gpu_key]
        pending_total = sum(gpu["pending"].values())
        # Available = actual free (from nvidia-smi, already accounts for active) - pending reservations
        # This ensures we don't over-allocate while replicas are initializing
        available = gpu["free"] - pending_total
        return max(0.0, available)
    
    def reserve(self, replica_id: str, gpu_key: str, required_gb: float) -> bool:
        """Reserve VRAM for a replica. Returns True if successful.
        
        This creates a PENDING reservation that's immediately subtracted from available.
        The reservation becomes ACTIVE when the replica successfully initializes.
        """
        if gpu_key not in self.gpus:
            return False
        
        gpu = self.gpus[gpu_key]
        available = self.get_available_vram(gpu_key)
        
        if available < required_gb:
            return False
        
        # Create pending reservation (subtracts from available immediately)
        gpu["pending"][replica_id] = required_gb
        return True
    
    def mark_initialized(self, replica_id: str, gpu_key: str):
        """Mark a replica as successfully initialized.
        
        Moves reservation from pending to active.
        """
        if gpu_key not in self.gpus:
            return
        
        gpu = self.gpus[gpu_key]
        if replica_id in gpu["pending"]:
            required_gb = gpu["pending"].pop(replica_id)
            gpu["active"][replica_id] = required_gb
    
    def release(self, replica_id: str, gpu_key: str):
        """Release VRAM reservation for a replica.
        
        Removes from both pending and active.
        """
        if gpu_key not in self.gpus:
            return
        
        gpu = self.gpus[gpu_key]
        if replica_id in gpu["pending"]:
            gpu["pending"].pop(replica_id)
        if replica_id in gpu["active"]:
            gpu["active"].pop(replica_id)
    
    def get_gpu_vram(self, gpu_key: str) -> Optional[Dict]:
        """Get VRAM info for a specific GPU, including available (free - pending)."""
        if gpu_key not in self.gpus:
            return None
        
        gpu = self.gpus[gpu_key]
        available = self.get_available_vram(gpu_key)
        return {
            "total": gpu["total"],
            "free": gpu["free"],  # Actual free from nvidia-smi
            "available": available,  # Free minus pending reservations
            "pending": sum(gpu["pending"].values()),
            "active": sum(gpu["active"].values()),
            "pending_count": len(gpu["pending"]),
            "active_count": len(gpu["active"])
        }
    
    def find_gpu_with_vram(self, required_gb: float, node_id: Optional[str] = None) -> Optional[str]:
        """Find a GPU with enough available VRAM (free - pending).
        
        If node_id is provided, only searches GPUs on that node.
        Returns the GPU key with the most available VRAM that has at least required_gb available.
        """
        candidates = []
        for gpu_key, gpu in self.gpus.items():
            # Skip old Ray node IDs
            if len(gpu_key) > 50 or gpu_key.startswith('c'):
                continue
            
            # Filter by node if specified
            if node_id and not gpu_key.startswith(node_id + ":"):
                continue
            
            available = self.get_available_vram(gpu_key)
            if available >= required_gb:
                candidates.append((gpu_key, available))
        
        if not candidates:
            return None
        
        # Return GPU with most available VRAM (least-loaded selection)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def get_all_gpus(self) -> Dict:
        """Get VRAM state for all GPUs."""
        result = {}
        for gpu_key, gpu in self.gpus.items():
            available = self.get_available_vram(gpu_key)
            result[gpu_key] = {
                "total": gpu["total"],
                "free": gpu["free"],
                "available": available,
                "pending": sum(gpu["pending"].values()),
                "active": sum(gpu["active"].values()),
                "pending_count": len(gpu["pending"]),
                "active_count": len(gpu["active"])
            }
        return result
    
    def clear_all_reservations(self) -> int:
        """Clear all reservations. Returns the number cleared."""
        total_cleared = 0
        for gpu_key in self.gpus:
            gpu = self.gpus[gpu_key]
            total_cleared += len(gpu["pending"]) + len(gpu["active"])
            gpu["pending"] = {}
            gpu["active"] = {}
        return total_cleared
    
    def clear_reservations_by_prefix(self, prefix: str) -> int:
        """Clear all reservations that start with the given prefix."""
        cleared = 0
        for gpu_key in self.gpus:
            gpu = self.gpus[gpu_key]
            # Clear from pending
            to_remove = [rid for rid in gpu["pending"].keys() if rid.startswith(prefix)]
            for rid in to_remove:
                gpu["pending"].pop(rid)
                cleared += 1
            # Clear from active
            to_remove = [rid for rid in gpu["active"].keys() if rid.startswith(prefix)]
            for rid in to_remove:
                gpu["active"].pop(rid)
                cleared += 1
        return cleared
    
    def acquire_gpu_lock(self, gpu_key: str, replica_id: str) -> bool:
        """Acquire initialization lock for a GPU. Returns True if acquired."""
        if gpu_key not in self.gpu_locks:
            self.gpu_locks[gpu_key] = replica_id
            return True
        
        if self.gpu_locks[gpu_key] is not None:
            return False
        
        self.gpu_locks[gpu_key] = replica_id
        return True
    
    def release_gpu_lock(self, gpu_key: str, replica_id: str):
        """Release initialization lock for a GPU."""
        if gpu_key in self.gpu_locks and self.gpu_locks[gpu_key] == replica_id:
            self.gpu_locks[gpu_key] = None
    
    def set_replica_gpu_assignment(self, replica_id: str, gpu_key: str):
        """Set the expected GPU assignment for a replica."""
        if not hasattr(self, 'replica_assignments'):
            self.replica_assignments: Dict[str, str] = {}
        self.replica_assignments[replica_id] = gpu_key
    
    def get_replica_gpu_assignment(self, replica_id: str) -> Optional[str]:
        """Get the expected GPU assignment for a replica."""
        if not hasattr(self, 'replica_assignments'):
            return None
        return self.replica_assignments.get(replica_id)


def get_vram_allocator():
    """Get or create the global VRAM allocator actor."""
    try:
        return ray.get_actor("vram_allocator", namespace="system")
    except ValueError:
        return VRAMAllocator.options(
            name="vram_allocator",
            namespace="system",
            lifetime="detached"
        ).remote()
