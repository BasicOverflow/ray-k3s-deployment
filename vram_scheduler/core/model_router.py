"""Model Router - load balances across multiple GPU deployments."""
import ray
from ray import serve
from typing import Dict, List, Optional, Union
import time
import asyncio


@serve.deployment(
    ray_actor_options={"num_cpus": 0.1},
    autoscaling_config=None,
    num_replicas=1
)
class ModelRouter:
    """Router that load balances requests across multiple GPU deployments using weighted selection."""
    
    def __init__(self, model_id: str, gpu_deployment_names: List[str]):
        """Initialize router with list of GPU deployment names to route to.
        
        Args:
            model_id: Model identifier
            gpu_deployment_names: List of deployment names (one per GPU)
        """
        self.model_id = model_id
        self.gpu_deployment_names = gpu_deployment_names
        self._handles = None
        self._base_weights = None  # Base weights from max_num_seqs
        self._current_weights = None  # Track current_weight for each deployment
        
        # Performance and load tracking
        self._response_times = None  # Exponential moving average of response times
        self._active_requests = None  # Count of active requests per deployment
        self._max_num_seqs = None  # max_num_seqs per deployment
        self._lock = asyncio.Lock()  # Lock for thread-safe updates
    
    def _get_handles(self):
        """Lazy initialization of deployment handles."""
        if self._handles is None:
            self._handles = []
            for deployment_name in self.gpu_deployment_names:
                # Each GPU deployment is in its own application: {model_id}-{gpu_name}
                app_name = f"{self.model_id}-{deployment_name}"
                try:
                    handle = serve.get_deployment_handle(deployment_name, app_name=app_name)
                    self._handles.append(handle)
                except Exception as e:
                    print(f"Warning: Could not get handle for {deployment_name}: {e}")
        
        # Filter out None handles
        self._handles = [h for h in self._handles if h is not None]
        return self._handles
    
    async def _initialize_weights(self):
        """Initialize weights and performance tracking."""
        if self._base_weights is not None:
            return
        
        handles = self._get_handles()
        if not handles:
            raise RuntimeError(f"No available GPU deployments for model {self.model_id}")
        
        # Query max_num_seqs from each deployment
        base_weights = []
        max_num_seqs_list = []
        for handle in handles:
            try:
                max_num_seqs = await handle.get_max_num_seqs.remote()
                max_num_seqs = max(max_num_seqs, 1)  # Ensure at least 1
                base_weights.append(max_num_seqs)
                max_num_seqs_list.append(max_num_seqs)
            except Exception as e:
                print(f"Warning: Could not get max_num_seqs, using default weight: {e}")
                base_weights.append(32)
                max_num_seqs_list.append(32)
        
        self._base_weights = base_weights
        self._max_num_seqs = max_num_seqs_list
        
        # Initialize tracking
        num_deployments = len(handles)
        self._current_weights = [0] * num_deployments
        self._response_times = [0.1] * num_deployments  # Start with 100ms default
        self._active_requests = [0] * num_deployments
        
        total_weight = sum(base_weights)
        print(f"[Router] Initialized adaptive load balancing: {dict(zip(self.gpu_deployment_names, base_weights))}")
        print(f"[Router] Total base weight: {total_weight}")
    
    def _calculate_dynamic_weights(self):
        """Calculate dynamic weights based on performance and load."""
        if self._base_weights is None:
            return None
        
        # Calculate performance factors (inverse of normalized response time)
        # Faster GPUs get higher performance factor
        min_response_time = min(self._response_times) if self._response_times else 0.1
        max_response_time = max(self._response_times) if self._response_times else 1.0
        response_range = max(max_response_time - min_response_time, 0.001)  # Avoid division by zero
        
        performance_factors = []
        for i, response_time in enumerate(self._response_times):
            # Normalize: faster = higher factor
            # Factor = (max - current) / range, normalized to [0.5, 2.0]
            normalized = (max_response_time - response_time) / response_range
            factor = 0.5 + (normalized * 1.5)  # Scale to [0.5, 2.0]
            performance_factors.append(factor)
        
        # Calculate load factors (based on active requests vs capacity)
        load_factors = []
        for i, active in enumerate(self._active_requests):
            max_seqs = self._max_num_seqs[i] if self._max_num_seqs else 32
            # Load factor: 1.0 when empty, increases with load
            # Use 1 + (active / max_seqs) to penalize loaded deployments
            load_factor = 1.0 + (active / max(max_seqs, 1))
            load_factors.append(load_factor)
        
        # Dynamic weight = (base_weight * performance_factor) / load_factor
        # This biases toward fast, less loaded GPUs
        dynamic_weights = []
        for i, base_weight in enumerate(self._base_weights):
            dynamic_weight = (base_weight * performance_factors[i]) / load_factors[i]
            dynamic_weights.append(max(dynamic_weight, 0.1))  # Ensure minimum weight
        
        return dynamic_weights
    
    def _get_next_handle(self):
        """Get next handle using adaptive weighted round-robin."""
        if self._base_weights is None or self._current_weights is None:
            raise RuntimeError("Weights not initialized. Call _initialize_weights() first.")
        
        handles = self._get_handles()
        if not handles:
            raise RuntimeError(f"No available GPU deployments for model {self.model_id}")
        
        if len(handles) != len(self._base_weights):
            raise RuntimeError(f"Mismatch: {len(handles)} handles but {len(self._base_weights)} weights")
        
        # Calculate dynamic weights based on performance and load
        dynamic_weights = self._calculate_dynamic_weights()
        if dynamic_weights is None:
            dynamic_weights = self._base_weights  # Fallback to base weights
        
        # Weighted Round-Robin with dynamic weights
        total_weight = sum(dynamic_weights)
        max_current_weight = -1
        selected_index = 0
        
        for i, weight in enumerate(dynamic_weights):
            # Add weight to current_weight
            self._current_weights[i] += weight
            
            # Track the highest
            if self._current_weights[i] > max_current_weight:
                max_current_weight = self._current_weights[i]
                selected_index = i
        
        # Subtract total weight from selected deployment
        self._current_weights[selected_index] -= total_weight
        
        return handles[selected_index], selected_index
    
    async def __call__(self, request: Union[str, Dict]) -> Union[str, List]:
        """Route request to a GPU deployment using adaptive load balancing."""
        # Initialize weights on first call
        await self._initialize_weights()
        
        # Get next handle and track which deployment was selected
        handle, deployment_index = self._get_next_handle()
        
        # Track active request
        async with self._lock:
            self._active_requests[deployment_index] += 1
        
        start_time = time.time()
        
        try:
            # Forward request to the selected GPU deployment
            # Supports both single prompts and batch prompts
            if isinstance(request, dict):
                # Already a dict - forward as-is (handles both "prompt" and "prompts" keys)
                result = await handle.remote(request)
            else:
                # Single string prompt - wrap in dict
                result = await handle.remote({"prompt": request})
            
            # Update performance metrics
            response_time = time.time() - start_time
            async with self._lock:
                # Exponential moving average: new_avg = alpha * new + (1 - alpha) * old
                alpha = 0.1  # Smoothing factor (10% new, 90% old)
                self._response_times[deployment_index] = (
                    alpha * response_time + (1 - alpha) * self._response_times[deployment_index]
                )
            
            return result
        finally:
            # Decrement active request count
            async with self._lock:
                self._active_requests[deployment_index] = max(0, self._active_requests[deployment_index] - 1)
    
    async def get_max_num_seqs(self) -> int:
        """Get max_num_seqs from first available GPU deployment."""
        handles = self._get_handles()
        if not handles:
            return 32  # Default fallback
        
        try:
            return await handles[0].get_max_num_seqs.remote()
        except Exception:
            return 32  # Default fallback

