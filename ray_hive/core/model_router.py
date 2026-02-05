"""Model Router - predictive, performance-aware load balancing with batch splitting."""
from ray import serve
from typing import Dict, List, Union, Tuple, Optional
import asyncio
import time
from dataclasses import dataclass


@dataclass
class PerformanceProfile:
    """Performance profile for a GPU deployment."""
    sm_count: Optional[int]
    total_vram_gb: float
    max_num_seqs: int
    estimated_tokens_per_sec: float
    last_updated: float


@serve.deployment(
    ray_actor_options={"num_cpus": 0.1},
    autoscaling_config=None,
    num_replicas=1
)
class ModelRouter:
    """Router with predictive latency scoring, batch splitting, and performance-aware routing."""
    
    def __init__(self, model_id: str, gpu_deployment_names: List[str]):
        self.model_id = model_id
        self.gpu_deployment_names = gpu_deployment_names
        self._handles = None
        self._capacity_cache = {}
        self._performance_profiles: Dict[str, PerformanceProfile] = {}
        self._latency_history: Dict[str, List[float]] = {}
        self._tokenizer_deployment = None
        self._max_vram_gb = None
        self._cache_ttl = 0.1
        self._profile_ttl = 5.0
        self._last_update = {}
        self._last_profile_update = {}
        self._lock = asyncio.Lock()
        self._batch_split_threshold = 100
    
    def _get_handles(self):
        if self._handles is None:
            self._handles = {}
            for deployment_name in self.gpu_deployment_names:
                handle = serve.get_deployment_handle(deployment_name, app_name=deployment_name)
                self._handles[deployment_name] = handle
        return self._handles
    
    async def _get_tokenizer_deployment(self):
        """Get a deployment handle that has tokenizer available."""
        if hasattr(self, '_tokenizer_deployment') and self._tokenizer_deployment is not None:
            return self._tokenizer_deployment
        
        handles = self._get_handles()
        for deployment_name in self.gpu_deployment_names:
            self._tokenizer_deployment = deployment_name
            return deployment_name
        
        raise RuntimeError("No deployments available for token counting")
    
    async def _get_capacity_info(self, deployment_name: str) -> Dict:
        """Get capacity info for a deployment (cached)."""
        current_time = time.time()
        
        if (deployment_name not in self._capacity_cache or 
            deployment_name not in self._last_update or
            current_time - self._last_update[deployment_name] > self._cache_ttl):
            handles = self._get_handles()
            capacity_info = await handles[deployment_name].get_capacity_info.remote()
            self._capacity_cache[deployment_name] = capacity_info
            self._last_update[deployment_name] = current_time
            
            if capacity_info.get("total_vram_gb") is not None:
                if self._max_vram_gb is None or capacity_info["total_vram_gb"] > self._max_vram_gb:
                    self._max_vram_gb = capacity_info["total_vram_gb"]
                    self._performance_profiles.clear()
        
        return self._capacity_cache[deployment_name]
    
    async def _calculate_performance_profile(self, deployment_name: str) -> PerformanceProfile:
        """Calculate performance profile for a deployment."""
        current_time = time.time()
        
        if (deployment_name in self._performance_profiles and
            deployment_name in self._last_profile_update and
            current_time - self._last_profile_update[deployment_name] < self._profile_ttl):
            return self._performance_profiles[deployment_name]
        
        capacity_info = await self._get_capacity_info(deployment_name)
        sm_count = capacity_info.get("sm_count")
        total_vram_gb = capacity_info.get("total_vram_gb")
        max_num_seqs = capacity_info.get("max_num_seqs")
        
        if sm_count is None:
            raise ValueError(f"SM count not available for deployment {deployment_name}")
        if total_vram_gb is None or total_vram_gb <= 0:
            raise ValueError(f"Total VRAM not available for deployment {deployment_name}")
        if max_num_seqs is None:
            raise ValueError(f"max_num_seqs not available for deployment {deployment_name}")
        
        efficiency_factor = 0.7
        base_throughput = sm_count * efficiency_factor * 1000
        
        if self._max_vram_gb is None or self._max_vram_gb <= 0:
            raise ValueError("Max VRAM not initialized")
        
        vram_factor = total_vram_gb / self._max_vram_gb
        estimated_tokens_per_sec = base_throughput * vram_factor
        
        profile = PerformanceProfile(
            sm_count=sm_count,
            total_vram_gb=total_vram_gb,
            max_num_seqs=max_num_seqs,
            estimated_tokens_per_sec=estimated_tokens_per_sec,
            last_updated=current_time
        )
        
        self._performance_profiles[deployment_name] = profile
        self._last_profile_update[deployment_name] = current_time
        
        return profile
    
    async def _count_tokens(self, text: Union[str, List[str]]) -> int:
        """Count tokens accurately using tokenizer from a deployment."""
        deployment_name = await self._get_tokenizer_deployment()
        handles = self._get_handles()
        handle = handles[deployment_name]
        
        token_count = await handle.count_tokens.remote(text)
        return token_count
    
    async def _extract_request_features(self, request: Union[str, Dict]) -> Dict:
        """Extract features from request for routing decisions."""
        if isinstance(request, dict):
            prompts = request.get("prompts") or ([request.get("prompt")] if request.get("prompt") else [])
            max_tokens = request.get("max_tokens", 100)
        else:
            prompts = [str(request)]
            max_tokens = 100
        
        if not prompts:
            prompts = [""]
        
        total_prompt_tokens = 0
        for prompt in prompts:
            prompt_tokens = await self._count_tokens(prompt)
            total_prompt_tokens += prompt_tokens
        
        avg_prompt_tokens = total_prompt_tokens / len(prompts) if prompts else 0
        total_sequence_tokens = avg_prompt_tokens + max_tokens
        
        return {
            "num_prompts": len(prompts),
            "prompt_tokens": total_prompt_tokens,
            "avg_prompt_tokens": avg_prompt_tokens,
            "max_tokens": max_tokens,
            "total_sequence_tokens": total_sequence_tokens,
            "is_batch": len(prompts) > 1
        }
    
    async def _predict_latency(self, deployment_name: str, request_features: Dict) -> float:
        """Predict latency for a request on a specific deployment."""
        profile = await self._calculate_performance_profile(deployment_name)
        capacity_info = await self._get_capacity_info(deployment_name)
        
        queue_depth = capacity_info.get("approximate_queue_depth", 0)
        max_num_seqs = profile.max_num_seqs
        
        load_factor = 1.0 + (queue_depth / max(max_num_seqs, 1)) * 0.5
        
        total_tokens = request_features["total_sequence_tokens"]
        if request_features["is_batch"]:
            num_prompts = request_features["num_prompts"]
            total_tokens = request_features["prompt_tokens"] + (request_features["max_tokens"] * num_prompts)
        
        if profile.estimated_tokens_per_sec <= 0:
            raise ValueError(f"Invalid estimated_tokens_per_sec for deployment {deployment_name}")
        
        base_latency = total_tokens / profile.estimated_tokens_per_sec
        
        predicted_latency = base_latency * load_factor
        
        return predicted_latency
    
    async def _calculate_predictive_score(self, deployment_name: str, request_features: Dict) -> float:
        """Calculate predictive routing score (higher is better)."""
        profile = await self._calculate_performance_profile(deployment_name)
        capacity_info = await self._get_capacity_info(deployment_name)
        
        predicted_latency = await self._predict_latency(deployment_name, request_features)
        
        queue_depth = capacity_info.get("approximate_queue_depth", 0)
        max_num_seqs = profile.max_num_seqs
        available_capacity = max(0, max_num_seqs - queue_depth)
        
        if profile.estimated_tokens_per_sec <= 0:
            raise ValueError(f"Invalid estimated_tokens_per_sec for deployment {deployment_name}")
        
        performance_factor = profile.estimated_tokens_per_sec / 1000.0
        
        queue_load = queue_depth / max(max_num_seqs, 1)
        
        if predicted_latency > 0:
            score = performance_factor / (predicted_latency * (1.0 + queue_load))
        else:
            score = performance_factor * available_capacity
        
        score *= available_capacity
        
        return score
    
    async def _select_best_deployment(self, request: Union[str, Dict]) -> str:
        """Select best deployment using predictive scoring."""
        request_features = await self._extract_request_features(request)
        
        best_deployment = None
        best_score = float('-inf')
        
        for deployment_name in self.gpu_deployment_names:
            score = await self._calculate_predictive_score(deployment_name, request_features)
            if score > best_score:
                best_score = score
                best_deployment = deployment_name
        
        if best_deployment is None:
            raise RuntimeError("Failed to select best deployment")
        
        return best_deployment
    
    async def _split_batch(self, prompts: List[str], max_tokens: int) -> List[Tuple[str, List[str]]]:
        """Split a large batch across multiple GPUs proportionally."""
        if len(prompts) <= self._batch_split_threshold:
            deployment_name = await self._select_best_deployment({"prompts": prompts, "max_tokens": max_tokens})
            return [(deployment_name, prompts)]
        
        total_capacity = 0
        deployment_capacities = {}
        
        for deployment_name in self.gpu_deployment_names:
            capacity_info = await self._get_capacity_info(deployment_name)
            profile = await self._calculate_performance_profile(deployment_name)
            
            queue_depth = capacity_info.get("approximate_queue_depth", 0)
            max_num_seqs = profile.max_num_seqs
            available_capacity = max(0, max_num_seqs - queue_depth)
            
            if profile.estimated_tokens_per_sec <= 0:
                raise ValueError(f"Invalid estimated_tokens_per_sec for deployment {deployment_name}")
            
            performance_factor = profile.estimated_tokens_per_sec / 1000.0
            weighted_capacity = available_capacity * performance_factor
            
            deployment_capacities[deployment_name] = weighted_capacity
            total_capacity += weighted_capacity
        
        if total_capacity == 0:
            raise RuntimeError("No available capacity for batch splitting")
        
        splits = []
        prompt_idx = 0
        
        for deployment_name in self.gpu_deployment_names:
            if deployment_capacities[deployment_name] <= 0:
                continue
            
            proportion = deployment_capacities[deployment_name] / total_capacity
            num_prompts = max(1, int(len(prompts) * proportion))
            
            if prompt_idx >= len(prompts):
                break
            
            end_idx = min(prompt_idx + num_prompts, len(prompts))
            split_prompts = prompts[prompt_idx:end_idx]
            
            if split_prompts:
                splits.append((deployment_name, split_prompts))
                prompt_idx = end_idx
        
        if prompt_idx < len(prompts):
            remaining = prompts[prompt_idx:]
            if remaining:
                if splits:
                    splits[-1] = (splits[-1][0], splits[-1][1] + remaining)
                else:
                    deployment_name = await self._select_best_deployment({"prompts": remaining, "max_tokens": max_tokens})
                    splits.append((deployment_name, remaining))
        
        if not splits:
            raise RuntimeError("Failed to create batch splits")
        
        return splits
    
    async def _route_batch(self, request: Dict) -> List:
        """Route a batch request, splitting across GPUs if needed."""
        prompts = request.get("prompts", [])
        max_tokens = request.get("max_tokens", 100)
        
        if len(prompts) <= self._batch_split_threshold:
            deployment_name = await self._select_best_deployment(request)
            handles = self._get_handles()
            result = await handles[deployment_name].remote(request)
            return result if isinstance(result, list) else [result]
        
        splits = await self._split_batch(prompts, max_tokens)
        handles = self._get_handles()
        
        tasks = []
        split_indices = []
        current_idx = 0
        
        for deployment_name, split_prompts in splits:
            split_request = request.copy()
            split_request["prompts"] = split_prompts
            split_indices.append((current_idx, current_idx + len(split_prompts)))
            current_idx += len(split_prompts)
            tasks.append(handles[deployment_name].remote(split_request))
        
        results = await asyncio.gather(*tasks)
        
        aggregated = []
        for result in results:
            if isinstance(result, list):
                aggregated.extend(result)
            else:
                aggregated.append(result)
        
        return aggregated
    
    async def __call__(self, request: Union[str, Dict]) -> Union[str, List]:
        """Route request to best deployment or split batch across multiple."""
        if isinstance(request, dict) and request.get("prompts") and len(request.get("prompts", [])) > 1:
            return await self._route_batch(request)
        
        deployment_name = await self._select_best_deployment(request)
        handles = self._get_handles()
        handle = handles[deployment_name]
        
        try:
            if isinstance(request, dict):
                result = await handle.remote(request)
            else:
                result = await handle.remote({"prompt": request})
            return result
        except Exception as e:
            raise e
    
    async def get_max_num_seqs(self) -> int:
        """Get total max_num_seqs across all deployments."""
        total = 0
        for deployment_name in self.gpu_deployment_names:
            capacity_info = await self._get_capacity_info(deployment_name)
            total += capacity_info.get("max_num_seqs", 0)
        return total
