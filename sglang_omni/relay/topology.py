from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Set

class NodeConfig(BaseModel):
    """
    Configuration for a single pipeline node.
    """
    rank: int                     # Globally unique node ID (stage ID)
    ip: str                       # IP address for control channel
    ctrl_port: int                # Control port (metadata transfer)
    device_name: str = "cuda:0"   # PyTorch device identifier

    # Backend-specific fields (optional)
    # mooncake_device: str = "mlx5_0"

#*****************************temporary***********************************
class PipelineConfig(BaseModel):
    """
    Pipeline topology configuration.
    Supports both linear pipeline (1-to-1) and multi-to-1 fan-in patterns.
    """
    nodes: List[NodeConfig]
    
    # Explicit topology mapping: rank -> list of upstream ranks
    # If None, falls back to linear pipeline (rank-1)
    upstream_map: Optional[Dict[int, List[int]]] = None
    
    # Explicit topology mapping: rank -> list of downstream ranks  
    # If None, falls back to linear pipeline (rank+1)
    downstream_map: Optional[Dict[int, List[int]]] = None

    # Internal cache for fast lookup
    _rank_map: Dict[int, NodeConfig] = {}

    def model_post_init(self, __context):
        # Build rank -> node index
        self._rank_map = {n.rank: n for n in self.nodes}

    def get_node(self, rank: int) -> NodeConfig:
        # Lazy rebuild in case post-init was skipped
        if rank not in self._rank_map:
            self._rank_map = {n.rank: n for n in self.nodes}
        return self._rank_map[rank]

    def get_next_rank(self, current_rank: int) -> Optional[int]:
        """
        Return downstream rank for sender (backward compatible, 1-to-1).
        For multiple downstreams, use get_next_ranks().
        """
        next_r = current_rank + 1
        return next_r if next_r in self._rank_map else None

    def get_prev_rank(self, current_rank: int) -> Optional[int]:
        """
        Return upstream rank for receiver (backward compatible, 1-to-1).
        For multiple upstreams, use get_prev_ranks().
        """
        prev_r = current_rank - 1
        return prev_r if prev_r in self._rank_map else None
    
    def get_prev_ranks(self, current_rank: int) -> List[int]:
        """
        Return all upstream ranks for receiver (supports multi-to-1).
        If explicit upstream_map is provided, use it; otherwise fall back to linear.
        """
        if self.upstream_map and current_rank in self.upstream_map:
            # Filter to only include ranks that exist
            return [r for r in self.upstream_map[current_rank] if r in self._rank_map]
        else:
            # Fall back to linear pipeline
            prev_r = current_rank - 1
            return [prev_r] if prev_r in self._rank_map else []
    
    def get_next_ranks(self, current_rank: int) -> List[int]:
        """
        Return all downstream ranks for sender (supports 1-to-multi).
        If explicit downstream_map is provided, use it; otherwise fall back to linear.
        """
        if self.downstream_map and current_rank in self.downstream_map:
            # Filter to only include ranks that exist
            return [r for r in self.downstream_map[current_rank] if r in self._rank_map]
        else:
            # Fall back to linear pipeline
            next_r = current_rank + 1
            return [next_r] if next_r in self._rank_map else []