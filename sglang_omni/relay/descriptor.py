
import torch
from typing import Optional, Tuple, Any
from pydantic import BaseModel, ConfigDict, Field
import logging

logger = logging.getLogger(__name__)


class SerializedTensorMeta(BaseModel):
    """Serialized tensor metadata for network transmission."""
    
    model_config = ConfigDict(frozen=True)
    
    ptr: int = Field(description="Memory pointer address")
    size: int = Field(description="Size in bytes")
    shape: Tuple[int, ...] = Field(description="Tensor shape")
    dtype_str: str = Field(description="Data type string, e.g., 'float16'")
    device: str = Field(description="Device identifier, e.g., 'cuda:0' or 'cpu'")
    type: str = Field(description="Data, e.g., video_embedding")


class Descriptor:
    """
    Memory descriptor for tensor-based memory management.
    
    A Descriptor represents a single memory segment from a torch.Tensor.
    It manages the registration and lifecycle of the tensor memory.
    """
    
    def __init__(self, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Argument `tensor` must be `torch.Tensor`, got {type(tensor)}")
        
        # Extract tensor metadata
        self._ptr: int = tensor.data_ptr()
        self._size: int = tensor.numel() * tensor.element_size()
        self._device: str = str(tensor.device)
        self._shape: Tuple[int, ...] = tuple(tensor.shape)
        self._dtype_str: str = str(tensor.dtype).split(".")[-1]
        
        # Keep reference to prevent garbage collection
        self._tensor_ref: torch.Tensor = tensor
        
        # Registration state
        self._connector: Optional[Any] = None
        self._is_registered: bool = False
        self._registration_handle: Optional[Any] = None
        
        # Cached serialized metadata
        self._serialized: Optional[SerializedTensorMeta] = None
        

    @property
    def metadata(self) -> SerializedTensorMeta:
        """
        Get serialized metadata for network transmission.
        
        Returns
        -------
        SerializedTensorMeta
            Serialized tensor metadata
        """
        if self._serialized is None:
            self._serialized = SerializedTensorMeta(
                ptr=self._ptr,
                size=self._size,
                shape=self._shape,
                dtype_str=self._dtype_str,
                device=self._device,
            )
        return self._serialized
    
    @staticmethod
    def from_serialized(serialized: SerializedTensorMeta) -> "Descriptor":

        dtype_map = {
            'bfloat16': torch.bfloat16,
        }
        dtype = dtype_map.get(serialized.dtype_str, torch.bfloat16)
        device = serialized.device
        
        placeholder = torch.empty(
            serialized.shape,
            dtype=dtype,
            device=device
        )
        
        desc = Descriptor(placeholder)
        desc._ptr = serialized.ptr
        desc._serialized = serialized
        
        return desc
