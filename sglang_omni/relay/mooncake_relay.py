import json
import logging
import os
import asyncio
import socket
import struct
from typing import List, Optional, Dict
import torch

from sglang.srt.environ import envs
from sglang.srt.utils import get_free_port, maybe_wrap_ipv6_address

logger = logging.getLogger(__name__)


def get_ib_devices_for_gpu(ib_device_str: Optional[str], gpu_id: int) -> Optional[str]:
    """
    Parse IB device string and get IB devices for a specific GPU ID.

    Supports all the following formats:
    1. Old format: "ib0, ib1, ib2"
    2. New format: {0: "ib0, ib1", 1: "ib2, ib3", 2: "ib4"}
    3. JSON file: path to a JSON file containing the mapping

    Args:
        ib_device_str: The original IB device string or path to JSON file
        gpu_id: The GPU ID to get devices for

    Returns:
        IB devices string for the GPU, or None if not available
    """
    if ib_device_str is None or not ib_device_str.strip():
        return None

    ib_device_str = ib_device_str.strip()

    # Check if it's a JSON file first and load its content
    is_json_file = ib_device_str.endswith(".json")
    if is_json_file:
        try:
            if os.path.isfile(ib_device_str):
                with open(ib_device_str, "r") as f:
                    ib_device_str = f.read()
            else:
                # File doesn't exist, treat as old format
                raise RuntimeError(f"File {ib_device_str} does not exist.")
        except (IOError, OSError) as e:
            # File reading failed, raise exception
            raise RuntimeError(f"Failed to read JSON file {ib_device_str}: {e}") from e

    # Check if it's JSON format (new format)
    try:
        parsed_json = json.loads(ib_device_str)
        if isinstance(parsed_json, dict):
            # Validate format - keys should be integers (or string rep), values should be strings
            gpu_mapping = {}
            for gpu_key, ib_devices in parsed_json.items():
                if (
                    isinstance(gpu_key, str)
                    and gpu_key.isdigit()
                    and isinstance(ib_devices, str)
                ):
                    gpu_mapping[int(gpu_key)] = ib_devices.strip()
                elif isinstance(gpu_key, int) and isinstance(ib_devices, str):
                    gpu_mapping[gpu_key] = ib_devices.strip()
                else:
                    raise ValueError(
                        f"Invalid format: keys must be integers (or string representations of integers) and values must be strings"
                    )

            if not gpu_mapping:
                raise ValueError("No valid GPU mappings found in JSON")

            # Return devices for specific GPU
            if gpu_id in gpu_mapping:
                return gpu_mapping[gpu_id]
            else:
                raise ValueError(
                    f"No IB devices configured for GPU {gpu_id}. Available GPUs: {list(gpu_mapping.keys())}"
                )

    except json.JSONDecodeError:
        if is_json_file:
            # It was supposed to be a JSON file but failed to parse
            raise RuntimeError(
                f"Failed to parse JSON content from file {ib_device_str}"
            )
        # Not JSON format, treat as old format - return same devices for all GPUs
        return ib_device_str


class MooncakeTransferEngine:
    """Low-level mooncake transfer engine wrapper."""

    def __init__(self, hostname: str, gpu_id: int, ib_device: Optional[str] = None):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run SGLang with MooncakeTransferEngine."
            ) from e

        self.engine = TransferEngine()
        self.hostname = hostname
        self.gpu_id = gpu_id
        self.ib_device = get_ib_devices_for_gpu(ib_device, gpu_id)

        self.initialize(
            hostname=self.hostname,
            device_name=self.ib_device,
        )
        self.session_id = (
            f"{maybe_wrap_ipv6_address(self.hostname)}:{self.engine.get_rpc_port()}"
        )

    def register(self, ptr, length):
        try:
            ret_value = self.engine.register_memory(ptr, length)
        except Exception:
            # Mark register as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake memory registration %s failed.", ptr)

    def deregister(self, ptr):
        try:
            ret_value = self.engine.unregister_memory(ptr)
        except Exception:
            # Mark deregister as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake memory deregistration %s failed.", ptr)

    def batch_register(self, ptrs: List[int], lengths: List[int]) -> int:
        """Batch register multiple memory regions."""
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            # Mark batch register as failed
            ret_value = -1
            if not hasattr(self.engine, "batch_register_memory"):
                raise RuntimeError(
                    "Mooncake's batch register requires a newer version of mooncake-transfer-engine. "
                    "Please upgrade Mooncake."
                )

        if ret_value != 0:
            logger.debug("Mooncake batch memory registration failed.")
        return ret_value

    def batch_deregister(self, ptrs: List[int]) -> int:
        """Batch deregister multiple memory regions."""
        try:
            ret_value = self.engine.batch_unregister_memory(ptrs)
        except Exception:
            # Mark batch deregister as failed
            ret_value = -1

        if ret_value != 0:
            logger.debug("Mooncake batch memory deregistration failed.")
        return ret_value


    def initialize(
        self,
        hostname: str,
        device_name: Optional[str],
    ) -> None:
        """Initialize the mooncake instance."""
        if envs.ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE.get():
            npu_phy_id = envs.ASCEND_NPU_PHY_ID.get()
            if npu_phy_id == -1:
                hostname += f":{get_free_port()}:npu_{self.gpu_id}"
            else:
                hostname += f":{get_free_port()}:npu_{npu_phy_id}"
            ret_value = self.engine.initialize(
                hostname,
                "P2PHANDSHAKE",
                "ascend",
                device_name if device_name is not None else "",
            )
        else:
            ret_value = self.engine.initialize(
                hostname,
                "P2PHANDSHAKE",
                "rdma",
                device_name if device_name is not None else "",
            )
        if ret_value != 0:
            logger.error("Mooncake Transfer Engine initialization failed.")
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""
        try:
            # the first time: based on session_id (which contains remote_ip) to construct a queue pair, and cache the queue pair
            # later: based on the cached queue pair to send data
            ret = self.engine.transfer_sync_write(
                session_id, buffer, peer_buffer_address, length
            )
        except Exception:
            # Mark transfer request as failed
            ret = -1

        if ret < 0:
            # Do not raise an exception here, since some transfer requests fail should be accepted and the execution thread should not be stopped.
            logger.debug(
                "Failed to transfer data from %s to %s - %s.",
                buffer,
                session_id,
                peer_buffer_address,
            )

        return ret

    def batch_transfer_sync(
        self,
        session_id: str,
        buffers: List[int],
        peer_buffer_addresses: List[int],
        lengths: List[int],
    ) -> int:
        """Synchronously transfer data to the specified addresses in batches."""
        try:
            ret = self.engine.batch_transfer_sync_write(
                session_id, buffers, peer_buffer_addresses, lengths
            )
        except Exception:
            ret = -1
            # Inform user to upgrade mooncake-transfer-engine >= 0.3.4.post2
            if not hasattr(self.engine, "batch_transfer_sync_write"):
                raise RuntimeError(
                    "Mooncake's batch transfer requires mooncake-transfer-engine >= 0.3.4.post2. "
                    "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'"
                )

        if ret < 0:
            logger.debug(
                "Failed to batch transfer data. Buffers: %s, Session: %s, Peer addresses: %s",
                buffers,
                session_id,
                peer_buffer_addresses,
            )
        return ret

    def get_session_id(self):
        return self.session_id


class MooncakeEngine:
    """
    High-level engine interface for Relay, wrapping MooncakeTransferEngine.
    
    This class provides:
    - Tensor registration and opening (register_tensor, open_tensor)
    - Async send/recv for TransferPackets (send, recv)
    - Session management for peer connections
    - Control channel (TCP) for metadata transfer
    - Data channel (mooncake) for tensor data transfer
    """
    
    def __init__(
        self,
        rank: int,
        topology: "PipelineConfig",
        hostname: Optional[str] = None,
        gpu_id: int = 0,
        ib_device: Optional[str] = None,
    ):
        """
        Initialize MooncakeEngine.
        
        Args:
            rank: Current rank ID
            topology: Pipeline topology configuration
            hostname: Hostname for mooncake (defaults to node IP from topology)
            gpu_id: GPU ID for this rank
            ib_device: IB device string or JSON file path
        """
        from topology import PipelineConfig  # noqa: F401
        
        self.rank = rank
        self.topology = topology
        self.gpu_id = gpu_id
        
        # Get node config
        node = topology.get_node(rank)
        
        # Get hostname from topology if not provided
        if hostname is None:
            hostname = node.ip
        
        # Initialize low-level mooncake engine
        self.mooncake = MooncakeTransferEngine(
            hostname=hostname,
            gpu_id=gpu_id,
            ib_device=ib_device
        )
        
        # Tensor registry: remote_ptr -> (tensor, registered_ptr, length)
        # registered_ptr is the actual pointer registered with mooncake
        self._tensor_registry: Dict[int, tuple] = {}
        self._next_remote_ptr = 1  # Simple counter for remote_ptr allocation
        
        # Reverse registry: tensor.data_ptr() -> remote_ptr
        self._ptr_to_remote: Dict[int, int] = {}
        
        # Session management: rank -> session_id
        self._sessions: Dict[int, str] = {}
        
        # Control channel: TCP connections for TransferPacket metadata
        self._ctrl_server: Optional[asyncio.Server] = None
        self._ctrl_connections: Dict[int, asyncio.StreamWriter] = {}  # rank -> writer
        self._ctrl_readers: Dict[int, asyncio.StreamReader] = {}  # rank -> reader
        self._recv_queue: asyncio.Queue = asyncio.Queue()
        
        # Buffer address mapping: (sender_rank, remote_ptr) -> peer_buffer_address
        # Used to map remote_ptr to actual buffer addresses for tensor transfer
        self._buffer_address_map: Dict[tuple, int] = {}
        
        # Pending transfers: waiting for buffer addresses from receiver
        self._pending_transfers: Dict[tuple, asyncio.Event] = {}  # (target_rank, req_id) -> event
        
        # Background tasks
        self._recv_task: Optional[asyncio.Task] = None
        
        # Start control channel server
        self._start_control_server(node.ctrl_port)
        
        logger.info(f"[Rank {rank}] MooncakeEngine initialized with session_id={self.mooncake.get_session_id()}")
    
    def register_tensor(self, tensor: torch.Tensor) -> int:
        """
        Register a tensor with mooncake and return a remote_ptr handle.
        
        Args:
            tensor: PyTorch tensor to register
            
        Returns:
            remote_ptr: Handle that can be used to reference this tensor remotely
        """
        # Get the underlying memory pointer
        if not tensor.is_cuda:
            raise ValueError("Mooncake only supports CUDA tensors")
        
        # Get the data pointer
        ptr = tensor.data_ptr()
        length = tensor.nbytes
        
        # Check if already registered
        if ptr in self._ptr_to_remote:
            return self._ptr_to_remote[ptr]
        
        # Register with mooncake
        self.mooncake.register(ptr, length)
        
        # Allocate a remote_ptr handle
        remote_ptr = self._next_remote_ptr
        self._next_remote_ptr += 1
        
        # Store in registry
        self._tensor_registry[remote_ptr] = (tensor, ptr, length)
        self._ptr_to_remote[ptr] = remote_ptr
        
        logger.debug(f"[Rank {self.rank}] Registered tensor {remote_ptr}: ptr={ptr}, size={length}")
        
        return remote_ptr
    
    def open_tensor(
        self,
        remote_ptr: int,
        shape: tuple,
        dtype_str: str,
        device: str
    ) -> torch.Tensor:
        """
        Open a tensor from a remote_ptr handle.
        
        This allocates a local tensor buffer that will receive data from the remote.
        After creating the buffer, it notifies the sender with the buffer address.
        
        Args:
            remote_ptr: Remote pointer handle (used as identifier)
            shape: Tensor shape
            dtype_str: Tensor dtype as string (e.g., "float16")
            device: Device string (e.g., "cuda:0")
            sender_rank: Optional sender rank (for notifying buffer address)
            
        Returns:
            torch.Tensor: Local tensor buffer for receiving data
        """
        # Map dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int64": torch.int64,
            "uint8": torch.uint8,
        }
        dtype = dtype_map.get(dtype_str, torch.float32)
        
        # Create a new tensor on the target device
        tensor = torch.empty(shape, dtype=dtype, device=device)
        
        # Register this tensor so we can use it for receiving
        ptr = tensor.data_ptr()
        length = tensor.nbytes
        self.mooncake.register(ptr, length)
        
        # Store in registry with the remote_ptr as key
        self._tensor_registry[remote_ptr] = (tensor, ptr, length)
        self._ptr_to_remote[ptr] = remote_ptr
        
        logger.debug(f"[Rank {self.rank}] Opened tensor {remote_ptr}: shape={shape}, dtype={dtype_str}, local_ptr={ptr}")
        
        return tensor
    
    def _start_control_server(self, port: int):
        """Start TCP server for control channel."""
        async def _handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            """Handle incoming control channel connections."""
            try:
                while True:
                    # Read message type (1 byte: 'P' for packet, 'B' for buffer address)
                    msg_type_bytes = await reader.readexactly(1)
                    if not msg_type_bytes:
                        break
                    msg_type = msg_type_bytes.decode('ascii')
                    
                    # Read message length (4 bytes)
                    length_bytes = await reader.readexactly(4)
                    msg_length = struct.unpack('>I', length_bytes)[0]
                    
                    # Read message data
                    msg_data = await reader.readexactly(msg_length)
                    msg_dict = json.loads(msg_data.decode('utf-8'))
                    
                    if msg_type == 'P':
                        # TransferPacket
                        from descriptor import TransferPacket  # noqa: F401
                        packet = TransferPacket(**msg_dict)
                        
                        # Put in recv queue
                        await self._recv_queue.put(packet)
                        
                        logger.debug(f"[Rank {self.rank}] Received control packet {packet.req_id}")
                    elif msg_type == 'B':
                        # Buffer address notification
                        sender_rank = msg_dict['sender_rank']
                        req_id = msg_dict['req_id']
                        buffer_map = msg_dict['buffer_map']  # {remote_ptr: buffer_address}
                        
                        # Store buffer addresses
                        for remote_ptr, buffer_addr in buffer_map.items():
                            self._buffer_address_map[(sender_rank, remote_ptr)] = buffer_addr
                        
                        # Notify pending transfer
                        key = (sender_rank, req_id)
                        if key in self._pending_transfers:
                            self._pending_transfers[key].set()
                        
                        logger.debug(f"[Rank {self.rank}] Received buffer addresses for req_id {req_id} from rank {sender_rank}")
            except asyncio.IncompleteReadError:
                # Client disconnected
                pass
            except Exception as e:
                logger.error(f"[Rank {self.rank}] Error in control server: {e}")
            finally:
                writer.close()
                await writer.wait_closed()
        
        async def _start_server():
            self._ctrl_server = await asyncio.start_server(
                _handle_client,
                '0.0.0.0',
                port
            )
            logger.info(f"[Rank {self.rank}] Control server started on port {port}")
        
        # Start server in background
        asyncio.create_task(_start_server())
    
    async def _ensure_control_connection(self, target_rank: int):
        """Ensure control channel connection to target_rank exists."""
        if target_rank in self._ctrl_connections:
            return
        
        target_node = self.topology.get_node(target_rank)
        
        # Connect to target's control server
        try:
            reader, writer = await asyncio.open_connection(
                target_node.ip,
                target_node.ctrl_port
            )
            self._ctrl_connections[target_rank] = writer
            self._ctrl_readers[target_rank] = reader
            logger.debug(f"[Rank {self.rank}] Connected to rank {target_rank} control channel")
        except Exception as e:
            logger.error(f"[Rank {self.rank}] Failed to connect to rank {target_rank}: {e}")
            raise
    
    def _get_session_id(self, target_rank: int) -> str:
        """Get or create session ID for a target rank."""
        if target_rank not in self._sessions:
            # Get target node info from topology
            target_node = self.topology.get_node(target_rank)
            # Construct session_id: hostname:port
            # Note: We need the target's mooncake RPC port, which should be
            # communicated during handshake. For now, we'll use a placeholder.
            # In practice, you might need to exchange session_ids via control channel.
            session_id = f"{target_node.ip}:{target_node.ctrl_port}"
            self._sessions[target_rank] = session_id
        
        return self._sessions[target_rank]
    
    async def send(self, target_rank: int, packet: "TransferPacket") -> None:
        """
        Asynchronously send a TransferPacket to target_rank.
        
        This method:
        1. Transfers tensor data using mooncake
        2. Sends packet metadata via control channel
        
        Args:
            target_rank: Target rank to send to
            packet: TransferPacket containing tensor metadata
        """
        from descriptor import TransferPacket  # noqa: F401
        
        # Ensure control connection exists
        await self._ensure_control_connection(target_rank)
        
        # Get session ID for target
        session_id = self._get_session_id(target_rank)
        
        # Step 1: Send packet metadata via control channel first
        # This allows receiver to create buffers via open_tensor
        packet_dict = packet.model_dump()
        packet_json = json.dumps(packet_dict).encode('utf-8')
        packet_length = len(packet_json)
        
        writer = self._ctrl_connections[target_rank]
        writer.write(b'P')  # Message type: Packet
        writer.write(struct.pack('>I', packet_length))
        writer.write(packet_json)
        await writer.drain()
        
        logger.debug(f"[Rank {self.rank}] Sent packet metadata {packet.req_id} to rank {target_rank}")
        
        # Step 2: Wait for receiver to create buffers and send back buffer addresses
        # The receiver will call open_tensor and send buffer addresses back
        transfer_key = (target_rank, packet.req_id)
        if transfer_key not in self._pending_transfers:
            self._pending_transfers[transfer_key] = asyncio.Event()
        
        # Wait for buffer addresses (with timeout)
        try:
            await asyncio.wait_for(self._pending_transfers[transfer_key].wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning(f"[Rank {self.rank}] Timeout waiting for buffer addresses from rank {target_rank}")
            # Continue anyway, might use fallback
        
        # Step 3: Transfer tensor data using mooncake
        loop = asyncio.get_event_loop()
        
        # Prepare tensor transfers
        buffers = []
        peer_addresses = []
        lengths = []
        
        for name, meta in packet.tensors.items():
            # Get local tensor from registry
            if meta.remote_ptr not in self._tensor_registry:
                raise ValueError(f"Tensor {name} with remote_ptr {meta.remote_ptr} not found in registry")
            
            local_tensor, local_ptr, local_length = self._tensor_registry[meta.remote_ptr]
            
            # Get peer buffer address from mapping
            buffer_key = (target_rank, meta.remote_ptr)
            if buffer_key in self._buffer_address_map:
                peer_addr = self._buffer_address_map[buffer_key]
            else:
                # Fallback: use remote_ptr as address (simplified mode)
                logger.warning(f"[Rank {self.rank}] Buffer address not found for {buffer_key}, using fallback")
                peer_addr = meta.remote_ptr
            
            buffers.append(local_ptr)
            peer_addresses.append(peer_addr)
            lengths.append(meta.size)
        
        # Transfer tensors using mooncake (run in executor to avoid blocking)
        if buffers:
            await loop.run_in_executor(
                None,
                lambda: self.mooncake.batch_transfer_sync(
                    session_id, buffers, peer_addresses, lengths
                ) if len(buffers) > 1 else self.mooncake.transfer_sync(
                    session_id, buffers[0], peer_addresses[0], lengths[0]
                )
            )
            logger.debug(f"[Rank {self.rank}] Transferred {len(buffers)} tensors to rank {target_rank}")
        
        # Cleanup
        if transfer_key in self._pending_transfers:
            del self._pending_transfers[transfer_key]
    
    async def recv(self) -> "TransferPacket":
        """
        Asynchronously receive a TransferPacket.
        
        Returns:
            TransferPacket: Received packet
        """
        packet = await self._recv_queue.get()
        logger.debug(f"[Rank {self.rank}] Received packet {packet.req_id}")
        
        # After receiving packet, if sender_rank is known, we'll notify sender
        # about buffer addresses after tensors are created in from_packet
        # This is handled by a separate method that should be called after from_packet
        return packet
    
    async def notify_buffers_ready(self, packet: "TransferPacket"):
        """
        Notify sender that buffers are ready for a received packet.
        This should be called after from_packet has created all tensors.
        
        Args:
            packet: The TransferPacket that was received and processed
        """
        if packet.sender_rank is None:
            return
        
        sender_rank = packet.sender_rank
        
        # Collect buffer addresses for all tensors in the packet
        buffer_map = {}
        for name, meta in packet.tensors.items():
            if meta.remote_ptr in self._tensor_registry:
                tensor, ptr, length = self._tensor_registry[meta.remote_ptr]
                buffer_map[meta.remote_ptr] = ptr
        
        if not buffer_map:
            return
        
        # Ensure control connection exists
        await self._ensure_control_connection(sender_rank)
        
        # Send buffer address notification
        notification = {
            'sender_rank': self.rank,  # We are the receiver, sender is the original sender
            'req_id': packet.req_id,
            'buffer_map': buffer_map
        }
        
        notification_json = json.dumps(notification).encode('utf-8')
        notification_length = len(notification_json)
        
        writer = self._ctrl_connections[sender_rank]
        writer.write(b'B')  # Message type: Buffer addresses
        writer.write(struct.pack('>I', notification_length))
        writer.write(notification_json)
        await writer.drain()
        
        logger.debug(f"[Rank {self.rank}] Notified rank {sender_rank} about buffer addresses for req_id {packet.req_id}")
    
    def get_session_id(self) -> str:
        """Get this engine's session ID for handshake."""
        return self.mooncake.get_session_id()
    
    def cleanup(self):
        """Clean up resources."""
        # Close control connections
        for writer in self._ctrl_connections.values():
            writer.close()
        self._ctrl_connections.clear()
        self._ctrl_readers.clear()
        
        # Close control server
        if self._ctrl_server:
            self._ctrl_server.close()
        
        # Deregister all tensors
        for remote_ptr, (tensor, ptr, length) in self._tensor_registry.items():
            self.mooncake.deregister(ptr)
        
        self._tensor_registry.clear()
        self._ptr_to_remote.clear()
