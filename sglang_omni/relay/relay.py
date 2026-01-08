import asyncio
import imp
from typing import List, Optional, Union
import logging

from descriptor import Descriptor
from topology import PipelineConfig
from descriptor import TransferPacket
from mooncake.mooncake_engine import MooncakeTransferEngine

logger = logging.getLogger(__name__)


async def run_relay(
    rank: int, upstream:str,down_stream:str, engine:str):
    relay = Relay(rank, upstream, down_stream, engine)
    while True:
   

class Relay:
    def __init__(self, rank: int, topology: "PipelineConfig", engine):

        self.rank = rank
        self.topology = topology
        self.engine = MooncakeTransferEngine
        

        self.default_upstream_count = len(topology.get_prev_ranks(rank))
        
        self.default_downstream_ranks = topology.get_next_ranks(rank)

    async def put(self, batch: List[Descriptor], target_ranks: Optional[List[int]] = None):

        destinations = target_ranks if target_ranks is not None else self.default_downstream_ranks
        
        if not destinations:
            logger.warning(f"[Rank {self.rank}] No downstream targets for put().")
            return

        # 将 Descriptor 转换为 TransferPacket
        # to_packet 内部会调用 register_tensor 注册每个 tensor
        packet = batch.to_packet(
            register_fn=self.engine.batch_register,
            my_rank=self.rank
        )


        send_tasks = []
        for target in destinations:
            send_tasks.append(self.engine.send(target, packet))
        
        await asyncio.gather(*send_tasks)
        # logger.debug(f"Sent batch {batch.req_id} to {destinations}")

    async def recv(self, expected_count: Optional[int] = None) -> "Descriptor":

        count = expected_count if expected_count is not None else self.default_upstream_count
        
        if count == 0:
            raise RuntimeError(f"[Rank {self.rank}] recv called but no upstream sources defined.")


        rx_tasks = [self._recv_single_packet() for _ in range(count)]
        
        descriptors = await asyncio.gather(*rx_tasks)


        aggregated_batch = "Descriptor".merge(descriptors)
        
        return aggregated_batch

    async def _recv_single_packet(self) -> "Descriptor":

        packet = await self.engine.recv()
        
        descriptor = "Descriptor".from_packet(packet, self.engine)
        
        # If engine supports notify_buffers_ready (e.g., MooncakeEngine),
        # notify sender that buffers are ready for data transfer
        if hasattr(self.engine, 'notify_buffers_ready'):
            await self.engine.notify_buffers_ready(packet)
        
        return descriptor