# SPDX-License-Identifier: Apache-2.0
"""Base classes for relay operations."""

from abc import ABC, abstractmethod
from typing import Any


class BaseReadOperation(ABC):
    """Abstract base class for read operations across different relay implementations.

    This class defines the common interface that all read operations must implement,
    regardless of the underlying transport mechanism (SHM, NIXL, etc.).
    """

    @abstractmethod
    async def wait_for_completion(self) -> None:
        """Wait for the read operation to complete.

        This method should block (or be a no-op for synchronous operations)
        until the data transfer is complete and the data is available.

        For synchronous operations (like SHM), this can be a no-op.
        For asynchronous operations (like NIXL), this should wait until completion.
        """


class BaseReadableOperation(ABC):
    """Abstract base class for readable operations (write side of a read).

    This class defines the common interface for operations that expose data
    for reading by remote peers.
    """

    @abstractmethod
    def metadata(self) -> Any:
        """Return metadata describing this readable operation.

        The metadata should contain sufficient information for a remote peer
        to initiate a read operation from this data source.

        Returns:
            Any: Metadata object (type depends on relay implementation).
        """

    @abstractmethod
    async def wait_for_completion(self) -> None:
        """Wait for the remote read operation to complete.

        This method should block until the remote peer has finished reading
        the data.
        """
