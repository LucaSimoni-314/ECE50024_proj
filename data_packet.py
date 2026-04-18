from typing import TypedDict
import numpy as np

class DataPacket(TypedDict):
    """
    Represents the summarized results of a single worker batch.
    Optimized for IPC (Inter-Process Communication) efficiency.
    """
    optimizer_name: str  # Identifier for the algorithm
    alpha: float         # Scaling parameter used in this batch
    c_val: float         # Sigmoid threshold used in this batch
    sum_error: float     # Total accumulated error across all trials in batch
    sum_time: int        # Total accumulated nanoseconds (prevents float drift)
    n: int               # The exact number of trials completed in this batch