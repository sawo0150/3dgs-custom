from __future__ import annotations

import os
from typing import Dict


def get_memory_stats() -> Dict[str, float]:
    stats = {
        "cpu_rss_mb": 0.0,
        "gpu_allocated_mb": 0.0,
        "gpu_reserved_mb": 0.0,
    }

    try:
        import psutil
        process = psutil.Process(os.getpid())
        stats["cpu_rss_mb"] = process.memory_info().rss / (1024 ** 2)
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
    except Exception:
        pass

    return stats
