# trion_core/utils/sampling_stats.py

import time
from collections import defaultdict

class SamplingStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = defaultdict(int)
        self.time_ms = defaultdict(float)
        self.records = []

    def log_step(self, *,
                 path,
                 entropy,
                 veff,
                 step_time_ms,
                 sampling_time_ms):
        self.count[path] += 1
        self.time_ms[path] += sampling_time_ms

        self.records.append({
            "path": path,
            "entropy": float(entropy),
            "veff": float(veff),
            "step_time_ms": step_time_ms,
            "sampling_time_ms": sampling_time_ms,
        })

    def summary(self):
        total = sum(self.count.values())
        out = {}

        for k in self.count:
            out[k] = {
                "count": self.count[k],
                "ratio": self.count[k] / max(total, 1),
                "avg_sampling_ms": self.time_ms[k] / max(self.count[k], 1),
            }

        return out


GLOBAL_SAMPLING_STATS = SamplingStats()
