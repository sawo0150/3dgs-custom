"""Bounded reorder of causal RR tickets, without adding replay tickets."""
import math
import random
from collections import Counter
from runtime.scheduler import CausalRandomReshuffling


class WindowERCB:
    def __init__(self, seed=0, gamma=math.log(3), window=32, force_reorder=False):
        if gamma < 0 or window < 1:
            raise ValueError("gamma>=0 and window>=1 required")
        self.base = CausalRandomReshuffling(seed)
        self.rng = random.Random(seed ^ 0xE7CB)
        self.gamma, self.window = gamma, window
        self.force_reorder = force_reorder
        self.active = self.base.active
        self.counts, self.pending = Counter(), []

    def add(self, ids):
        ids = list(ids)
        self.base.add(ids)
        self.counts.update({i: 0 for i in ids})

    def draw(self):
        if self.gamma == 0 and not self.force_reorder:
            item = self.base.draw()
        else:
            if not self.pending:
                # Never cross the current RR epoch boundary when reserving.
                n = min(self.window, len(self.base.remaining) or len(self.active))
                tickets = [self.base.draw() for _ in range(n)]
                mean = sum(self.counts.values()) / len(self.active)
                target = mean * .5
                ranked = []
                for item in tickets:
                    deficit = max(0., 1-self.counts[item]/target) if target else 0.
                    u = max(self.rng.random(), 1e-15)
                    ranked.append((self.gamma*deficit-math.log(-math.log(u)), item))
                self.pending = [item for _, item in sorted(ranked)]
            item = self.pending.pop()
        self.counts[item] += 1
        return item
