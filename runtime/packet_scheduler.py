"""Exp77: packet replay with exactly paired legacy ERCB outer randomness."""
import random

from runtime.scheduler import RelativeFloorIntervalSoftmaxRandomReshuffling


class PacketIntervalRandomReshuffling(RelativeFloorIntervalSoftmaxRandomReshuffling):
    """Only the inner policy changes; each add() is a closed causal interval.

    A shadow legacy inner draw preserves the shared RNG stream used by the
    original outer sampler. This costs CPU work and must count as overhead.
    Packet randomness is separate. repeats=1 returns the exact legacy draw.
    No temporal stratification, eviction, admission gate, or extra updates.
    """
    def __init__(self, seed=0, beta=1.0986122886681098, block_size=8,
                 packet_size=4, repeats=2):
        super().__init__(seed, beta, block_size)
        if packet_size < 1 or repeats < 1:
            raise ValueError("packet_size and repeats must be positive")
        self.packet_size, self.repeats = int(packet_size), int(repeats)
        self.packet_rng = random.Random(seed ^ 0x77A11)
        self.packet_states = {}

    def _draw_frame(self, interval_id):
        legacy_item = super()._draw_frame(interval_id)
        if self.repeats == 1:
            return legacy_item
        state = self.packet_states.setdefault(interval_id, {
            "cycle": [], "packet": [], "pass": [], "passes_left": 0})
        if not state["pass"]:
            if state["passes_left"] == 0:
                if not state["cycle"]:
                    state["cycle"] = list(self.intervals[interval_id])
                    self.packet_rng.shuffle(state["cycle"])
                size = min(self.packet_size, len(state["cycle"]))
                state["packet"] = [state["cycle"].pop() for _ in range(size)]
                state["passes_left"] = self.repeats
            state["pass"] = list(state["packet"])
            self.packet_rng.shuffle(state["pass"])
            state["passes_left"] -= 1
        return state["pass"].pop()
