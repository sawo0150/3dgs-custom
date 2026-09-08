"""View schedulers for causal replay over a monotonically growing pool."""
from __future__ import annotations
import json, math, random
from collections import Counter
from pathlib import Path

class CausalRandomReshuffling:
    """Random permutation; arrivals enter a random unfinished-epoch slot."""
    def __init__(self, seed=0):
        self.rng, self.active, self.remaining, self.counts = random.Random(seed), [], [], Counter()
    def add(self, ids):
        for item in ids:
            self.active.append(item)
            self.remaining.insert(self.rng.randrange(len(self.remaining) + 1), item)
    def draw(self):
        if not self.remaining:
            self.remaining = self.active.copy(); self.rng.shuffle(self.remaining)
        item = self.remaining.pop(); self.counts[item] += 1
        return item

class CountBalancedRandomReshuffling:
    """Uniform draw among least-trained views; static-pool behavior is exact RR."""
    def __init__(self, seed=0):
        self.rng, self.active, self.counts = random.Random(seed), [], Counter()
    def add(self, ids):
        for item in ids: self.active.append(item); self.counts[item] += 0
    def draw(self):
        minimum = min(self.counts[i] for i in self.active)
        feasible = [i for i in self.active if self.counts[i] == minimum]
        item = self.rng.choice(feasible); self.counts[item] += 1
        return item

class SoftCountBalancedSampler:
    """Entropy/fairness interpolation: p(i) proportional to exp(-beta*n_i)."""
    def __init__(self, seed=0, beta=1.0):
        self.rng, self.beta, self.active, self.counts = random.Random(seed), beta, [], Counter()
    def add(self, ids):
        for item in ids: self.active.append(item); self.counts[item] += 0
    def draw(self):
        minimum = min(self.counts[i] for i in self.active)
        weights = [math.exp(-self.beta * (self.counts[i] - minimum)) for i in self.active]
        item = self.rng.choices(self.active, weights=weights, k=1)[0]; self.counts[item] += 1
        return item

class FloorProtectedRandomReshuffling(CausalRandomReshuffling):
    """Causal RR with a minimum raw-count floor relative to active mean."""
    def __init__(self, seed=0, floor_ratio=0.25):
        super().__init__(seed); self.floor_ratio = floor_ratio
    def draw(self):
        mean = sum(self.counts[i] for i in self.active) / len(self.active)
        deficit = [i for i in self.active if self.counts[i] < self.floor_ratio * mean]
        if deficit:
            item = self.rng.choice(deficit)
            if item in self.remaining: self.remaining.remove(item)
            self.counts[item] += 1
            return item
        return super().draw()

class LagBoundedRandomReshuffling(CausalRandomReshuffling):
    """RR subject to a maximum raw-count lag L whenever the lag is feasible.

    Arrivals may enter more than L behind the oldest views; in that case the
    minimum-count cohort is uniformly serviced until the invariant is restored.
    For a static pool and L>=1 this retains exact random reshuffling.
    """
    def __init__(self, seed=0, max_lag=8):
        super().__init__(seed); self.max_lag = int(max_lag)
        if self.max_lag < 1: raise ValueError("max_lag must be >= 1")
    def draw(self):
        minimum = min(self.counts[i] for i in self.active)
        maximum = max(self.counts[i] for i in self.active)
        if maximum - minimum > self.max_lag:
            feasible = [i for i in self.active if self.counts[i] == minimum]
        else:
            if not self.remaining:
                self.remaining = self.active.copy(); self.rng.shuffle(self.remaining)
            feasible = [i for i in self.remaining if self.counts[i] + 1 - minimum <= self.max_lag]
            if not feasible:
                feasible = [i for i in self.active if self.counts[i] == minimum]
        item = self.rng.choice(feasible)
        if item in self.remaining: self.remaining.remove(item)
        self.counts[item] += 1
        return item

class MixedDeficitRandomReshuffling(CausalRandomReshuffling):
    """With probability rho service the minimum-count cohort, otherwise RR."""
    def __init__(self, seed=0, rho=.1):
        super().__init__(seed); self.rho = float(rho)
        if not 0 <= self.rho <= 1: raise ValueError("rho must be in [0,1]")
    def draw(self):
        if self.rng.random() < self.rho:
            minimum = min(self.counts[i] for i in self.active)
            item = self.rng.choice([i for i in self.active if self.counts[i] == minimum])
            if item in self.remaining: self.remaining.remove(item)
            self.counts[item] += 1
            return item
        return super().draw()

class BlockWeightedRandomReshuffling:
    """Plackett-Luce weighted sampling without replacement inside fixed blocks.

    Each block contains at most K distinct views drawn from the entire causal
    pool.  beta=0 is a uniform random K-permutation; beta>0 gently favors views
    with fewer lifetime selections while preserving zero duplicates per block.
    Arrivals during a block become eligible at the next block boundary.
    """
    def __init__(self, seed=0, beta=.02, block_size=128):
        self.rng, self.beta, self.block_size = random.Random(seed), float(beta), int(block_size)
        if self.beta < 0 or self.block_size < 1: raise ValueError("beta>=0 and block_size>=1 required")
        self.active, self.remaining, self.counts = [], [], Counter()
    def add(self, ids):
        for item in ids: self.active.append(item); self.counts[item] += 0
    def _new_block(self):
        minimum=min(self.counts[i] for i in self.active)
        # Gumbel top-k is an exact Plackett-Luce ordered sample without replacement.
        ranked=[]
        for item in self.active:
            u=max(self.rng.random(),1e-15); g=-math.log(-math.log(u))
            ranked.append((-self.beta*(self.counts[item]-minimum)+g,item))
        ranked.sort(reverse=True); chosen=[item for _,item in ranked[:min(self.block_size,len(ranked))]]
        self.remaining=list(reversed(chosen))
    def draw(self):
        if not self.remaining: self._new_block()
        item=self.remaining.pop(); self.counts[item]+=1; return item

class StablePoolBlockWeightedRandomReshuffling:
    """Causal RR during growth, then count-weighted block RR on a stable pool.

    The scheduler follows CausalRandomReshuffling exactly while topology can
    still densify or arrivals are recent. It switches once ``phase_start`` steps
    elapsed and no view arrived for ``block_size`` consecutive steps. The switch
    is causal and permanent; later arrivals enter the next weighted block.
    """
    def __init__(self, seed=0, beta=.02, block_size=128, phase_start=15000):
        self.rng, self.beta, self.block_size = random.Random(seed), float(beta), int(block_size)
        self.phase_start = int(phase_start)
        if self.beta < 0 or self.block_size < 1 or self.phase_start < 0:
            raise ValueError("beta>=0, block_size>=1, phase_start>=0 required")
        self.active, self.remaining, self.counts = [], [], Counter()
        self.steps, self.idle_steps, self.weighted_phase = 0, 0, False
        self.weighted_phase_start = None
    def add(self, ids):
        self.steps += 1
        ids = list(ids)
        self.idle_steps = 0 if ids else self.idle_steps + 1
        for item in ids:
            self.active.append(item); self.counts[item] += 0
            if not self.weighted_phase:
                self.remaining.insert(self.rng.randrange(len(self.remaining) + 1), item)
        if (not self.weighted_phase and self.steps >= self.phase_start
                and self.idle_steps >= self.block_size):
            self.weighted_phase = True
            self.weighted_phase_start = self.steps
            self.remaining = []
    def _new_weighted_block(self):
        minimum = min(self.counts[i] for i in self.active)
        ranked = []
        for item in self.active:
            u = max(self.rng.random(), 1e-15)
            g = -math.log(-math.log(u))
            ranked.append((-self.beta * (self.counts[item] - minimum) + g, item))
        ranked.sort(reverse=True)
        chosen = [item for _, item in ranked[:min(self.block_size, len(ranked))]]
        self.remaining = list(reversed(chosen))
    def draw(self):
        if not self.remaining:
            if self.weighted_phase:
                self._new_weighted_block()
            else:
                self.remaining = self.active.copy(); self.rng.shuffle(self.remaining)
        item = self.remaining.pop(); self.counts[item] += 1
        return item

class EntropyFloorRandomReshuffling:
    """RR coverage floor plus randomized deficit service in unique K-blocks.

    At each virtual block position, probability ``rho`` is assigned uniformly
    to the current minimum-count cohort and the remaining probability follows a
    persistent uniform RR queue. Deficit draws do not consume that RR queue, so
    under-served views can catch up across blocks. Already chosen block members
    are excluded, giving zero within-block duplicates. ``rho=0`` recovers the
    persistent RR coverage law, modulo arrivals waiting for the next boundary.
    """
    def __init__(self, seed=0, rho=.25, block_size=128):
        self.rng, self.rho, self.block_size = random.Random(seed), float(rho), int(block_size)
        if not 0 <= self.rho <= 1 or self.block_size < 1:
            raise ValueError("rho in [0,1] and block_size>=1 required")
        self.active, self.rr_remaining, self.remaining, self.counts = [], [], [], Counter()
    def add(self, ids):
        for item in ids:
            self.active.append(item); self.counts[item] += 0
            self.rr_remaining.insert(self.rng.randrange(len(self.rr_remaining) + 1), item)
    def _uniform_candidate(self, excluded):
        eligible = [item for item in self.rr_remaining if item not in excluded]
        if not eligible:
            refill = self.active.copy(); self.rng.shuffle(refill)
            self.rr_remaining.extend(refill)
            eligible = [item for item in self.rr_remaining if item not in excluded]
        item = self.rng.choice(eligible)
        self.rr_remaining.remove(item)
        return item
    def _new_block(self):
        size = min(self.block_size, len(self.active))
        selected, excluded = [], set()
        virtual = {item: self.counts[item] for item in self.active}
        for _ in range(size):
            eligible = [item for item in self.active if item not in excluded]
            if self.rng.random() < self.rho:
                minimum = min(virtual[item] for item in eligible)
                item = self.rng.choice([item for item in eligible if virtual[item] == minimum])
            else:
                item = self._uniform_candidate(excluded)
            selected.append(item); excluded.add(item); virtual[item] += 1
        self.remaining = list(reversed(selected))
    def draw(self):
        if not self.remaining: self._new_block()
        item = self.remaining.pop(); self.counts[item] += 1
        return item

class IntervalSoftmaxRandomReshuffling:
    """Hierarchical KF-interval softmax sampling without replacement.

    Every non-empty ``add`` call defines one causal keyframe interval.  The
    outer sampler draws at most ``block_size`` distinct intervals with
    Plackett--Luce weights ``exp(-beta * interval_service_count)``.  The inner
    sampler maintains an independent persistent random-reshuffling queue over
    the frames belonging to the selected interval.

    Consequently, an outer block cannot collapse onto one temporal interval,
    while the softmax count has the directly auditable meaning "total optimizer
    updates assigned to this KF interval".  Arrivals during an unfinished outer
    block become eligible at the next block boundary.
    """
    def __init__(self, seed=0, beta=.02, block_size=16):
        self.rng, self.beta, self.block_size = random.Random(seed), float(beta), int(block_size)
        if self.beta < 0 or self.block_size < 1:
            raise ValueError("beta>=0 and block_size>=1 required")
        self.active, self.remaining, self.counts = [], [], Counter()
        self.intervals, self.interval_counts = [], Counter()
        self.frame_remaining, self.frame_to_interval = {}, {}
    def add(self, ids):
        ids = list(ids)
        if not ids:
            return
        interval_id = len(self.intervals)
        self.intervals.append(ids)
        self.frame_remaining[interval_id] = []
        self.interval_counts[interval_id] += 0
        for item in ids:
            self.active.append(item)
            self.counts[item] += 0
            self.frame_to_interval[item] = interval_id
    def _new_interval_block(self):
        minimum = min(self.interval_counts.values())
        ranked = []
        for interval_id in range(len(self.intervals)):
            u = max(self.rng.random(), 1e-15)
            gumbel = -math.log(-math.log(u))
            score = -self.beta * (self.interval_counts[interval_id] - minimum) + gumbel
            ranked.append((score, interval_id))
        ranked.sort(reverse=True)
        chosen = [interval_id for _, interval_id in ranked[:min(self.block_size, len(ranked))]]
        self.remaining = list(reversed(chosen))
    def _draw_frame(self, interval_id):
        remaining = self.frame_remaining[interval_id]
        if not remaining:
            remaining.extend(self.intervals[interval_id])
            self.rng.shuffle(remaining)
        return remaining.pop()
    def draw(self):
        if not self.remaining:
            self._new_interval_block()
        interval_id = self.remaining.pop()
        item = self._draw_frame(interval_id)
        self.interval_counts[interval_id] += 1
        self.counts[item] += 1
        return item

class SizeAwareIntervalSoftmaxRandomReshuffling(IntervalSoftmaxRandomReshuffling):
    """Interval-level no-replacement mixing with a frame-uniform base measure.

    Intervals may contain different numbers of frames.  Uniform interval quota
    can therefore oversample every frame in a short interval.  This variant
    uses ``|G_j|`` as the beta=0 base weight and applies count correction to the
    mean service per member, ``c_j / |G_j|``:

        log w_j = log |G_j| - beta * (c_j/|G_j| - min_k c_k/|G_k|).

    The outer interval block is still without replacement and the inner frame
    sampler is still persistent RR.
    """
    def _new_interval_block(self):
        rates = {
            interval_id: self.interval_counts[interval_id] / len(members)
            for interval_id, members in enumerate(self.intervals)
        }
        minimum = min(rates.values())
        ranked = []
        for interval_id, members in enumerate(self.intervals):
            u = max(self.rng.random(), 1e-15)
            gumbel = -math.log(-math.log(u))
            score = math.log(len(members)) - self.beta * (rates[interval_id] - minimum) + gumbel
            ranked.append((score, interval_id))
        ranked.sort(reverse=True)
        chosen = [interval_id for _, interval_id in ranked[:min(self.block_size, len(ranked))]]
        self.remaining = list(reversed(chosen))

class MeanNormalizedSizeAwareIntervalSoftmaxRandomReshuffling(
        SizeAwareIntervalSoftmaxRandomReshuffling):
    """Scale-free exposure softmax over interval random reshuffling.

    Let ``r_j = c_j / |G_j|`` be service per member and let ``mu`` be the
    current frame-weighted mean service.  The interval law is

        w_j = |G_j| exp(-gamma * r_j / mu).

    This is the mean-normalized entropy-regularized count-balancing law:
    subtracting the common mean inside the exponent would cancel during
    normalization, while division by ``mu`` makes ``gamma`` dimensionless and
    invariant to a common rescaling of all service counts.  There is no floor,
    clipping, or phase switch.  At zero total service the law is exactly the
    frame-uniform base measure.  Outer intervals and inner frames remain
    without replacement.
    """

    def __init__(self, seed=0, gamma=math.log(2.0), block_size=8):
        super().__init__(seed, gamma, block_size)
        self.gamma = self.beta

    def _new_interval_block(self):
        frame_count = sum(len(members) for members in self.intervals)
        mean_service = sum(self.interval_counts.values()) / frame_count
        ranked = []
        for interval_id, members in enumerate(self.intervals):
            rate = self.interval_counts[interval_id] / len(members)
            normalized_rate = 0.0 if mean_service <= 1e-12 else rate / mean_service
            u = max(self.rng.random(), 1e-15)
            gumbel = -math.log(-math.log(u))
            score = math.log(len(members)) - self.gamma * normalized_rate + gumbel
            ranked.append((score, interval_id))
        ranked.sort(reverse=True)
        chosen = [interval_id for _, interval_id in ranked[:min(self.block_size, len(ranked))]]
        self.remaining = list(reversed(chosen))

class LossAwareSizeAwareIntervalSoftmaxRandomReshuffling(SizeAwareIntervalSoftmaxRandomReshuffling):
    """Count-balanced interval RR calibrated by observed photometric residual.

    Lifetime count alone treats every under-served interval as equally useful.
    In an incremental map that can waste updates on late views whose remaining
    error cannot be repaired by appearance replay.  This scheduler retains the
    size-aware count softmax but multiplies its weight by a scale-free EMA-loss
    factor::

        w_j = |G_j| exp[-beta (c_j/|G_j| - r_min)] loss_j**loss_alpha.

    Unknown intervals use the median observed loss, so the utility term is
    neutral until the interval has actually been sampled.  The outer block is
    still Plackett--Luce without replacement and the inner frame queue is still
    persistent random reshuffling.
    """
    def __init__(self, seed=0, beta=.02, block_size=16, loss_alpha=.5, loss_decay=.9):
        super().__init__(seed, beta, block_size)
        self.loss_alpha = float(loss_alpha)
        self.loss_decay = float(loss_decay)
        if self.loss_alpha < 0 or not 0 <= self.loss_decay < 1:
            raise ValueError("loss_alpha>=0 and loss_decay in [0,1) required")
        self.interval_loss_ema = {}

    def observe(self, item, photometric_loss):
        interval_id = self.frame_to_interval[item]
        value = max(float(photometric_loss), 1e-12)
        previous = self.interval_loss_ema.get(interval_id)
        self.interval_loss_ema[interval_id] = (
            value if previous is None
            else self.loss_decay * previous + (1.0 - self.loss_decay) * value
        )

    def _new_interval_block(self):
        rates = {
            interval_id: self.interval_counts[interval_id] / len(members)
            for interval_id, members in enumerate(self.intervals)
        }
        minimum = min(rates.values())
        observed = sorted(self.interval_loss_ema.values())
        neutral_loss = observed[len(observed) // 2] if observed else 1.0
        ranked = []
        for interval_id, members in enumerate(self.intervals):
            loss_estimate = self.interval_loss_ema.get(interval_id, neutral_loss)
            u = max(self.rng.random(), 1e-15)
            gumbel = -math.log(-math.log(u))
            score = (
                math.log(len(members))
                - self.beta * (rates[interval_id] - minimum)
                + self.loss_alpha * math.log(max(loss_estimate, 1e-12))
                + gumbel
            )
            ranked.append((score, interval_id))
        ranked.sort(reverse=True)
        chosen = [interval_id for _, interval_id in ranked[:min(self.block_size, len(ranked))]]
        self.remaining = list(reversed(chosen))

class BoundedSizeAwareIntervalSoftmaxRandomReshuffling(
        SizeAwareIntervalSoftmaxRandomReshuffling):
    """Scene-independent bounded count correction over interval RR.

    Raw lifetime counts grow with sequence length, so a fixed coefficient in
    ``exp(-beta * count)`` has different meanings across scenes and training
    horizons.  Normalize the per-member interval service rate to [0, 1] and
    interpret ``gamma`` as the log maximum correction odds::

        z_j = (r_j-r_min)/(r_max-r_min),
        w_j = |G_j| exp(-gamma*z_j).

    Thus the most under-served interval receives at most ``exp(gamma)`` times
    the count-correction odds of the most served interval, independent of the
    absolute count range.  A zero span exactly recovers the size-aware uniform
    base law.
    """
    def __init__(self, seed=0, gamma=math.log(2.0), block_size=16):
        super().__init__(seed, gamma, block_size)
        self.gamma = self.beta

    def _new_interval_block(self):
        rates = {
            interval_id: self.interval_counts[interval_id] / len(members)
            for interval_id, members in enumerate(self.intervals)
        }
        minimum, maximum = min(rates.values()), max(rates.values())
        span = maximum - minimum
        ranked = []
        for interval_id, members in enumerate(self.intervals):
            normalized = 0.0 if span <= 1e-12 else (rates[interval_id] - minimum) / span
            u = max(self.rng.random(), 1e-15)
            gumbel = -math.log(-math.log(u))
            score = math.log(len(members)) - self.gamma * normalized + gumbel
            ranked.append((score, interval_id))
        ranked.sort(reverse=True)
        chosen = [interval_id for _, interval_id in ranked[:min(self.block_size, len(ranked))]]
        self.remaining = list(reversed(chosen))

class TwoPassSizeAwareIntervalSoftmaxRandomReshuffling(
        SizeAwareIntervalSoftmaxRandomReshuffling):
    """Soft, non-blocking two-pass coverage followed by interval RR.

    The old maturity gate blocks admission until every eligible view has been
    selected twice.  Here the same two-update requirement is a bounded sampling
    preference instead of a barrier.  For interval ``j`` with per-member service
    rate ``r_j = c_j / |G_j|``::

        d_j = max(0, 1 - r_j / 2)
        w_j = |G_j| exp(gamma * d_j).

    An entirely unserved interval receives at most ``exp(gamma)`` times the
    correction odds, the bonus decreases linearly as its members complete two
    passes, and it is exactly zero thereafter.  Thus persistent lifetime-count
    equalization cannot keep stealing service after coverage has been repaired.
    The outer interval block and inner frame queues remain without replacement.
    """
    coverage_quota = 2.0

    def __init__(self, seed=0, gamma=math.log(2.0), block_size=16):
        super().__init__(seed, gamma, block_size)
        self.gamma = self.beta

    def _new_interval_block(self):
        ranked = []
        for interval_id, members in enumerate(self.intervals):
            rate = self.interval_counts[interval_id] / len(members)
            deficit = max(0.0, 1.0 - rate / self.coverage_quota)
            u = max(self.rng.random(), 1e-15)
            gumbel = -math.log(-math.log(u))
            score = math.log(len(members)) + self.gamma * deficit + gumbel
            ranked.append((score, interval_id))
        ranked.sort(reverse=True)
        chosen = [interval_id for _, interval_id in ranked[:min(self.block_size, len(ranked))]]
        self.remaining = list(reversed(chosen))

class RelativeFloorIntervalSoftmaxRandomReshuffling(
        SizeAwareIntervalSoftmaxRandomReshuffling):
    """Count softmax below a scale-free fraction of current mean service.

    A fixed pass count can become inactive exactly where a long stream remains
    imbalanced.  Let ``mu`` be total optimizer service divided by the current
    number of frames and let ``r_j`` be interval service per member.  This
    scheduler uses the single law::

        d_j = max(0, 1 - r_j / (mu/2))
        w_j = |G_j| exp(gamma * d_j).

    Only intervals below one half of the current frame-average service receive
    a bonus.  The maximum correction odds are ``exp(gamma)`` and all intervals
    at or above the relative floor retain the frame-uniform base weight.  The
    one-half floor is fixed, leaving ``gamma`` as the only correction parameter.
    """
    relative_floor_ratio = 0.5

    def __init__(self, seed=0, gamma=math.log(2.0), block_size=8):
        super().__init__(seed, gamma, block_size)
        self.gamma = self.beta

    def _new_interval_block(self):
        frame_count = sum(len(members) for members in self.intervals)
        mean_service = sum(self.interval_counts.values()) / frame_count
        target = self.relative_floor_ratio * mean_service
        ranked = []
        for interval_id, members in enumerate(self.intervals):
            rate = self.interval_counts[interval_id] / len(members)
            deficit = 0.0 if target <= 1e-12 else max(0.0, 1.0 - rate / target)
            u = max(self.rng.random(), 1e-15)
            gumbel = -math.log(-math.log(u))
            score = math.log(len(members)) + self.gamma * deficit + gumbel
            ranked.append((score, interval_id))
        ranked.sort(reverse=True)
        chosen = [interval_id for _, interval_id in ranked[:min(self.block_size, len(ranked))]]
        self.remaining = list(reversed(chosen))

class StagedSizeAwareIntervalSoftmaxRandomReshuffling(SizeAwareIntervalSoftmaxRandomReshuffling):
    """Use exact causal RR while topology grows, then interval correction.

    3DGS changes its Gaussian topology only before ``phase_start``.  During that
    phase this scheduler has exactly the same draw law (and, for the same seed,
    exactly the same draws) as :class:`CausalRandomReshuffling`.  Starting at
    ``phase_start`` it switches once to size-aware interval softmax RR.  All
    service received during the RR phase remains in the interval/frame counts,
    so the second phase corrects the exposure inherited from topology growth.
    """
    def __init__(self, seed=0, beta=.02, block_size=16, phase_start=15000):
        super().__init__(seed, beta, block_size)
        self.phase_start = int(phase_start)
        if self.phase_start < 0:
            raise ValueError("phase_start must be >= 0")
        self.steps, self.staged_phase, self.staged_phase_start = 0, False, None
        self.rr_remaining = []
    def add(self, ids):
        self.steps += 1
        ids = list(ids)
        super().add(ids)
        if not self.staged_phase:
            for item in ids:
                self.rr_remaining.insert(self.rng.randrange(len(self.rr_remaining) + 1), item)
            if self.steps >= self.phase_start:
                self.staged_phase = True
                self.staged_phase_start = self.steps
                self.remaining = []
    def draw(self):
        if self.staged_phase:
            return super().draw()
        if not self.rr_remaining:
            self.rr_remaining = self.active.copy()
            self.rng.shuffle(self.rr_remaining)
        item = self.rr_remaining.pop()
        self.counts[item] += 1
        self.interval_counts[self.frame_to_interval[item]] += 1
        return item

class StagedLossAwareSizeAwareIntervalSoftmaxRandomReshuffling(
        LossAwareSizeAwareIntervalSoftmaxRandomReshuffling):
    """Shared-checkpoint harness for loss-aware replay-only comparisons.

    This is not a proposed two-stage production policy.  It reconstructs the
    exact causal-RR service history that produced a shared topology checkpoint,
    then activates the loss-aware replay picker at ``phase_start``.  Residual
    EMAs deliberately start empty at the branch because the checkpoint does not
    contain historical per-interval losses.
    """
    def __init__(self, seed=0, beta=.02, block_size=16, phase_start=15000,
                 loss_alpha=.5, loss_decay=.9):
        super().__init__(seed, beta, block_size, loss_alpha, loss_decay)
        self.phase_start = int(phase_start)
        if self.phase_start < 0:
            raise ValueError("phase_start must be >= 0")
        self.steps, self.staged_phase, self.staged_phase_start = 0, False, None
        self.rr_remaining = []

    def add(self, ids):
        self.steps += 1
        ids = list(ids)
        super().add(ids)
        if not self.staged_phase:
            for item in ids:
                self.rr_remaining.insert(self.rng.randrange(len(self.rr_remaining) + 1), item)
            if self.steps >= self.phase_start:
                self.staged_phase = True
                self.staged_phase_start = self.steps
                self.remaining = []

    def draw(self):
        if self.staged_phase:
            return super().draw()
        if not self.rr_remaining:
            self.rr_remaining = self.active.copy()
            self.rng.shuffle(self.rr_remaining)
        item = self.rr_remaining.pop()
        self.counts[item] += 1
        self.interval_counts[self.frame_to_interval[item]] += 1
        return item

    def observe(self, item, photometric_loss):
        if self.staged_phase:
            super().observe(item, photometric_loss)

class StagedBoundedSizeAwareIntervalSoftmaxRandomReshuffling(
        BoundedSizeAwareIntervalSoftmaxRandomReshuffling):
    """Shared-checkpoint harness for bounded replay-only comparisons."""
    def __init__(self, seed=0, gamma=math.log(2.0), block_size=16,
                 phase_start=15000):
        super().__init__(seed, gamma, block_size)
        self.phase_start = int(phase_start)
        if self.phase_start < 0:
            raise ValueError("phase_start must be >= 0")
        self.steps, self.staged_phase, self.staged_phase_start = 0, False, None
        self.rr_remaining = []

    def add(self, ids):
        self.steps += 1
        ids = list(ids)
        super().add(ids)
        if not self.staged_phase:
            for item in ids:
                self.rr_remaining.insert(self.rng.randrange(len(self.rr_remaining) + 1), item)
            if self.steps >= self.phase_start:
                self.staged_phase = True
                self.staged_phase_start = self.steps
                self.remaining = []

    def draw(self):
        if self.staged_phase:
            return super().draw()
        if not self.rr_remaining:
            self.rr_remaining = self.active.copy()
            self.rng.shuffle(self.rr_remaining)
        item = self.rr_remaining.pop()
        self.counts[item] += 1
        self.interval_counts[self.frame_to_interval[item]] += 1
        return item

class StagedTwoPassSizeAwareIntervalSoftmaxRandomReshuffling(
        TwoPassSizeAwareIntervalSoftmaxRandomReshuffling):
    """Shared-checkpoint harness for two-pass replay-only comparisons."""
    def __init__(self, seed=0, gamma=math.log(2.0), block_size=16,
                 phase_start=15000):
        super().__init__(seed, gamma, block_size)
        self.phase_start = int(phase_start)
        if self.phase_start < 0:
            raise ValueError("phase_start must be >= 0")
        self.steps, self.staged_phase, self.staged_phase_start = 0, False, None
        self.rr_remaining = []

    def add(self, ids):
        self.steps += 1
        ids = list(ids)
        super().add(ids)
        if not self.staged_phase:
            for item in ids:
                self.rr_remaining.insert(self.rng.randrange(len(self.rr_remaining) + 1), item)
            if self.steps >= self.phase_start:
                self.staged_phase = True
                self.staged_phase_start = self.steps
                self.remaining = []

    def draw(self):
        if self.staged_phase:
            return super().draw()
        if not self.rr_remaining:
            self.rr_remaining = self.active.copy()
            self.rng.shuffle(self.rr_remaining)
        item = self.rr_remaining.pop()
        self.counts[item] += 1
        self.interval_counts[self.frame_to_interval[item]] += 1
        return item

def make_scheduler(name, seed=0, beta=1.0, block_size=128, phase_start=0,
                   loss_alpha=.5):
    return {"causal_rr": CausalRandomReshuffling,
            "count_balanced_rr": CountBalancedRandomReshuffling,
            "soft_count": lambda s: SoftCountBalancedSampler(s, beta),
            "floor_rr": lambda s: FloorProtectedRandomReshuffling(s, beta),
            "lag_rr": lambda s: LagBoundedRandomReshuffling(s, beta),
            "mixed_deficit_rr": lambda s: MixedDeficitRandomReshuffling(s, beta),
            "block_weighted_rr": lambda s: BlockWeightedRandomReshuffling(s, beta, block_size),
            "stable_pool_block_rr": lambda s: StablePoolBlockWeightedRandomReshuffling(s, beta, block_size, phase_start),
            "entropy_floor_rr": lambda s: EntropyFloorRandomReshuffling(s, beta, block_size),
            "interval_softmax_rr": lambda s: IntervalSoftmaxRandomReshuffling(s, beta, block_size),
            "interval_size_softmax_rr": lambda s: SizeAwareIntervalSoftmaxRandomReshuffling(s, beta, block_size),
            "normalized_interval_size_softmax_rr": lambda s: MeanNormalizedSizeAwareIntervalSoftmaxRandomReshuffling(s, beta, block_size),
            "staged_interval_size_softmax_rr": lambda s: StagedSizeAwareIntervalSoftmaxRandomReshuffling(s, beta, block_size, phase_start),
            "loss_interval_size_softmax_rr": lambda s: LossAwareSizeAwareIntervalSoftmaxRandomReshuffling(s, beta, block_size, loss_alpha),
            "staged_loss_interval_size_softmax_rr": lambda s: StagedLossAwareSizeAwareIntervalSoftmaxRandomReshuffling(s, beta, block_size, phase_start, loss_alpha),
            "bounded_interval_size_softmax_rr": lambda s: BoundedSizeAwareIntervalSoftmaxRandomReshuffling(s, beta, block_size),
            "staged_bounded_interval_size_softmax_rr": lambda s: StagedBoundedSizeAwareIntervalSoftmaxRandomReshuffling(s, beta, block_size, phase_start),
            "two_pass_interval_size_softmax_rr": lambda s: TwoPassSizeAwareIntervalSoftmaxRandomReshuffling(s, beta, block_size),
            "staged_two_pass_interval_size_softmax_rr": lambda s: StagedTwoPassSizeAwareIntervalSoftmaxRandomReshuffling(s, beta, block_size, phase_start),
            "relative_floor_interval_softmax_rr": lambda s: RelativeFloorIntervalSoftmaxRandomReshuffling(s, beta, block_size)}[name](seed)

def load_arrival_iterations(path, image_names):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_name = payload.get("arrival_iteration_by_name", payload)
    missing = [n for n in image_names if n not in by_name]
    if missing: raise ValueError(f"arrival schedule misses {len(missing)} images; first={missing[0]}")
    return [int(by_name[n]) for n in image_names]

def scheduler_summary(scheduler, image_names, arrivals, total):
    counts = [int(scheduler.counts[i]) for i in range(len(image_names))]
    payload = {"total_iterations": total, "pool_size": len(image_names),
               "arrival_iteration": dict(zip(image_names, arrivals)),
               "selection_count": dict(zip(image_names, counts)),
               "count_min": min(counts), "count_max": max(counts),
               "count_mean": sum(counts) / len(counts)}
    if hasattr(scheduler, "interval_counts"):
        payload["interval_selection_count"] = {
            str(interval_id): int(scheduler.interval_counts[interval_id])
            for interval_id in range(len(scheduler.intervals))
        }
        payload["frame_interval_id"] = {
            image_names[item]: int(scheduler.frame_to_interval[item])
            for item in range(len(image_names))
        }
        payload["interval_members"] = {
            str(interval_id): [image_names[item] for item in members]
            for interval_id, members in enumerate(scheduler.intervals)
        }
    if hasattr(scheduler, "loss_alpha"):
        payload["loss_alpha"] = float(scheduler.loss_alpha)
        payload["interval_loss_ema"] = {
            str(interval_id): float(value)
            for interval_id, value in scheduler.interval_loss_ema.items()
        }
    if hasattr(scheduler, "gamma"):
        payload["gamma"] = float(scheduler.gamma)
    if hasattr(scheduler, "relative_floor_ratio"):
        payload["relative_floor_ratio"] = float(scheduler.relative_floor_ratio)
    return payload
