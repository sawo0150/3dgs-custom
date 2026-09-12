"""Latency-first causal replay policies used by ERCB exp02.

The classes in this module deliberately share one ticket executor.  A draw is
only a *proposed* service; ``complete`` is called after an optimizer update was
actually applied.  This distinction is required for pair policies and for the
last-step behaviour of the upstream 3DGS loop.
"""
from __future__ import annotations

import math
import random
from collections import Counter, deque

import numpy as np


def _soft_choice(rng, items, scores, temperature):
    if len(items) == 1:
        return items[0]
    tau = max(float(temperature), 1e-6)
    maximum = max(scores)
    weights = [math.exp(max(-50.0, min(50.0, (s - maximum) / tau))) for s in scores]
    return rng.choices(items, weights=weights, k=1)[0]


class TicketedArchiveBase:
    """Common direct-service tickets plus a persistent archive RR path.

    ``ticket_fraction`` is implemented as a deterministic token bucket.  An
    overdue oldest ticket preempts the bucket.  Tickets are retired only by
    ``complete``; drawing or computing a gradient is not completion.
    """

    def __init__(self, seed=0, block_size=8, ticket_fraction=.75,
                 deadline_steps=32, temperature=.25, candidate_size=24):
        self.rng = random.Random(seed)
        self.block_size = max(1, int(block_size))
        self.ticket_fraction = float(ticket_fraction)
        self.deadline_steps = max(1, int(deadline_steps))
        self.temperature = float(temperature)
        self.candidate_size = max(2, int(candidate_size))
        if not 0.0 <= self.ticket_fraction <= 1.0:
            raise ValueError("ticket_fraction must be in [0,1]")
        self.active, self.rr_remaining = [], []
        self.counts, self.completed_counts = Counter(), Counter()
        self.pending, self.pending_set = deque(), set()
        self.arrival_step, self.first_service_step = {}, {}
        self.features, self.loss_ema, self.sketches = {}, {}, {}
        self.step = 0
        self.ticket_credit = 0.0
        self.last_draw_was_ticket = False
        self.last_draw_role = "archive"
        self.unapplied_draws = 0

    def set_features(self, features):
        self.features = {int(k): np.asarray(v, dtype=np.float64) for k, v in features.items()}

    def add(self, ids):
        self.step += 1
        # Accumulate fractional service rate while debt exists.  Capping at one
        # before subtraction aliases every fraction in (0.5, 1] to 0.5.
        # When there is no debt, keep at most one token so idle time cannot bank
        # an arbitrarily large future burst.
        if self.pending_set or ids:
            self.ticket_credit += self.ticket_fraction
        else:
            self.ticket_credit = min(1.0, self.ticket_credit + self.ticket_fraction)
        for item in ids:
            item = int(item)
            self.active.append(item)
            self.counts[item] += 0
            self.completed_counts[item] += 0
            self.arrival_step[item] = self.step
            self.pending.append(item)
            self.pending_set.add(item)
            self.rr_remaining.insert(self.rng.randrange(len(self.rr_remaining) + 1), item)

    def _ticket_due(self):
        if not self.pending:
            return False
        oldest = self.pending[0]
        return self.step - self.arrival_step[oldest] >= self.deadline_steps - 1

    def _take_ticket(self):
        while self.pending and self.pending[0] not in self.pending_set:
            self.pending.popleft()
        if not self.pending:
            return None
        item = self.pending[0]
        if self.ticket_credit >= 1.0:
            self.ticket_credit -= 1.0
        return item

    def _rr_candidates(self, exclude=()):
        excluded = set(exclude)
        if not self.rr_remaining:
            self.rr_remaining = self.active.copy()
            self.rng.shuffle(self.rr_remaining)
        candidates = []
        for item in reversed(self.rr_remaining):
            if item not in excluded:
                candidates.append(item)
                if len(candidates) >= self.candidate_size:
                    break
        if not candidates:
            candidates = [item for item in self.active if item not in excluded]
        return candidates

    def _consume_rr(self, item):
        if item in self.rr_remaining:
            self.rr_remaining.remove(item)

    def _draw_archive(self):
        candidates = self._rr_candidates()
        item = candidates[0]
        self._consume_rr(item)
        return item

    def _should_ticket(self):
        return bool(self.pending_set) and (self._ticket_due() or self.ticket_credit >= 1.0)

    def draw(self):
        if self._should_ticket():
            item = self._take_ticket()
            role = "ticket"
        else:
            item = self._draw_archive()
            role = "archive"
        self.counts[item] += 1
        self.last_draw_was_ticket = role == "ticket"
        self.last_draw_role = role
        return item

    def observe(self, item, photometric_loss):
        value = max(float(photometric_loss), 1e-12)
        old = self.loss_ema.get(int(item))
        self.loss_ema[int(item)] = value if old is None else .9 * old + .1 * value

    def observe_sketch(self, item, sketch):
        self.sketches[int(item)] = np.asarray(sketch, dtype=np.float64)

    def complete(self, items, iteration, applied=True):
        if not applied:
            self.unapplied_draws += len(items)
            return
        for item in items:
            item = int(item)
            self.completed_counts[item] += 1
            if item not in self.first_service_step:
                self.first_service_step[item] = int(iteration)
            if item in self.pending_set:
                self.pending_set.remove(item)
        while self.pending and self.pending[0] not in self.pending_set:
            self.pending.popleft()


class TicketArchiveRR(TicketedArchiveBase):
    """Common ticket executor with unmodified archive random reshuffling."""


class TicketMomentCompensation(TicketedArchiveBase):
    """Dense proposal 1: ticketed fresh views plus moment compensation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = None
        self.prefix_position = 0

    def _target_and_prefix(self):
        available = [i for i in self.active if i in self.features]
        if not available:
            return None
        target = np.mean([self.features[i] for i in available], axis=0)
        if self.prefix is None or self.prefix_position >= self.block_size:
            self.prefix = np.zeros_like(target)
            self.prefix_position = 0
        return target

    def _record_moment(self, item, target=None):
        if target is None:
            target = self._target_and_prefix()
        if target is not None:
            self.prefix += self.features[item] - target
            self.prefix_position += 1

    def _draw_moment_archive(self, exclude=()):
        candidates = self._rr_candidates(exclude=exclude)
        target = self._target_and_prefix()
        if target is None:
            item = candidates[0]
            self._consume_rr(item)
            return item
        scores = []
        for item in candidates:
            delta = self.features[item] - target
            # Negative incremental prefix energy; higher is better.
            scores.append(-float(2 * np.dot(self.prefix, delta) + np.dot(delta, delta)))
        item = _soft_choice(self.rng, candidates, scores, self.temperature)
        self._record_moment(item, target)
        self._consume_rr(item)
        return item

    def _draw_archive(self):
        return self._draw_moment_archive()

    def draw(self):
        if self._should_ticket():
            item = self._take_ticket()
            role = "ticket"
            # The mandatory fresh view is part of the same quadrature prefix;
            # archive choices must compensate the bias it actually introduced.
            self._record_moment(item)
        else:
            item = self._draw_archive()
            role = "archive"
        self.counts[item] += 1
        self.last_draw_was_ticket = role == "ticket"
        self.last_draw_role = role
        return item


class TicketServiceField(TicketedArchiveBase):
    """Dense proposal 2: local cross-view effective-service field.

    A sparse causal k-neighbour feature graph is evaluated only on the small RR
    shortlist.  Indirect credit never retires a direct-service ticket.
    """

    def _draw_archive(self):
        candidates = self._rr_candidates()
        known = [i for i in self.active if i in self.features]
        if not known:
            return super()._draw_archive()
        matrix = np.stack([self.features[i] for i in known])
        scale = float(np.median(np.linalg.norm(matrix[1:] - matrix[:-1], axis=1))) if len(known) > 1 else 1.0
        scale = max(scale, 1e-6)
        scores = []
        for item in candidates:
            distance = np.linalg.norm(matrix - self.features[item], axis=1)
            neighbours = np.argsort(distance)[:min(8, len(known))]
            effective = 0.0
            gain = 0.0
            for index in neighbours:
                other = known[int(index)]
                kernel = math.exp(-float(distance[index] ** 2) / (2 * scale * scale))
                service = self.completed_counts[other]
                effective += kernel * service
                gain += kernel * math.exp(-service)
            scores.append(gain / (1.0 + effective))
        item = _soft_choice(self.rng, candidates, scores, self.temperature)
        self._consume_rr(item)
        return item


class TicketDebtUtility(TicketedArchiveBase):
    """Candidate A: service debt plus signed-sketch marginal utility."""

    def _draw_archive(self):
        candidates = self._rr_candidates()
        sketches = [self.sketches[i] for i in self.active if i in self.sketches]
        reference = np.mean(sketches, axis=0) if sketches else None
        median_loss = float(np.median(list(self.loss_ema.values()))) if self.loss_ema else 1.0
        scores = []
        for item in candidates:
            debt = 1.0 if item in self.pending_set else 0.0
            utility = self.loss_ema.get(item, median_loss)
            if reference is not None and item in self.sketches:
                z = self.sketches[item]
                utility *= max(0.0, float(np.dot(reference, z))) / (1e-8 + float(np.dot(z, z)) ** .5)
            scores.append(debt + utility)
        item = _soft_choice(self.rng, candidates, scores, self.temperature)
        self._consume_rr(item)
        return item


class TicketPrefixBalance(TicketedArchiveBase):
    """Candidate B: deadline-feasible short-block cached-sketch balancing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = []

    def _build_block(self):
        size = min(self.block_size, len(self.active))
        pending = [item for item in self.pending if item in self.pending_set]
        ticket_quota = min(len(pending), int(math.ceil(self.ticket_fraction * size)))
        forced = pending[:ticket_quota]
        for item in self.pending:
            if item in self.pending_set and self.step - self.arrival_step[item] >= self.deadline_steps - size:
                if item not in forced:
                    forced.append(item)
                if len(forced) >= size:
                    break
        forced = forced[:size]
        archive_candidates = self._rr_candidates(exclude=forced)
        candidates = list(dict.fromkeys(forced + archive_candidates))[:max(size, self.candidate_size)]
        known = [self.sketches[i] for i in self.active if i in self.sketches]
        if not known:
            chosen = forced + [item for item in candidates if item not in forced][:size - len(forced)]
        else:
            target = np.mean(known, axis=0)
            residual = np.zeros_like(target)
            chosen = []
            remaining = candidates.copy()
            forced_remaining = set(forced)
            while remaining and len(chosen) < size:
                slots_left = size - len(chosen)
                eligible = ([item for item in remaining if item in forced_remaining]
                            if len(forced_remaining) >= slots_left else remaining)
                scored = []
                for item in eligible:
                    z = self.sketches.get(item, target)
                    delta = z - target
                    scored.append(-float(np.dot(residual + delta, residual + delta)))
                item = _soft_choice(self.rng, eligible, scored, self.temperature)
                chosen.append(item)
                residual += self.sketches.get(item, target) - target
                remaining.remove(item)
                forced_remaining.discard(item)
        for item in chosen:
            self._consume_rr(item)
        self.block = list(reversed(chosen))

    def draw(self):
        if not self.block:
            self._build_block()
        item = self.block.pop()
        self.counts[item] += 1
        self.last_draw_was_ticket = item in self.pending_set
        self.last_draw_role = "ticket" if self.last_draw_was_ticket else "archive"
        return item


class TicketApproxMIR(TicketedArchiveBase):
    """Candidate C: cached signed-sketch interference replay."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_fresh_sketch = None

    def observe_sketch(self, item, sketch):
        super().observe_sketch(item, sketch)
        if self.last_draw_was_ticket:
            self.last_fresh_sketch = np.asarray(sketch, dtype=np.float64)

    def _draw_archive(self):
        candidates = self._rr_candidates()
        if self.last_fresh_sketch is None:
            return super()._draw_archive()
        scores = []
        for item in candidates:
            z = self.sketches.get(item)
            scores.append(0.0 if z is None else max(0.0, -float(np.dot(z, self.last_fresh_sketch))))
        # Keep 25% global RR exploration so conflict replay cannot monopolize.
        if self.rng.random() < .25:
            item = candidates[0]
        else:
            item = _soft_choice(self.rng, candidates, scores, self.temperature)
        self._consume_rr(item)
        return item


class TicketPairBase(TicketMomentCompensation):
    """Alternates a fresh/ticket draw and a compensating archive draw."""

    pair_policy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_phase = 0
        self.pair_first_item = None

    def draw(self):
        if self.pair_phase == 0:
            if self.pending_set:
                item, role = self._take_ticket(), "pair_new"
                self._record_moment(item)
            else:
                item, role = super()._draw_archive(), "pair_new_archive"
            self.pair_first_item = item
            self.pair_phase = 1
        else:
            excluded = (self.pair_first_item,) if len(self.active) > 1 else ()
            item = self._draw_moment_archive(exclude=excluded)
            role = "pair_old"
            self.pair_phase = 0
        self.counts[item] += 1
        self.last_draw_was_ticket = role == "pair_new" 
        self.last_draw_role = role
        return item


class TicketPairMean(TicketPairBase):
    """Dense proposal 3: reference-descent optimized fresh/old mixture."""

    pair_mix_rule = "mean_descent"


class TicketPairSafe(TicketPairBase):
    """Candidate D: two-loss first-order non-worsening safe mixture."""

    pair_mix_rule = "safe_interval"


LATENCY_SCHEDULERS = {
    "ticket_archive_rr": TicketArchiveRR,
    "ticket_moment_compensation": TicketMomentCompensation,
    "ticket_service_field": TicketServiceField,
    "ticket_debt_utility": TicketDebtUtility,
    "ticket_prefix_balance": TicketPrefixBalance,
    "ticket_approx_mir": TicketApproxMIR,
    "ticket_pair_mean": TicketPairMean,
    "ticket_pair_safe": TicketPairSafe,
}


def make_latency_scheduler(name, seed=0, block_size=8, ticket_fraction=.75,
                           deadline_steps=32, temperature=.25,
                           candidate_size=24):
    return LATENCY_SCHEDULERS[name](
        seed=seed, block_size=block_size, ticket_fraction=ticket_fraction,
        deadline_steps=deadline_steps, temperature=temperature,
        candidate_size=candidate_size,
    )
