from runtime.scheduler import CausalRandomReshuffling, CountBalancedRandomReshuffling, FloorProtectedRandomReshuffling, LagBoundedRandomReshuffling, BlockWeightedRandomReshuffling, StablePoolBlockWeightedRandomReshuffling, EntropyFloorRandomReshuffling, IntervalSoftmaxRandomReshuffling, SizeAwareIntervalSoftmaxRandomReshuffling, MeanNormalizedSizeAwareIntervalSoftmaxRandomReshuffling, StagedSizeAwareIntervalSoftmaxRandomReshuffling, LossAwareSizeAwareIntervalSoftmaxRandomReshuffling, StagedLossAwareSizeAwareIntervalSoftmaxRandomReshuffling, BoundedSizeAwareIntervalSoftmaxRandomReshuffling, StagedBoundedSizeAwareIntervalSoftmaxRandomReshuffling, TwoPassSizeAwareIntervalSoftmaxRandomReshuffling, StagedTwoPassSizeAwareIntervalSoftmaxRandomReshuffling, RelativeFloorIntervalSoftmaxRandomReshuffling

def test_static_count_balanced_is_exact_reshuffling():
    s=CountBalancedRandomReshuffling(7); s.add(range(9)); draws=[s.draw() for _ in range(27)]
    assert all(sorted(draws[k:k+9])==list(range(9)) for k in range(0,27,9))

def test_growing_count_balanced_catches_up_when_capacity_exists():
    s=CountBalancedRandomReshuffling(3); s.add(range(4)); [s.draw() for _ in range(20)]; s.add(range(4,8)); [s.draw() for _ in range(36)]
    counts=[s.counts[i] for i in range(8)]; assert max(counts)-min(counts)<=1

def test_causal_rr_static_epoch_has_no_duplicate():
    s=CausalRandomReshuffling(2); s.add(range(11)); assert len({s.draw() for _ in range(11)})==11

def test_floor_rr_prevents_zero_service_after_arrival():
    s=FloorProtectedRandomReshuffling(4, .25); s.add(range(4)); [s.draw() for _ in range(40)]; s.add(range(4,8)); [s.draw() for _ in range(8)]
    assert min(s.counts[i] for i in range(4,8)) >= 1

def test_lag_rr_static_pool_is_exact_reshuffling():
    s=LagBoundedRandomReshuffling(9,8); s.add(range(13)); draws=[s.draw() for _ in range(39)]
    assert all(sorted(draws[k:k+13])==list(range(13)) for k in range(0,39,13))

def test_lag_rr_restores_bound_after_late_arrival():
    s=LagBoundedRandomReshuffling(5,4); s.add(range(4)); [s.draw() for _ in range(40)]; s.add(range(4,8)); [s.draw() for _ in range(48)]
    c=[s.counts[i] for i in range(8)]; assert max(c)-min(c)<=4

def test_block_weighted_rr_has_no_duplicate_inside_block():
    s=BlockWeightedRandomReshuffling(3,.1,16); s.add(range(40)); draws=[s.draw() for _ in range(48)]
    assert all(len(set(draws[k:k+16]))==16 for k in range(0,48,16))

def test_block_beta_zero_is_uniform_full_rr_when_pool_fits():
    s=BlockWeightedRandomReshuffling(4,0,128); s.add(range(21)); draws=[s.draw() for _ in range(42)]
    assert sorted(draws[:21])==list(range(21)) and sorted(draws[21:])==list(range(21))

def test_stable_pool_scheduler_matches_causal_rr_before_switch():
    baseline=CausalRandomReshuffling(11)
    staged=StablePoolBlockWeightedRandomReshuffling(11,.05,4,100)
    baseline_draws=[]; staged_draws=[]
    for step in range(1,30):
        added=range(5) if step==1 else (range(5,8) if step==7 else [])
        baseline.add(added); staged.add(added)
        baseline_draws.append(baseline.draw()); staged_draws.append(staged.draw())
    assert baseline_draws == staged_draws
    assert not staged.weighted_phase

def test_stable_pool_scheduler_switches_causally_and_has_unique_blocks():
    s=StablePoolBlockWeightedRandomReshuffling(5,.1,4,5)
    draws=[]
    for step in range(1,13):
        s.add(range(8) if step==1 else [])
        draws.append(s.draw())
    assert s.weighted_phase_start == 5
    assert len(set(draws[4:8])) == 4
    assert len(set(draws[8:12])) == 4

def test_entropy_floor_has_no_duplicate_inside_block():
    s=EntropyFloorRandomReshuffling(8,.35,16); s.add(range(50))
    draws=[s.draw() for _ in range(64)]
    assert all(len(set(draws[k:k+16]))==16 for k in range(0,64,16))

def test_entropy_floor_rho_zero_preserves_static_full_rr_epochs():
    s=EntropyFloorRandomReshuffling(2,0,16); s.add(range(40))
    draws=[s.draw() for _ in range(80)]
    assert sorted(draws[:40])==list(range(40))
    assert sorted(draws[40:])==list(range(40))

def test_interval_softmax_has_unique_intervals_inside_outer_block():
    s=IntervalSoftmaxRandomReshuffling(4,.05,4)
    for start in range(0,30,3): s.add(range(start,start+3))
    draws=[s.draw() for _ in range(12)]
    interval_draws=[s.frame_to_interval[item] for item in draws]
    assert all(len(set(interval_draws[k:k+4]))==4 for k in range(0,12,4))

def test_interval_softmax_inner_sampler_is_random_reshuffling():
    s=IntervalSoftmaxRandomReshuffling(7,0,1); s.add(range(5))
    draws=[s.draw() for _ in range(10)]
    assert sorted(draws[:5])==list(range(5))
    assert sorted(draws[5:])==list(range(5))

def test_interval_softmax_beta_zero_is_exact_interval_rr_when_pool_fits():
    s=IntervalSoftmaxRandomReshuffling(8,0,16)
    for item in range(7): s.add([item])
    draws=[s.draw() for _ in range(14)]
    assert sorted(draws[:7])==list(range(7))
    assert sorted(draws[7:])==list(range(7))

def test_interval_softmax_arrival_is_causal_and_waits_for_boundary():
    s=IntervalSoftmaxRandomReshuffling(2,0,4)
    for item in range(4): s.add([item])
    first=[s.draw() for _ in range(2)]
    s.add([4,5])
    rest=[s.draw() for _ in range(2)]
    assert 4 not in first+rest and 5 not in first+rest
    assert s.draw() in range(6)

def test_size_aware_interval_softmax_has_unique_outer_blocks():
    s=SizeAwareIntervalSoftmaxRandomReshuffling(3,.05,4)
    for start,size in ((0,2),(2,3),(5,4),(9,2),(11,5),(16,3)):
        s.add(range(start,start+size))
    draws=[s.draw() for _ in range(12)]
    interval_draws=[s.frame_to_interval[item] for item in draws]
    assert all(len(set(interval_draws[k:k+4]))==4 for k in range(0,12,4))

def test_size_aware_beta_zero_uses_interval_size_as_base_probability():
    s=SizeAwareIntervalSoftmaxRandomReshuffling(13,0,1)
    s.add([0]); s.add([1,2,3])
    draws=[s.draw() for _ in range(12000)]
    interval_draws=[s.frame_to_interval[item] for item in draws]
    ratio=interval_draws.count(1)/interval_draws.count(0)
    assert 2.8 < ratio < 3.2

def test_normalized_interval_softmax_is_invariant_to_count_scale():
    import math
    a=MeanNormalizedSizeAwareIntervalSoftmaxRandomReshuffling(71,math.log(3),4)
    b=MeanNormalizedSizeAwareIntervalSoftmaxRandomReshuffling(71,math.log(3),4)
    for start,size in ((0,2),(2,3),(5,4),(9,2),(11,5),(16,3)):
        a.add(range(start,start+size));b.add(range(start,start+size))
    for interval_id,count in enumerate((2,7,13,19,31,43)):
        a.interval_counts[interval_id]=count
        b.interval_counts[interval_id]=count*9
    a._new_interval_block();b._new_interval_block()
    assert a.remaining == b.remaining

def test_normalized_interval_softmax_has_unique_outer_blocks():
    s=MeanNormalizedSizeAwareIntervalSoftmaxRandomReshuffling(73,.7,4)
    for start,size in ((0,2),(2,3),(5,4),(9,2),(11,5),(16,3)):
        s.add(range(start,start+size))
    draws=[s.draw() for _ in range(12)]
    interval_draws=[s.frame_to_interval[item] for item in draws]
    assert all(len(set(interval_draws[k:k+4]))==4 for k in range(0,12,4))

def test_staged_interval_scheduler_matches_causal_rr_before_switch():
    baseline=CausalRandomReshuffling(11)
    staged=StagedSizeAwareIntervalSoftmaxRandomReshuffling(11,.02,4,100)
    baseline_draws=[]; staged_draws=[]
    for step in range(1,30):
        added=range(5) if step==1 else (range(5,8) if step==7 else [])
        baseline.add(added); staged.add(added)
        baseline_draws.append(baseline.draw()); staged_draws.append(staged.draw())
    assert baseline_draws == staged_draws
    assert not staged.staged_phase
    assert sum(staged.interval_counts.values()) == len(staged_draws)

def test_staged_interval_scheduler_switches_at_phase_and_has_unique_blocks():
    s=StagedSizeAwareIntervalSoftmaxRandomReshuffling(5,.02,4,5)
    draws=[]
    for step in range(1,13):
        s.add([2*step-2,2*step-1] if step <= 6 else [])
        draws.append(s.draw())
    assert s.staged_phase_start == 5
    post_intervals=[s.frame_to_interval[item] for item in draws[4:]]
    assert all(len(set(post_intervals[k:k+4]))==4 for k in range(0,8,4))

def test_staged_scheduler_can_fast_forward_to_a_shared_branch_point():
    arrivals=[1,1,4,7,7,11]
    continuous=StagedSizeAwareIntervalSoftmaxRandomReshuffling(17,.02,3,10)
    prefix=StagedSizeAwareIntervalSoftmaxRandomReshuffling(17,.02,3,10)
    next_continuous=next_prefix=0
    continuous_draws=[]
    for step in range(1,21):
        added=[]
        while next_continuous<len(arrivals) and arrivals[next_continuous]<=step:
            added.append(next_continuous);next_continuous+=1
        continuous.add(added);continuous_draws.append(continuous.draw())
    for step in range(1,10):
        added=[]
        while next_prefix<len(arrivals) and arrivals[next_prefix]<=step:
            added.append(next_prefix);next_prefix+=1
        prefix.add(added);assert prefix.draw()==continuous_draws[step-1]
    resumed=[]
    for step in range(10,21):
        added=[]
        while next_prefix<len(arrivals) and arrivals[next_prefix]<=step:
            added.append(next_prefix);next_prefix+=1
        prefix.add(added);resumed.append(prefix.draw())
    assert resumed == continuous_draws[9:]

def test_loss_aware_interval_sampler_favors_observed_high_loss():
    s=LossAwareSizeAwareIntervalSoftmaxRandomReshuffling(23,0,1,2.0,0.0)
    s.add([0]);s.add([1])
    s.observe(0,10.0);s.observe(1,1.0)
    draws=[s.draw() for _ in range(2000)]
    assert draws.count(0) > 1900

def test_staged_loss_aware_matches_causal_rr_and_ignores_prefix_losses():
    baseline=CausalRandomReshuffling(31)
    staged=StagedLossAwareSizeAwareIntervalSoftmaxRandomReshuffling(31,.02,4,100,.5)
    baseline_draws=[];staged_draws=[]
    for step in range(1,30):
        added=range(5) if step==1 else (range(5,8) if step==7 else [])
        baseline.add(added);staged.add(added)
        baseline_draws.append(baseline.draw());staged_draws.append(staged.draw())
        staged.observe(staged_draws[-1],100.0)
    assert baseline_draws == staged_draws
    assert staged.interval_loss_ema == {}

def test_bounded_interval_sampler_caps_extreme_count_odds():
    import math
    s=BoundedSizeAwareIntervalSoftmaxRandomReshuffling(37,math.log(2),1)
    s.add([0]);s.add([1])
    draws=[]
    for _ in range(12000):
        s.interval_counts[0]=0;s.interval_counts[1]=100000
        s.remaining=[]
        draws.append(s.draw())
    ratio=draws.count(0)/draws.count(1)
    assert 1.85 < ratio < 2.15

def test_staged_bounded_scheduler_matches_causal_rr_before_switch():
    baseline=CausalRandomReshuffling(41)
    staged=StagedBoundedSizeAwareIntervalSoftmaxRandomReshuffling(41,.7,4,100)
    baseline_draws=[];staged_draws=[]
    for step in range(1,30):
        added=range(5) if step==1 else (range(5,8) if step==7 else [])
        baseline.add(added);staged.add(added)
        baseline_draws.append(baseline.draw());staged_draws.append(staged.draw())
    assert baseline_draws == staged_draws

def test_two_pass_interval_sampler_caps_bonus_and_turns_it_off():
    import math
    trials=12000
    active=TwoPassSizeAwareIntervalSoftmaxRandomReshuffling(43,math.log(2),1)
    active.add([0]);active.add([1])
    draws=[]
    for _ in range(trials):
        active.interval_counts[0]=0;active.interval_counts[1]=2
        active.remaining=[];draws.append(active.draw())
    ratio=draws.count(0)/draws.count(1)
    assert 1.85 < ratio < 2.15

    mature=TwoPassSizeAwareIntervalSoftmaxRandomReshuffling(47,math.log(8),1)
    control=SizeAwareIntervalSoftmaxRandomReshuffling(47,0,1)
    mature.add([0]);mature.add([1]);control.add([0]);control.add([1])
    mature.interval_counts[0]=2;mature.interval_counts[1]=200
    control.interval_counts[0]=2;control.interval_counts[1]=200
    assert [mature.draw() for _ in range(100)] == [control.draw() for _ in range(100)]

def test_staged_two_pass_scheduler_matches_causal_rr_before_switch():
    baseline=CausalRandomReshuffling(53)
    staged=StagedTwoPassSizeAwareIntervalSoftmaxRandomReshuffling(53,.7,4,100)
    baseline_draws=[];staged_draws=[]
    for step in range(1,30):
        added=range(5) if step==1 else (range(5,8) if step==7 else [])
        baseline.add(added);staged.add(added)
        baseline_draws.append(baseline.draw());staged_draws.append(staged.draw())
    assert baseline_draws == staged_draws

def test_relative_floor_interval_sampler_has_bounded_softmax_bonus():
    import math
    s=RelativeFloorIntervalSoftmaxRandomReshuffling(59,math.log(2),1)
    s.add([0]);s.add([1])
    draws=[]
    for _ in range(12000):
        # mean=2, relative floor=1: interval 0 has full deficit, 1 has none.
        s.interval_counts[0]=0;s.interval_counts[1]=4
        s.remaining=[];draws.append(s.draw())
    ratio=draws.count(0)/draws.count(1)
    assert 1.85 < ratio < 2.15
