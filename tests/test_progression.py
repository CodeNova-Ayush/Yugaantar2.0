"""Regression tests for ProgressionEventTracker and progression summary aggregation."""

from __future__ import annotations

import numpy as np

from doom_env.sensors import RAY_LOCKED, RAY_WALL
from doom_env.spaces import VITALS_INDEX, observation_space
from training.progression.tracker import (
    ProgressionEventTracker,
    progression_step_reward,
    summarize_progression_episodes,
)


def _obs(door_seen: bool = False, has_red_key: bool = False) -> dict:
    obs = observation_space().sample()
    obs["rays"]["hit_type"][:] = RAY_WALL
    if door_seen:
        obs["rays"]["hit_type"][0] = RAY_LOCKED
    obs["vitals"][:] = 0.0
    if has_red_key:
        obs["vitals"][VITALS_INDEX["key_red_card"]] = 1.0
    return obs


def test_no_key_never_initiates_or_passes():
    tracker = ProgressionEventTracker(door_sec=2, key_vital_idx=VITALS_INDEX["key_red_card"])
    floors = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ceils = np.array([128.0, 128.0, 0.0], dtype=np.float32)

    for _ in range(5):
        o = _obs(door_seen=True, has_red_key=False)
        tracker.observe_pre_step(o)
        tracker.record_post_step(o, floors, ceils, exit_reached=False)

    assert tracker.door_observed_before_key is True
    assert tracker.key_picked_up is False
    assert tracker.opening_initiated_after_key is False
    assert tracker.door_passable_after_key is False
    assert tracker.ordered_sequence_complete is False


def test_partial_opening_and_reversal_not_passable():
    tracker = ProgressionEventTracker(door_sec=2, key_vital_idx=VITALS_INDEX["key_red_card"])
    floors = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Step 1: see locked door
    o1 = _obs(door_seen=True, has_red_key=False)
    tracker.observe_pre_step(o1)
    tracker.record_post_step(o1, floors, np.array([128.0, 128.0, 0.0]), exit_reached=False)

    # Step 2: pick up key while door is occluded
    o2 = _obs(door_seen=False, has_red_key=True)
    tracker.observe_pre_step(o2)
    tracker.record_post_step(o2, floors, np.array([128.0, 128.0, 0.0]), exit_reached=False)
    assert tracker.door_sensed_at_key_pickup is False
    assert tracker.steps_door_absent_before_key == 2

    # Step 3: door starts opening (20 units < 56 clearance) then reverses to 0
    o3 = _obs(door_seen=True, has_red_key=True)
    tracker.observe_pre_step(o3)
    tracker.record_post_step(o3, floors, np.array([128.0, 128.0, 20.0]), exit_reached=False)
    tracker.observe_pre_step(o3)
    tracker.record_post_step(o3, floors, np.array([128.0, 128.0, 0.0]), exit_reached=False)

    assert tracker.opening_initiated_after_key is True
    assert tracker.door_passable_after_key is False
    assert tracker.ordered_sequence_complete is False


def test_successful_opening_and_event_ordering():
    tracker = ProgressionEventTracker(door_sec=2, key_vital_idx=VITALS_INDEX["key_red_card"])
    floors = np.array([0.0, 0.0, 16.0], dtype=np.float32)  # non-zero floor tests clearance = ceil - floor

    # Step 1: door observed
    o1 = _obs(door_seen=True, has_red_key=False)
    tracker.observe_pre_step(o1)
    tracker.record_post_step(o1, floors, np.array([128.0, 128.0, 16.0]), exit_reached=False)

    # Step 2: key picked up
    o2 = _obs(door_seen=False, has_red_key=True)
    tracker.observe_pre_step(o2)
    tracker.record_post_step(o2, floors, np.array([128.0, 128.0, 16.0]), exit_reached=False)

    # Step 3: opening initiated (clearance = 32 < 56)
    o3 = _obs(door_seen=True, has_red_key=True)
    tracker.observe_pre_step(o3)
    tracker.record_post_step(o3, floors, np.array([128.0, 128.0, 48.0]), exit_reached=False)

    # Step 4: passable (clearance = 64 >= 56) and exit reached
    tracker.observe_pre_step(o3)
    tracker.record_post_step(o3, floors, np.array([128.0, 128.0, 80.0]), exit_reached=True)

    assert tracker.ordered_sequence_complete is True
    assert tracker.door_first_observed_decision == 0
    assert tracker.key_pickup_decision == 2
    assert tracker.opening_initiated_decision == 3
    assert tracker.door_passable_decision == 4


def test_key_before_door_is_not_ordered_sequence():
    tracker = ProgressionEventTracker(door_sec=2, key_vital_idx=VITALS_INDEX["key_red_card"])
    floors = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    o_key = _obs(door_seen=False, has_red_key=True)
    tracker.observe_pre_step(o_key)
    tracker.record_post_step(o_key, floors, np.array([128.0, 128.0, 0.0]), exit_reached=False)
    tracker.observe_pre_step(o_key)
    tracker.record_post_step(o_key, floors, np.array([128.0, 128.0, 64.0]), exit_reached=True)

    assert tracker.door_observed_before_key is False
    assert tracker.door_passable_after_key is True
    assert tracker.ordered_sequence_complete is False


def test_zero_denominator_summary():
    ep = {
        "outcome": "timeout",
        "decisions": 525,
        "key_picked_up": False,
        "opening_initiated_after_key": False,
        "door_passable_after_key": False,
        "door_observed_before_key": True,
        "ordered_sequence_complete": False,
    }
    summ = summarize_progression_episodes([ep])
    assert summ["unlock_initiation_numerator"] == 0
    assert summ["unlock_initiation_denominator"] == 0
    assert summ["unlock_initiation_rate"] is None
    assert summ["passable_door_numerator"] == 0
    assert summ["passable_door_denominator"] == 0
    assert summ["passable_door_rate"] is None
    assert summ["unconditional_exit_rate"] == 0.0


def test_progression_rewards_at_most_once_per_episode_and_reset():
    floors = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def run_episode() -> tuple[float, float, ProgressionEventTracker]:
        tracker = ProgressionEventTracker(door_sec=2, key_vital_idx=VITALS_INDEX["key_red_card"])
        key_sum = 0.0
        door_sum = 0.0
        # 1. Door seen
        o1 = _obs(door_seen=True, has_red_key=False)
        tracker.observe_pre_step(o1)
        wk, wd = tracker.key_picked_up, tracker.door_passable_after_key
        tracker.record_post_step(o1, floors, np.array([128.0, 128.0, 0.0]), exit_reached=False)
        r = progression_step_reward(tracker, wk, wd, False, False, key_reward=0.25, door_passable_reward=0.5, exit_reward=1.0, death_penalty=-1.0, step_cost=-0.002)
        key_sum += r["key_bonus"]
        door_sum += r["door_passable_bonus"]

        # 2. Key picked up (should fire +0.25 once)
        o2 = _obs(door_seen=False, has_red_key=True)
        for _ in range(2):
            tracker.observe_pre_step(o2)
            wk, wd = tracker.key_picked_up, tracker.door_passable_after_key
            tracker.record_post_step(o2, floors, np.array([128.0, 128.0, 0.0]), exit_reached=False)
            r = progression_step_reward(tracker, wk, wd, False, False, key_reward=0.25, door_passable_reward=0.5, exit_reward=1.0, death_penalty=-1.0, step_cost=-0.002)
            key_sum += r["key_bonus"]
            door_sum += r["door_passable_bonus"]

        # 3. Door opens -> closes -> reopens (should fire +0.5 only once)
        for ceil_h in (64.0, 0.0, 64.0):
            o3 = _obs(door_seen=True, has_red_key=True)
            tracker.observe_pre_step(o3)
            wk, wd = tracker.key_picked_up, tracker.door_passable_after_key
            tracker.record_post_step(o3, floors, np.array([128.0, 128.0, ceil_h]), exit_reached=False)
            r = progression_step_reward(tracker, wk, wd, False, False, key_reward=0.25, door_passable_reward=0.5, exit_reward=1.0, death_penalty=-1.0, step_cost=-0.002)
            key_sum += r["key_bonus"]
            door_sum += r["door_passable_bonus"]

        # 4. Exit
        tracker.observe_pre_step(o3)
        wk, wd = tracker.key_picked_up, tracker.door_passable_after_key
        tracker.record_post_step(o3, floors, np.array([128.0, 128.0, 64.0]), exit_reached=True)
        return key_sum, door_sum, tracker

    k1, d1, t1 = run_episode()
    assert k1 == 0.25 and d1 == 0.5 and t1.exit_decision == 7 and t1.ordered_sequence_complete
    # Second fresh episode resets and earns the exact same one-time bonuses
    k2, d2, t2 = run_episode()
    assert k2 == 0.25 and d2 == 0.5 and t2.exit_decision == 7 and t2.ordered_sequence_complete


def test_diagnostic_failure_categories_and_transitions():
    floors = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    door_center = (300.0, 100.0)
    use_on = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int8)
    use_off = np.zeros(9, dtype=np.int8)

    # 1. key_no_return
    t1 = ProgressionEventTracker(door_sec=2, key_vital_idx=VITALS_INDEX["key_red_card"], door_center=door_center, max_decisions=525)
    o_key = _obs(door_seen=False, has_red_key=True)
    t1.observe_pre_step(o_key)
    t1.record_post_step(o_key, floors, np.array([128.0, 128.0, 0.0]), False, action=use_off, player_pos=(50.0, 100.0))
    t1.observe_pre_step(o_key)
    t1.record_post_step(o_key, floors, np.array([128.0, 128.0, 0.0]), False, action=use_on, player_pos=(60.0, 100.0))
    assert t1.remaining_budget_at_key == 524
    assert t1.returned_near_door_after_key is False
    assert t1.use_rising_edges_near_door == 0
    assert t1.failure_category == "key_no_return"

    # 2. return_no_opening (returns within 96 units, presses USE twice -> 1 rising edge, door stays closed)
    t2 = ProgressionEventTracker(door_sec=2, key_vital_idx=VITALS_INDEX["key_red_card"], door_center=door_center)
    t2.observe_pre_step(o_key)
    t2.record_post_step(o_key, floors, np.array([128.0, 128.0, 0.0]), False, action=use_off, player_pos=(50.0, 100.0))
    o_near = _obs(door_seen=True, has_red_key=True)
    t2.observe_pre_step(o_near)
    t2.record_post_step(o_near, floors, np.array([128.0, 128.0, 0.0]), False, action=use_on, player_pos=(250.0, 100.0))
    t2.observe_pre_step(o_near)
    t2.record_post_step(o_near, floors, np.array([128.0, 128.0, 0.0]), False, action=use_on, player_pos=(250.0, 100.0))
    assert t2.door_resensed_after_key is True and t2.door_resensed_decision == 2
    assert t2.returned_near_door_after_key is True and t2.use_rising_edges_near_door == 1
    assert t2.failure_category == "return_no_opening"

    # 3. opening_no_passable with reversal
    t3 = ProgressionEventTracker(door_sec=2, key_vital_idx=VITALS_INDEX["key_red_card"], door_center=door_center)
    t3.observe_pre_step(o_key)
    t3.record_post_step(o_key, floors, np.array([128.0, 128.0, 0.0]), False, action=use_off, player_pos=(50.0, 100.0))
    t3.observe_pre_step(o_near)
    t3.record_post_step(o_near, floors, np.array([128.0, 128.0, 24.0]), False, action=use_on, player_pos=(250.0, 100.0))
    t3.observe_pre_step(o_near)
    t3.record_post_step(o_near, floors, np.array([128.0, 128.0, 12.0]), False, action=use_on, player_pos=(250.0, 100.0))
    assert t3.door_reversals_after_key == 1
    assert t3.failure_category == "opening_no_passable"

    # 4. passable_no_exit
    t4 = ProgressionEventTracker(door_sec=2, key_vital_idx=VITALS_INDEX["key_red_card"], door_center=door_center)
    t4.observe_pre_step(o_key)
    t4.record_post_step(o_key, floors, np.array([128.0, 128.0, 0.0]), False, action=use_off, player_pos=(50.0, 100.0))
    t4.observe_pre_step(o_near)
    t4.record_post_step(o_near, floors, np.array([128.0, 128.0, 64.0]), False, action=use_on, player_pos=(250.0, 100.0))
    assert t4.failure_category == "passable_no_exit"
