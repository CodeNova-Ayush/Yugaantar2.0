"""Testable progression-event tracker and summary aggregator.

Uses ceiling-minus-floor clearance >= 56.0 map units for door passability,
matching the player height in docs/CONTRACT.md §2.2.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from doom_env.sensors import RAY_LOCKED
from doom_env.spaces import BUTTONS

PLAYER_CLEARANCE_UNITS = 56.0
DOOR_PROXIMITY_UNITS = 96.0  # 1.5 x 64-unit cell around door center (spatial proximity != USE positioning)
USE_BUTTON_IDX = BUTTONS.index("USE")


@dataclass
class ProgressionEventTracker:
    door_sec: int
    key_vital_idx: int
    min_clearance: float = PLAYER_CLEARANCE_UNITS
    door_center: tuple[float, float] | None = None
    proximity_radius: float = DOOR_PROXIMITY_UNITS
    max_decisions: int = 525

    decision: int = 0
    door_first_observed_decision: int | None = None
    door_last_observed_decision: int | None = None
    door_observed_before_key: bool = False

    key_picked_up: bool = False
    key_pickup_decision: int | None = None
    remaining_budget_at_key: int | None = None
    door_sensed_at_key_pickup: bool | None = None
    steps_door_absent_before_key: int | None = None

    door_resensed_after_key: bool = False
    door_resensed_decision: int | None = None
    returned_near_door_after_key: bool = False
    returned_near_door_decision: int | None = None
    min_door_distance_after_key: float | None = None
    use_rising_edges_near_door: int = 0
    _prev_use: int = 0

    opening_initiated_after_key: bool = False
    opening_initiated_decision: int | None = None
    door_reversals_after_key: int = 0
    _prev_clearance: float | None = None
    _door_was_opening: bool = False

    door_passable_after_key: bool = False
    door_passable_decision: int | None = None

    exit_reached: bool = False
    exit_decision: int | None = None

    def observe_pre_step(self, obs: dict) -> None:
        """Record whether RAY_LOCKED is present in the current observation before acting."""
        if np.any(obs["rays"]["hit_type"] == RAY_LOCKED):
            if self.door_first_observed_decision is None:
                self.door_first_observed_decision = self.decision
            self.door_last_observed_decision = self.decision
            if not self.key_picked_up:
                self.door_observed_before_key = True

    def record_post_step(
        self,
        obs: dict,
        sector_floor: np.ndarray,
        sector_ceiling: np.ndarray,
        exit_reached: bool,
        *,
        action: np.ndarray | None = None,
        player_pos: tuple[float, float] | None = None,
    ) -> None:
        """Update key pickup, return proximity, USE edges, door opening/reversals/passability, and exit."""
        self.decision += 1

        use_now = int(action[USE_BUTTON_IDX]) if action is not None else 0
        use_rising = (self._prev_use == 0) and (use_now == 1)
        self._prev_use = use_now

        was_key_already = self.key_picked_up
        if not self.key_picked_up and float(obs["vitals"][self.key_vital_idx]) == 1.0:
            self.key_picked_up = True
            self.key_pickup_decision = self.decision
            self.remaining_budget_at_key = self.max_decisions - self.decision
            sensed_now = bool(np.any(obs["rays"]["hit_type"] == RAY_LOCKED))
            self.door_sensed_at_key_pickup = sensed_now
            if sensed_now:
                self.door_last_observed_decision = self.decision
                self.steps_door_absent_before_key = 0
            elif self.door_last_observed_decision is not None:
                self.steps_door_absent_before_key = self.decision - self.door_last_observed_decision
            else:
                self.steps_door_absent_before_key = self.decision

        if was_key_already:
            if not self.door_resensed_after_key and np.any(obs["rays"]["hit_type"] == RAY_LOCKED):
                self.door_resensed_after_key = True
                self.door_resensed_decision = self.decision

            if player_pos is not None and self.door_center is not None:
                dist = float(math.hypot(player_pos[0] - self.door_center[0], player_pos[1] - self.door_center[1]))
                if self.min_door_distance_after_key is None or dist < self.min_door_distance_after_key:
                    self.min_door_distance_after_key = dist
                near_now = dist <= self.proximity_radius
                if near_now:
                    if not self.returned_near_door_after_key:
                        self.returned_near_door_after_key = True
                        self.returned_near_door_decision = self.decision
                    if use_rising:
                        self.use_rising_edges_near_door += 1

        if self.key_picked_up:
            clearance = float(sector_ceiling[self.door_sec]) - float(sector_floor[self.door_sec])
            if not self.opening_initiated_after_key and clearance > 0.0:
                self.opening_initiated_after_key = True
                self.opening_initiated_decision = self.decision
            if self._prev_clearance is not None:
                if clearance > self._prev_clearance + 1e-3:
                    self._door_was_opening = True
                elif self._door_was_opening and clearance < self._prev_clearance - 1e-3:
                    self.door_reversals_after_key += 1
                    self._door_was_opening = False
            self._prev_clearance = clearance

            if not self.door_passable_after_key and clearance >= self.min_clearance:
                self.door_passable_after_key = True
                self.door_passable_decision = self.decision

        if exit_reached and not self.exit_reached:
            self.exit_reached = True
            self.exit_decision = self.decision

    @property
    def failure_category(self) -> str:
        """Mutually exclusive outcome/failure taxonomy."""
        if self.exit_reached:
            return "exit"
        if not self.key_picked_up:
            return "no_key"
        if self.door_passable_after_key:
            return "passable_no_exit"
        if self.opening_initiated_after_key:
            if self.door_center is not None and not self.returned_near_door_after_key:
                return "ambiguous_opening_without_proximity"
            return "opening_no_passable"
        if self.door_center is not None and not self.returned_near_door_after_key:
            return "key_no_return"
        return "return_no_opening"

    @property
    def ordered_sequence_complete(self) -> bool:
        """True iff door_seen_before_key -> key -> opening -> passable -> exit occurred in order."""
        return (
            self.door_observed_before_key
            and self.key_picked_up
            and self.opening_initiated_after_key
            and self.door_passable_after_key
            and self.exit_reached
            and self.door_first_observed_decision is not None
            and self.key_pickup_decision is not None
            and self.opening_initiated_decision is not None
            and self.door_passable_decision is not None
            and self.exit_decision is not None
            and self.door_first_observed_decision < self.key_pickup_decision
            and self.key_pickup_decision <= self.opening_initiated_decision
            and self.opening_initiated_decision <= self.door_passable_decision
            and self.door_passable_decision <= self.exit_decision
        )


def progression_step_reward(
    tracker: ProgressionEventTracker,
    was_key_picked: bool,
    was_door_passable: bool,
    terminated: bool,
    exit_reached: bool,
    *,
    key_reward: float,
    door_passable_reward: float,
    exit_reward: float,
    death_penalty: float,
    step_cost: float,
) -> dict[str, float]:
    """Compute decomposed progression reward with one-time key and passability bonuses."""
    key_bonus = float(key_reward) if (not was_key_picked and tracker.key_picked_up) else 0.0
    door_bonus = float(door_passable_reward) if (not was_door_passable and tracker.door_passable_after_key) else 0.0
    term_bonus = (float(exit_reward) if exit_reached else float(death_penalty)) if terminated else 0.0
    s_cost = float(step_cost)
    return {
        "step_cost": s_cost,
        "key_bonus": key_bonus,
        "door_passable_bonus": door_bonus,
        "terminal_bonus": term_bonus,
        "total": s_cost + key_bonus + door_bonus + term_bonus,
    }


def summarize_progression_episodes(episodes: list[dict]) -> dict:
    """Aggregate progression episode dicts with explicit numerators and denominators."""
    n = len(episodes)
    exits = [e for e in episodes if e["outcome"] == "exit"]
    key_eps = [e for e in episodes if e["key_picked_up"]]
    init_eps = [e for e in key_eps if e["opening_initiated_after_key"]]
    pass_eps = [e for e in key_eps if e["door_passable_after_key"]]
    seen_eps = [e for e in episodes if e["door_observed_before_key"]]
    seq_eps = [e for e in episodes if e["ordered_sequence_complete"]]

    categories = (
        "exit",
        "no_key",
        "key_no_return",
        "return_no_opening",
        "opening_no_passable",
        "passable_no_exit",
        "ambiguous_opening_without_proximity",
    )
    failure_counts = {c: sum(1 for e in episodes if e.get("failure_category") == c) for c in categories}
    budgets = [e["remaining_budget_at_key"] for e in key_eps if e.get("remaining_budget_at_key") is not None]
    use_edges = [e["use_rising_edges_near_door"] for e in key_eps if e.get("use_rising_edges_near_door") is not None]

    k_den = len(key_eps)
    return {
        "episodes": n,
        "exits": len(exits),
        "unconditional_exit_rate": len(exits) / n if n else None,
        "door_observed_before_key_count": len(seen_eps),
        "key_picked_up_count": k_den,
        "key_pickup_rate": k_den / n if n else None,
        "mean_remaining_budget_at_key": float(np.mean(budgets)) if budgets else None,
        "door_resensed_after_key_count": sum(1 for e in key_eps if e.get("door_resensed_after_key")),
        "returned_near_door_after_key_count": sum(1 for e in key_eps if e.get("returned_near_door_after_key")),
        "mean_use_rising_edges_near_door_given_key": float(np.mean(use_edges)) if use_edges else None,
        "total_door_reversals_after_key": sum(int(e.get("door_reversals_after_key", 0)) for e in key_eps),
        "unlock_initiation_numerator": len(init_eps),
        "unlock_initiation_denominator": k_den,
        "unlock_initiation_rate": len(init_eps) / k_den if k_den else None,
        "passable_door_numerator": len(pass_eps),
        "passable_door_denominator": k_den,
        "passable_door_rate": len(pass_eps) / k_den if k_den else None,
        "ordered_sequence_count": len(seq_eps),
        "ordered_sequence_rate": len(seq_eps) / n if n else None,
        "failure_categories": failure_counts,
        "timeouts": sum(1 for e in episodes if e["outcome"] == "timeout"),
        "deaths": sum(1 for e in episodes if e["outcome"] == "death"),
        "decisions_mean": float(np.mean([e["decisions"] for e in episodes])) if episodes else None,
        "exit_decisions_mean": float(np.mean([e["decisions"] for e in exits])) if exits else None,
    }
