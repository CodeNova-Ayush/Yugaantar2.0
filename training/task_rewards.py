"""Training-only event rewards. Never imported by submission inference."""
import math
import numpy as np


class TaskRewardTracker:
    def __init__(self, state):
        self.kills = int(state['kills'])
        self.damage = float(state['damage_taken'])
        self.visited = {self.cell(state)}
        self.unlocked_doors_opened: set[int] = set()
        locks = state.get('door_lock')
        if locks is not None and len(locks) > 0:
            self.prev_door_clearance = (state['sector_ceiling'] - state['sector_floor']).astype(float, copy=True)
        else:
            self.prev_door_clearance = None

    @staticmethod
    def cell(state):
        return (math.floor(state['x'] / 64), math.floor(state['y'] / 64))

    def step(
        self,
        state,
        kill_reward=0.,
        damage_penalty=0.,
        exploration_reward=0.,
        unlocked_door_reward=0.,
        door_reversal_penalty=0.,
    ):
        kills, damage = int(state['kills']), float(state['damage_taken'])
        cell = self.cell(state)
        unlocked_bonus = 0.0
        reversal_cost = 0.0
        locks = state.get('door_lock')
        if locks is not None and len(locks) > 0:
            clearance = (state['sector_ceiling'] - state['sector_floor']).astype(float, copy=False)
            for idx in range(len(locks)):
                if locks[idx] == 0 and clearance[idx] >= 56.0 and idx not in self.unlocked_doors_opened:
                    self.unlocked_doors_opened.add(idx)
                    unlocked_bonus += float(unlocked_door_reward)
                if (
                    locks[idx] >= 0
                    and self.prev_door_clearance is not None
                    and 0.0 < self.prev_door_clearance[idx] < 120.0
                    and clearance[idx] < self.prev_door_clearance[idx]
                ):
                    reversal_cost -= float(door_reversal_penalty)
            self.prev_door_clearance = clearance.copy()

        result = {
            'kill_bonus': max(0, kills - self.kills) * kill_reward,
            'damage_cost': -max(0., damage - self.damage) * damage_penalty,
            'exploration_bonus': exploration_reward if cell not in self.visited else 0.,
        }
        if unlocked_door_reward > 0:
            result['unlocked_door_bonus'] = unlocked_bonus
        if door_reversal_penalty > 0:
            result['door_reversal_cost'] = reversal_cost
        self.kills, self.damage = kills, damage
        self.visited.add(cell)
        return result
