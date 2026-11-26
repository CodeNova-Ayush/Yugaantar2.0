"""Training-only event rewards. Never imported by submission inference."""
import math


class TaskRewardTracker:
    def __init__(self, state):
        self.kills = int(state['kills'])
        self.damage = float(state['damage_taken'])
        self.visited = {self.cell(state)}

    @staticmethod
    def cell(state):
        return (math.floor(state['x'] / 64), math.floor(state['y'] / 64))

    def step(self, state, kill_reward=0., damage_penalty=0., exploration_reward=0.):
        kills, damage = int(state['kills']), float(state['damage_taken'])
        cell = self.cell(state)
        result = {
            'kill_bonus': max(0, kills - self.kills) * kill_reward,
            'damage_cost': -max(0., damage - self.damage) * damage_penalty,
            'exploration_bonus': exploration_reward if cell not in self.visited else 0.,
        }
        self.kills, self.damage = kills, damage
        self.visited.add(cell)
        return result
