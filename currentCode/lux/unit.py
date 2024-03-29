import math
import sys
from typing import List
import numpy as np
from dataclasses import dataclass
from lux.cargo import UnitCargo
from lux.config import EnvConfig

# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])


@dataclass
class Unit:
    team_id: int
    unit_id: str
    unit_type: str  # "LIGHT" or "HEAVY"
    pos: np.ndarray
    power: int
    cargo: UnitCargo
    env_cfg: EnvConfig
    unit_cfg: dict
    action_queue: List

    def to_string(self):
        return self.unit_type_short() + "(" + str(self.power) + ") @" + str(self.pos) + ' ' + self.unit_id_short()

    def to_string2(self):
        return self.to_string() + str(self.cargo)

    def cargo_space_left(self):
        return self.cargo_space() - self.cargo.total()

    def unit_type_short(self):
        if len(self.unit_type) > 0:
            return self.unit_type[0]
        else:
            return ''

    def unit_id_short(self):
        if len(self.unit_type) > 4:
            return 'u' + self.unit_id[4:]
        else:
            return self.unit_id

    @property
    def agent_id(self):
        if self.team_id == 0: return "player_0"
        return "player_1"

    def charge_per_turn(self):
        return self.env_cfg.ROBOTS[self.unit_type].CHARGE;

    def action_queue_cost(self):
        cost = self.env_cfg.ROBOTS[self.unit_type].ACTION_QUEUE_POWER_COST
        return cost

    def move_cost_to(self, game_state, target_pos):
        board = game_state.board
        if target_pos[0] < 0 or target_pos[1] < 0 or target_pos[1] >= len(board.rubble) or target_pos[0] >= len(board.rubble[0]):
            # print("Warning, tried to get move cost for going off the map", file=sys.stderr)
            return None
        factory_there = board.factory_occupancy_map[target_pos[0], target_pos[1]]
        if factory_there not in game_state.teams[self.agent_id].factory_strains and factory_there != -1:
            # print("Warning, tried to get move cost for going onto a opposition factory", file=sys.stderr)
            return None
        rubble_at_target = board.rubble[target_pos[0]][target_pos[1]]

        return math.floor(self.unit_cfg.MOVE_COST + self.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target)

    def on_factory(self,game_state):
        board = game_state.board
        return board.factory_occupancy_map[self.pos[0], self.pos[1]] != -1

    # USE actions.move_cost to take also into account the queue cost
    def _move_cost(self, game_state, direction,extra_cost =0):
        cost =  self.move_cost_to(game_state, self.pos + move_deltas[direction])
        if cost is None: return None
        return cost + extra_cost

    # USE actions.can_move to take also into account the queue cost
    def _can_move_to(self, game_state, direction, extra_cost = 0):
        cost = self._move_cost(game_state,direction, extra_cost = extra_cost )
        if cost is None:
            return False
        else:
            return self.power >=  cost

    def move(self, direction, repeat=0, n=1):
        if isinstance(direction, int):
            direction = direction
        else:
            pass
        return np.array([0, direction, 0, 0, repeat, n])

    def transfer(self, transfer_direction, transfer_resource, transfer_amount, repeat=0, n=1):
        assert transfer_resource < 5 and transfer_resource >= 0
        assert transfer_direction < 5 and transfer_direction >= 0
        return np.array([1, transfer_direction, transfer_resource, transfer_amount, repeat, n])

    def pickup(self, pickup_resource, pickup_amount, repeat=0, n=1):
        assert pickup_resource < 5 and pickup_resource >= 0
        return np.array([2, 0, pickup_resource, pickup_amount, repeat, n])

    def dig_cost(self):
        return self.unit_cfg.DIG_COST

    def dig(self, repeat=0, n=1):
        return np.array([3, 0, 0, 0, repeat, n])

    def self_destruct_cost(self, game_state):
        return self.unit_cfg.SELF_DESTRUCT_COST

    def self_destruct(self, repeat=0, n=1):
        return np.array([4, 0, 0, 0, repeat, n])

    def recharge(self, x, repeat=0, n=1):
        return np.array([5, 0, 0, x, repeat, n])

    def battery_capacity(self):
        return 150 if self.unit_type == "LIGHT" else 3000

    def battery_capacity_left(self):
        return self.battery_capacity() - self.power

    def battery_info(self):
        return "battery(" + str(self.power) + "/" + str(self.battery_capacity()) + " lft=" + str(self.battery_capacity_left()) + ")"

    def cargo_space(self):
        return 100 if self.unit_type == "LIGHT" else 1000

    def def_move_cost(self):
        return 1 if self.unit_type == "LIGHT" else 20

    def rubble_dig_hurdle(self):  # SEEMS SOMETHING SPECIFIC
        return 5 if self.unit_type == "LIGHT" else 100

    def pos_location(self):
        return (self.pos[0], self.pos[1])

    def get_distance(self, pos):
        return abs(self.pos[0] - pos[0]) + abs(self.pos[1] - pos[1])

    # Manhattan Distance between one points and one vector, return a vector
    def get_distance_vector(self, points):
        return 2 * np.mean(np.abs(points - self.pos), 1)

    def is_heavy(self):
        return self.unit_type == 'HEAVY'

    def is_light(self):
        return self.unit_type == 'LIGHT'

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.pos}"
        return out
