import lux.kit
from action import *
from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys
def pr(*args, sep=' ', end='\n', force=False):  # print conditionally
    if (True or f): # change first parameter to False to disable logging
        print(*args, sep=sep, file=sys.stderr)

def prx(*args): pr(*args, force=True)



class Queue:
    def get_queue_length(unit:lux.kit.Unit):
        return len(unit.action_queue)

    def has_queue(unit: lux.kit.Unit):
        return Queue.get_queue_length(unit)>0

    def next_action(unit: lux.kit.Unit):
        if Queue.has_queue(unit):
            return unit.action_queue[0]
        else:
            return None

    def is_dig(a):
        if a is not None and len(a)>0:
            return (a[0]==3)
        else:
            return False

    def is_next_queue_dig(unit: lux.kit.Unit):
        return Queue.is_dig(Queue.next_action(unit))


class Action_Queue():
    def __init__(self, game_state) -> None:
        self.game_state = game_state
        self.actions = dict()

    def build_heavy(self, f):
        self.actions[f.unit_id] = f.build_heavy()

    def build_light(self, f):
        self.actions[f.unit_id] = f.build_light()

    def water(self, f):
        self.actions[f.unit_id] = f.water()

    def can_dig(self, unit):
        if not Queue.is_next_queue_dig(unit):
            # not already digging action, dig cost + cost action queue cost
            return unit.power >= (unit.dig_cost(self.game_state) + unit.action_queue_cost(self.game_state))
        else:
            # already digging action, only cost for dig
            return unit.power >= unit.dig_cost(self.game_state)


    def dig(self, unit):
        if not Queue.is_next_queue_dig(unit):
            # only dig if we are not already digging.
            self.actions[unit.unit_id] = [unit.dig(repeat=True)]

    def dropcargo_or_recharge(self, unit):
        if unit.cargo.ore > 0:
            self.actions[unit.unit_id] = [unit.transfer(0, 1, unit.cargo.ore, repeat=False)]
        elif unit.cargo.ice > 0:
            self.actions[unit.unit_id] = [unit.transfer(0, 0, unit.cargo.ice, repeat=False)]
        elif unit.power < unit.battery_capacity() * 0.1:
            self.actions[unit.unit_id] = [unit.pickup(4, unit.battery_capacity() - unit.power)]

    def move(self, unit, direction):
        self.actions[unit.unit_id] = [unit.move(direction, repeat=False)]

    def get_next_action(self, unit):
        pass