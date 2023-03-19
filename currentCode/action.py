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

    def is_transfer_ore(a):
        if a is not None and len(a)>0:
            return (a[0]==1 and a[2]==1)
        else:
            return False

    def is_move(a, direction):
        if a is not None and len(a)>0:
            return (a[0]==0 and a[1]==direction)
        else:
            return False

    def is_next_queue_transfer_ore(unit: lux.kit.Unit):
        return Queue.is_transfer_ore(Queue.next_action(unit))

    def is_next_queue_move(unit: lux.kit.Unit, direction):
        return Queue.is_move(Queue.next_action(unit), direction)

    def is_transfer_ice(a):
        if a is not None and len(a)>0:
            return (a[0]==1 and a[2]==0)
        else:
            return False

    def is_next_queue_transfer_ice(unit: lux.kit.Unit):
        return Queue.is_transfer_ice(Queue.next_action(unit))

    def is_pickup(a):
        if a is not None and len(a)>0:
            return (a[0]==2)
        else:
            return False

    def is_next_queue_pickup(unit: lux.kit.Unit):
        return Queue.is_pickup(Queue.next_action(unit))

    def action_pickup_power(unit: lux.kit.Unit, power, repeat=False):
        return unit.pickup(4, power, repeat=repeat)

    def action_transfer_ore(unit: lux.kit.Unit):
        return unit.transfer(0, 1, unit.cargo.ore, repeat=False)

    def action_transfer_ice(unit: lux.kit.Unit):
        return unit.transfer(0, 0, unit.cargo.ice, repeat=False)

    def real_cost_dig(unit: lux.kit.Unit):
        if not Queue.is_next_queue_dig(unit):
            return unit.action_queue_cost() + unit.dig_cost()
        else:
            return unit.dig_cost()


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
            return unit.power >= (unit.dig_cost() + unit.action_queue_cost())
        else:
            # already digging action, only cost for dig
            return unit.power >= unit.dig_cost()

    def dropcargo_or_recharge(self, unit: lux.kit.Unit):
        if unit.cargo.ore > 0 and not Queue.is_next_queue_transfer_ore(unit):
            if unit.power < unit.battery_capacity() * 0.1 and not Queue.is_next_queue_pickup(unit):
                self.actions[unit.unit_id] = [Queue.action_transfer_ore(unit),
                                              Queue.action_pickup_power(unit, unit.battery_capacity() - unit.power)]
            else:
                self.actions[unit.unit_id] = [Queue.action_transfer_ore(unit)]
        elif unit.cargo.ice > 0 and not Queue.is_next_queue_transfer_ore(unit):
            if unit.power < unit.battery_capacity() * 0.1 and not Queue.is_next_queue_pickup(unit):
                self.actions[unit.unit_id] = [Queue.action_transfer_ice(unit),
                                          Queue.action_pickup_power(unit, unit.battery_capacity() - unit.power)]
            else:
                self.actions[unit.unit_id] = [Queue.action_transfer_ice(unit)]
        elif unit.power < unit.battery_capacity() * 0.1 and not Queue.is_next_queue_pickup(unit):
            self.actions[unit.unit_id] = [Queue.action_pickup_power(unit, unit.battery_capacity() - unit.power)]

    def set_new_actions(self, unit, unit_actions,PREFIX):
        if len(unit_actions)>20:
            prx(PREFIX,"Actions too long",len(unit_actions),"truncating")
            unit_actions = unit_actions[:20]
        self.actions[unit.unit_id] = unit_actions

    def dig(self, unit):
        if not Queue.is_next_queue_dig(unit):
            # only dig if we are not already digging.
            self.actions[unit.unit_id] = [unit.dig(repeat=False, n=9999)]

    def transfer_ice(self, unit):
        self.actions[unit.unit_id] = [Queue.action_transfer_ice(unit)]

    def transfer_ore(self, unit):
        self.actions[unit.unit_id] = [Queue.action_transfer_ore(unit)]

    def move(self, unit, direction):
        if not Queue.is_next_queue_move(unit,direction):
            self.actions[unit.unit_id] = [unit.move(direction, repeat=False)]


