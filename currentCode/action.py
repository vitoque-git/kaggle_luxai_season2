import lux.kit
from action import *
from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys


def pr(*args, sep=' ', end='\n', force=False):  # print conditionally
    if (True or force):  # change first parameter to False to disable logging
        print(*args, sep=sep, file=sys.stderr)


def prx(*args): pr(*args, force=True)


class Queue:
    def get_queue_length(unit: lux.kit.Unit):
        return len(unit.action_queue)

    def has_queue(unit: lux.kit.Unit):
        return Queue.get_queue_length(unit) > 0

    def next_action(unit: lux.kit.Unit):
        if Queue.has_queue(unit):
            return unit.action_queue[0]
        else:
            return None

    def is_dig(a):
        if a is not None and len(a) > 0:
            return (a[0] == 3)
        else:
            return False

    def is_next_queue_dig(unit: lux.kit.Unit):
        return Queue.is_dig(Queue.next_action(unit))

    def is_transfer_ice(a):
        if a is not None and len(a) > 0:
            return (a[0] == 1 and a[2] == 0)
        else:
            return False

    def is_transfer_ore(a):
        if a is not None and len(a) > 0:
            return (a[0] == 1 and a[2] == 1)
        else:
            return False

    def is_move(a, direction=None):
        if a is not None and len(a) > 0:
            return (a[0] == 0 and (direction is None or a[1] == direction))
        else:
            return False

    def is_next_queue_transfer_ore(unit: lux.kit.Unit):
        return Queue.is_transfer_ore(Queue.next_action(unit))

    def is_next_queue_move(unit: lux.kit.Unit, direction=None):
        return Queue.is_move(Queue.next_action(unit), direction=direction)

    def is_next_queue_transfer_ice(unit: lux.kit.Unit):
        return Queue.is_transfer_ice(Queue.next_action(unit))

    def is_pickup(a):
        if a is not None and len(a) > 0:
            return (a[0] == 2)
        else:
            return False

    def is_next_queue_pickup(unit: lux.kit.Unit):
        return Queue.is_pickup(Queue.next_action(unit))

    def action_pickup_power(unit: lux.kit.Unit, power, repeat=False, n=1):
        if n> 1:
            return unit.pickup(4, power, repeat=repeat)
        else:
            return unit.pickup(4, power, repeat=repeat,n=n)

    def action_pickup_full_power(unit: lux.kit.Unit, repeat=False, n=1):
        return Queue.action_pickup_power(unit, unit.battery_capacity() - unit.power, repeat=repeat, n=n)

    def action_transfer_ice(unit: lux.kit.Unit, direction=0, amount=None):
        if amount is None: amount = unit.cargo.ice
        return unit.transfer(direction, 0, unit.cargo.ice, repeat=False)

    def action_transfer_ore(unit: lux.kit.Unit, direction=0, amount=None):
        if amount is None: amount = unit.cargo.ore
        return unit.transfer(direction, 1, unit.cargo.ore, repeat=False)

    def action_transfer_power(unit: lux.kit.Unit, direction, amount):
        return unit.transfer(direction, 4, amount, repeat=False)

    def real_cost_dig(unit: lux.kit.Unit):
        if not Queue.is_next_queue_dig(unit):
            return unit.action_queue_cost() + unit.dig_cost()
        else:
            return unit.dig_cost()


class Action_Queue():
    def __init__(self, game_state) -> None:
        self.game_state = game_state
        self.actions = dict()
        self.units_cannot_move = []

    def set_cannot_move(self, unit: lux.kit.Unit):
        self.units_cannot_move.append(unit.unit_id)

    def build_heavy(self, f):
        self.actions[f.unit_id] = f.build_heavy()

    def build_light(self, f):
        self.actions[f.unit_id] = f.build_light()

    def water(self, f):
        self.actions[f.unit_id] = f.water()

    def can_dig(self, unit: lux.kit.Unit):
        if not Queue.is_next_queue_dig(unit):
            # not already digging action, dig cost + cost action queue cost
            return unit.power >= (unit.dig_cost() + unit.action_queue_cost())
        else:
            # already digging action, only cost for dig
            return unit.power >= unit.dig_cost()

    def move_cost(self, unit, game_state, direction):
        if not Queue.is_next_queue_move(unit, direction):
            # not already moving there, move cost + cost action queue cost
            return unit._move_cost(game_state, direction, extra_cost=unit.action_queue_cost())
        else:
            # already moving there, only move cost
            return unit._move_cost(game_state, direction)

    def can_move(self, unit: lux.kit.Unit, game_state, direction=0):
        if unit.unit_id in self.units_cannot_move: return False
        if not Queue.is_next_queue_move(unit, direction):
            # not already moving there, move cost + cost action queue cost
            return unit._can_move_to(game_state, direction, extra_cost=unit.action_queue_cost())
        else:
            # already moving there, only move cost
            return unit._can_move_to(game_state, direction)

    def dropcargo_or_recharge(self, unit: lux.kit.Unit, force_recharge=False):
        do_recharge = unit.power < unit.battery_capacity() * 0.1 or force_recharge
        if unit.cargo.ore > 0 and not Queue.is_next_queue_transfer_ore(unit):
            if do_recharge:
                self.actions[unit.unit_id] = [Queue.action_transfer_ore(unit), Queue.action_pickup_full_power(unit, n=30)]
            else:
                self.actions[unit.unit_id] = [Queue.action_transfer_ore(unit)]
        elif unit.cargo.ice > 0 and not Queue.is_next_queue_transfer_ice(unit):
            if do_recharge:
                self.actions[unit.unit_id] = [Queue.action_transfer_ice(unit), Queue.action_pickup_full_power(unit, n=30)]
            else:
                self.actions[unit.unit_id] = [Queue.action_transfer_ice(unit)]
        elif unit.cargo.ice == 0 and unit.cargo.ore == 0 and do_recharge and not Queue.is_next_queue_pickup(unit):
            if not Queue.is_next_queue_pickup(unit):
                self.actions[unit.unit_id] = [Queue.action_pickup_full_power(unit, n=30)]

    def set_new_actions(self, unit, unit_actions, PREFIX):
        if len(unit_actions) > 0 and Queue.has_queue(unit):
            first_action = unit_actions[0]
            if (Queue.is_dig(first_action) and Queue.is_next_queue_dig(unit)):
                return
            if (Queue.is_move(first_action) and Queue.is_next_queue_move(unit)):
                if first_action[1] == unit.action_queue[0][1]:
                    return

        if len(unit_actions) > 20:
            # prx(PREFIX, "Actions too long", len(unit_actions), "truncating")
            unit_actions = unit_actions[:20]
        self.actions[unit.unit_id] = unit_actions

    def dig(self, unit):
        if not Queue.is_next_queue_dig(unit):
            # only dig if we are not already digging.
            self.actions[unit.unit_id] = [unit.dig(repeat=False, n=9999)]

    def transfer_ice(self, unit, direction=0, amount=None):
        self.actions[unit.unit_id] = [Queue.action_transfer_ice(unit,direction=direction, amount=amount)]

    def transfer_ore(self, unit, direction=0, amount=None):
        self.actions[unit.unit_id] = [Queue.action_transfer_ore(unit,direction=direction, amount=amount)]

    def transfer_energy(self, unit, direction, amount):
        self.actions[unit.unit_id] = [Queue.action_transfer_power(unit, direction, amount)]

    def move(self, unit, direction):
        if not Queue.is_next_queue_move(unit, direction):
            self.actions[unit.unit_id] = [unit.move(direction, repeat=False)]

    def clear_action(self, unit, PREFIX):
        if Queue.has_queue(unit):
            self.actions[unit.unit_id] = []

    def validate_actions(self, PREFIX, units, game_state, player_helper_me):


        unit_action_next = {}
        #build the complete list of next actions
        for unit_id, unit in iter(sorted(units.items())):
            if unit.unit_id in self.actions:
                if len(self.actions[unit.unit_id])>0:
                    unit_action_next[unit.unit_id] = self.actions[unit.unit_id][0]
                else:
                    unit_action_next[unit.unit_id] = None
            else:
                if Queue.has_queue(unit):
                    unit_action_next[unit.unit_id] = Queue.next_action(unit)

        # VALIDATE COLLISIONS
        move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
        unit_next_location = {} # key is location, payload unit id
        for unit_id, unit in iter(sorted(units.items())):
            new_pos = unit.pos_location()
            for u, a in unit_action_next.items():
                for direction in [1, 2, 3, 4]:
                    if Queue.is_move(a, direction):
                        new_pos = np.array(unit.pos) + move_deltas[direction]

            if (new_pos[0], new_pos[1]) in unit_next_location:
                pr(PREFIX,"TCFAIL", unit.unit_id, "clashes with ", unit_next_location[new_pos[0], new_pos[1]])
            unit_next_location[(new_pos[0], new_pos[1])] = unit.unit_id

        # VALIDATE DIG ON OUE OWN LICHEN
        for unit_id, unit in iter(sorted(units.items())):
            if unit_id in unit_action_next:
                a = unit_action_next[unit_id]
                if Queue.is_dig(a) and self.can_dig(unit):
                    if player_helper_me.get_lichen_amount(game_state, unit.pos):
                        pr(PREFIX, "TCFAIL", unit_id, " dig on our own lichen",unit.pos)
                        if unit.unit_id in self.actions:
                            pr(PREFIX, 'new action used', self.actions[unit.unit_id])
                        else:
                            pr(PREFIX, 'old action used', unit.action_queue)
                        # exit(0)



