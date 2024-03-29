import copy
import math

from numpy import dtype

import lux.unit
from action import *
from path_finder import *
from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import my_turn_to_place_factory
from playerhelper import PlayerHelper
from properties import Prop
from utils import *
import sys
import numpy as np
import networkx as nx


# create robot in the best location (especially first)
# do not bring ice to city that do not need
# do not harvest ice when all cities do not need water

def pr(*args, sep=' ', end='\n', force=False):  # print conditionally
    if (True or force):  # change first parameter to False to disable logging
        print(*args, sep=sep, file=sys.stderr)


def prx(*args): pr(*args, force=True)


def prc(*args):  # print conditionally
    if (False and (('xu_12' in args[0]) or ('u_8' in args[0]))):
        pr(*args, force=True)


class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        self.faction_names = {
            'player_0': 'TheBuilders',
            'player_1': 'FirstMars'
        }

        self.bot_factory = {}  # key unit id
        self.bots_task = {}  # key unit id
        self.bot_resource = {}  # key unit id

        self.factory_bots = {}  # key factory id
        self.factory_queue = {}  # key factory id
        self.move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
        self.built_robots = []

        self.G = nx.Graph()
        self.path_finder = Path_Finder(Prop.is_prod())

        self.me = PlayerHelper()
        self.him = PlayerHelper()

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        '''
        Early Phase
        '''

        actions = dict()
        if step == 0:
            # Declare faction
            actions['faction'] = self.faction_names[self.player]
            actions['bid'] = 0  # Learnable
        else:
            # Factory placement period
            # optionally convert observations to python objects with utility functions
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            opp_factories = [f.pos for _, f in game_state.factories[self.opp_player].items()]
            my_factories = [f.pos for _, f in game_state.factories[self.player].items()]

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will all location where it is possible to place a factory
                potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))

                ice_map = game_state.board.ice
                ore_map = game_state.board.ore
                ice_tile_locations = np.argwhere(ice_map == 1)  # numpy position of every ice tile
                ore_tile_locations = np.argwhere(ore_map == 1)  # numpy position of every ore tile

                # variable to hold the result in each loop
                best_loc = potential_spawns[0]
                chosen_params = ()

                # the radius around which we look for rubles
                DIST_RUBLE = 10
                WATER_TO_ALLOCATE = 300
                METAL_TO_ALLOCATE = 300

                min_dist = 10e6
                # loop each potential position
                for loc in potential_spawns:
                    # the array of ice and ore distances
                    ice_tile_distances, ice_positions = get_distance_vector_from_areas_given_center(loc,ice_tile_locations )
                    ore_tile_distances, ore_positions = get_distance_vector_from_areas_given_center(loc,ore_tile_locations )
                    ice_distance = np.min(ice_tile_distances)
                    ore_distance = np.min(ore_tile_distances)


                    # the density of ruble between the location and d_ruble
                    x = loc[0]
                    y = loc[1]

                    x1 = max(x - DIST_RUBLE, 0)
                    x2 = min(x + DIST_RUBLE, 47)
                    y1 = max(y - DIST_RUBLE, 0)
                    y2 = min(y + DIST_RUBLE, 47)

                    area_around_factory = obs["board"]["rubble"][
                                          x1:x2,
                                          y1:y2]
                    # zero the centre 3x3 where we put the factory
                    area_factory = obs["board"]["rubble"][max(x - 1, 0):min(x + 1, 47), max(y - 1, 0):max(y + 1, 47)]
                    sum_rubble_10x10 = np.sum(area_around_factory)
                    sum_rubble_3x3 = np.sum(area_factory)
                    area_size = (x2 - x1) * (y2 - y1) - 9
                    potential_lichen = area_size * 100 - (sum_rubble_10x10 - sum_rubble_3x3)
                    exploded_lichen = area_size * 100 - (sum_rubble_10x10 - sum_rubble_3x3 + 450)

                    # prx(x,y, area_around_factory)
                    # prx(x,y, area_factory)


                    closes_opp_factory_dist = 0
                    if len(opp_factories) >= 1:
                        closes_opp_factory_dist = np.min(get_distance_vector_from_areas_given_center(loc,opp_factories)[0])
                    closes_my_factory_dist = 0
                    if len(my_factories) >= 1:
                        closes_my_factory_dist = np.min(get_distance_vector_from_areas_given_center(loc,my_factories)[0])



                    kpi_build_factory = 0


                    kpi_build_factory = 0
                    remaining_lichen = potential_lichen
                    if water_left > 0:
                        #Normal KPIs
                        if closes_my_factory_dist < 20:
                            remaining_lichen = potential_lichen * (20. + closes_my_factory_dist) / 40.

                        # if we still have water, we crate meaningful factories
                        kpi_build_factory = (ice_distance * 10000.) \
                                            + (ore_distance * 1.) \
                                            - (remaining_lichen / 1000.)

                        if closes_opp_factory_dist < 20:
                            kpi_build_factory = kpi_build_factory - closes_opp_factory_dist / 5.
                        else:
                            kpi_build_factory = kpi_build_factory - 20./5.
                    else:
                        # water is zero. Create disruptive factory near the enemy
                        kpi_build_factory = closes_opp_factory_dist * closes_opp_factory_dist
                        if closes_my_factory_dist < 20:
                            kpi_build_factory = kpi_build_factory - (closes_my_factory_dist * closes_my_factory_dist) / 2.
                        else:
                            kpi_build_factory = kpi_build_factory - 200.

                        kpi_build_factory = kpi_build_factory - (exploded_lichen / 100.)


                    # pr(step, 'XXX', x, y, ice_distance, ore_distance, closes_opp_factory_dist, closes_my_factory_dist, area_size, sum_rubble_10x10,
                    #         sum_rubble_3x3, potential_lichen, remaining_lichen, kpi_build_factory)

                    if kpi_build_factory < min_dist:
                        min_dist = kpi_build_factory
                        best_loc = loc
                        chosen_params = ("ice=" + str(ice_distance),
                                         "ore=" + str(ore_distance),
                                         "potLic=" + str(potential_lichen),
                                         "effLic=" + str(remaining_lichen),
                                         "oppDis=" + str(closes_opp_factory_dist),
                                         "frdDis=" + str(closes_my_factory_dist),
                                         "kpi=" + str(kpi_build_factory.round(2))
                                         )

                # choose location that is the best according to the KPI above
                spawn_loc = best_loc
                actions['spawn'] = spawn_loc

                # we assign to each factory 300 or what is left (this means we build 3 factories with 300, 300, 150
                actions['metal'] = min(METAL_TO_ALLOCATE, metal_left)
                actions['water'] = min(WATER_TO_ALLOCATE, water_left)

                prx('Created factory in ', spawn_loc, '(', actions['water'], actions['metal'], ')based on ', chosen_params)

        return actions

    # TODO we should first loop on HEAVY, then lights

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        '''
        1. Regular Phase
        2. Building Robots
        '''

        game_state = obs_to_game_state(step, self.env_cfg, obs)
        self.path_finder.build_path(game_state, self.player, self.opp_player)
        actions = Action_Queue(game_state)

        turn = obs["real_env_steps"]

        # prx("---------Turn number ", turn)
        t_prefix = "T_" + str(turn)
        turn_left = 1001 - turn

        # unit positions
        self.me.set_player(game_state, self.player)
        self.him.set_player(game_state, self.opp_player)
        units = game_state.units[self.player]

        # factories variable
        factories = game_state.factories[self.player]
        factory_tiles, factory_ids = [], []
        self.built_robots = []

        # check if any unit has died
        for bot in list(self.bots_task.keys()):
            if bot not in units:

                try:
                    unit_factory = self.bot_factory[bot]
                    unit_task = self.bots_task[bot]
                    prx(t_prefix, bot, "has died.. task was", unit_task, unit_factory)
                    # remove from dictionaries
                    self.bots_task.pop(bot)
                    if bot in self.bot_factory: self.bot_factory.pop(bot)
                    if bot in self.bot_resource: self.bot_resource.pop(bot)
                    if unit_task in ['ice', 'ore', 'rubble']:
                        # the below make sure we create a new one
                        if bot in self.factory_bots[unit_factory][unit_task]:
                            self.factory_bots[unit_factory][unit_task].remove(bot)
                except Exception as err:
                    prx(t_prefix, bot, "has died.. TCFAIL ON EXCEPTION:", err)

        # FACTORY LOOP
        for factory_id, factory in factories.items():

            if factory_id not in self.factory_bots.keys():
                prx(t_prefix, factory_id, " new factory")
                self.factory_bots[factory_id] = {
                    'ice': [],
                    'ore': [],
                    'rubble': [],
                    'kill': [],
                }

                self.factory_queue[factory_id] = []

            factory_tiles += [factory.pos]
            factory_ids += [factory_id]

        factory_tiles = np.array(factory_tiles)  # Factory locations (to go back to)

        # Resource map and locations
        ice_map = game_state.board.ice
        ore_map = game_state.board.ore
        rubble_map = game_state.board.rubble

        ice_locations_all = np.argwhere(ice_map >= 1)  # numpy position of every ice tile
        ore_locations_all = np.argwhere(ore_map >= 1)  # numpy position of every ore tile
        rubble_locations_all = np.argwhere(rubble_map >= 1)  # numpy position of every rubble tile

        ice_locations = ice_locations_all
        ore_locations = ore_locations_all
        rubble_locations = rubble_locations_all

        # x = len(rubble_locations)
        rubble_and_opposite_lichen_locations = np.vstack((rubble_locations, self.him.lichen_locations))
        # prx(t_prefix, 'rubble_and_opposite_lichen_locations', len(rubble_and_opposite_lichen_locations), '=', x ,'+', len(self.him.lichen_locations) )

        # remove exausted locations
        for unit_id, unit in iter(sorted(units.items())):
            if unit_id in self.bots_task:
                if self.bots_task[unit_id] == 'rubble':
                    if unit_id in self.bot_resource:
                        old_loc = self.bot_resource[unit_id]
                        rubble = self.get_rubble_amount(game_state, old_loc)
                        lichen = self.him.get_lichen_amount(game_state, old_loc)
                        if rubble == 0 and lichen == 0:
                            prx(t_prefix, unit_id, "old resource", old_loc, 'exhausted')
                            self.bot_resource.pop(unit_id)

        #TEAM work
        enable_transfer = True
        heavy_to_avoid = []
        for unit_id, task in self.bots_task.items():
            if task not in ['ice', 'ore']:
                continue
            if unit_id not in units or unit_id not in self.bot_resource:
                pr("TCFAIL 306", unit_id, 'not in units')
                continue
            unit = units[unit_id]
            unit_on_resource = unit.pos_location() == self.bot_resource[unit_id]
            if unit_on_resource:
                # heavy tends to stay on a location for long time, we can exclude from possible paths
                if unit.is_heavy():
                    # prx(t_prefix,"heavy on resource, excluding from path",unit._to_string())
                    heavy_to_avoid.append(unit.pos_location())

            if enable_transfer and (unit.cargo.ore + unit.cargo.ice > 0):
                for friend_id, friend_task in self.bots_task.items():
                    if friend_task != task:
                        continue
                    if friend_id not in units or friend_id not in self.bot_resource:
                        pr("TCFAIL 312", unit_id, 'not in units')
                        continue
                    friend = units[friend_id]
                    if friend_id != unit_id and self.bot_resource[friend_id] == self.bot_resource[unit_id] \
                            and get_distance(friend.pos_location(), unit.pos_location()) == 1 and \
                            (unit.pos_location() == self.bot_resource[unit_id] or
                             (
                                     np.min(get_distance_vector(friend.pos, self.me.get_factories_centers())) < np.min(get_distance_vector(unit.pos, self.me.get_factories_centers()))
                             )):
                        # TODO CHECK IF ENEMY CLOSE
                        transferred_cargo = False
                        transferred_power = False
                        prx(t_prefix, "share the same resource, adjacent", unit.to_string2(), "|", friend.to_string2())

                        if friend.cargo_space_left() > 0 and unit.cargo.total() > 0 and \
                                (unit.cargo.total() > max(friend.cargo.total(), unit.cargo_space()/20) or not unit_on_resource):
                            direction_unit_to_friend, m = get_straight_direction(unit, friend.pos)
                            if unit.cargo.ice > 0:
                                amount = min(unit.cargo.ice, friend.cargo_space_left())
                                actions.transfer_ice(unit, direction_unit_to_friend, amount)
                                prx(t_prefix, "transferring", amount,"ice from", unit.to_string2(), "to", friend.to_string2())
                            elif unit.cargo.ore > 0:
                                amount = min(unit.cargo.ore, friend.cargo_space_left())
                                actions.transfer_ore(unit, direction_unit_to_friend, amount)
                                prx(t_prefix, "transferring", amount, "ore from", unit.to_string2(), "to", friend.to_string2())
                            transferred_cargo = True
                        if unit.battery_capacity() > 0 and friend.power > friend.battery_capacity() / 50:
                            direction_friend_to_unit, m = get_straight_direction(friend, unit.pos)
                            amount = min(friend.power, unit.battery_capacity())
                            actions.transfer_energy(friend, direction_friend_to_unit, amount)
                            prx(t_prefix, "transferring", amount,"power from", unit.to_string2(), "to", friend.to_string2())
                            transferred_power = True

                        if transferred_power and not transferred_cargo:
                            actions.set_cannot_move(unit)
                        elif transferred_cargo and not transferred_power:
                            actions.set_cannot_move(friend)







        # UNIT LOOP
        for unit_id, unit in iter(sorted(units.items())):

            #already in team actions
            if unit_id in actions.actions:
                continue

            PREFIX = t_prefix + " " + unit.to_string()
            # prc(PREFIX)
            if unit_id not in self.bots_task.keys():
                self.bots_task[unit_id] = ''

            distance_to_closest_opponent, distance_to_closest_opponent_heavy, distance_to_closest_opponent_light = 10e6, 10e6, 10e6
            if self.him.get_num_units() != 0:
                opp_pos = np.array(list(self.him.get_unit_positions()), dtype=dtype)
                opponent_unit_distances, distance_to_closest_opponent, opponent_pos_min_distance = self.get_distances_info(unit.pos, opp_pos)

            if self.him.get_num_heavy() != 0:
                opp_heavy_pos = np.array(list(self.him.get_heavy_positions()), dtype=dtype)
                opponent_heavy_unit_distances, distance_to_closest_opponent_heavy, opponent_heavy_pos_min_distance = self.get_distances_info(unit.pos,
                                                                                                                                             opp_heavy_pos)

            if self.him.get_num_lights() != 0:
                opp_light_pos = np.array(list(self.him.get_light_positions()), dtype=dtype)
                opponent_light_unit_distances, distance_to_closest_opponent_light, opponent_light_pos_min_distance = self.get_distances_info(unit.pos,
                                                                                                                                             opp_light_pos)

            on_factory = unit.pos_location() in self.me.get_factories_areas()
            factory_min_distance = 10000
            if len(factory_tiles) > 0:
                factory_unit_distances, distance_to_factory, closest_factory_area = self.get_distances_info(unit.pos, self.me.get_factories_areas())
                factory_distances, min_index, closest_factory_center = self.get_distances_info(unit.pos, factory_tiles)

                # unit not yet assigned to a factory
                if unit_id not in self.bot_factory.keys() \
                        or self.bot_factory[unit_id] not in factory_ids:
                    # assign this unit to closest factory (presumably where it has been created_
                    min_index = np.argmin(factory_distances)
                    self.bot_factory[unit_id] = factory_ids[min_index]
                else:
                    closest_factory_center = factories[self.bot_factory[unit_id]].pos

            unit_factory = self.bot_factory[unit_id]

            # UNIT TASK AND RESOURCE DECISION
            if unit.power < unit.action_queue_cost():
                continue

            if len(factory_tiles) > 0:

                direction = 0

                # Assigning task for the bot
                if self.bots_task[unit_id] == '':
                    task = 'ice'

                    this_factory_queue = self.factory_queue[unit_factory]
                    if len(this_factory_queue) != 0:
                        prx(PREFIX, "QUEUE", this_factory_queue)
                        task = this_factory_queue.pop(0)

                    prx(PREFIX, 'from', unit_factory, unit.unit_type, 'assigned task', task, ' queue len ', len(this_factory_queue))
                    # if task =='kill' and unit.is_light():
                    #     prx(PREFIX, 'Cannot get a light killer! Rubble instead')
                    #     task = 'rubble'

                    self.bots_task[unit_id] = task
                    self.factory_bots[unit_factory][task].append(unit_id)

                # assign a resource to this bot
                if unit_id not in self.bot_resource:
                    unit_task = self.bots_task[unit_id]
                    if unit_task in ['ore', 'ice']:
                        if unit_task == 'ore':
                            c, sorted_resources_to_factory = get_map_distances(ore_locations, unit.pos)
                            # using unit.pos as and not factory.pos, equivalent on spawn
                        elif unit_task == 'ice':
                            c, sorted_resources_to_factory = get_map_distances(ice_locations, unit.pos)

                        for resource in sorted_resources_to_factory:
                            # ORE AND ICE
                            resource_location = (resource[0], resource[1])
                            bots_already = list(self.bot_resource.values()).count(resource_location)
                            # dis = unit.get_distance(resource_location)
                            a, dis, c = self.get_distances_info(np.array(resource_location), self.me.get_factories_areas())
                            if bots_already == 0 \
                                    or (bots_already <= 1 and dis > 4) \
                                    or (bots_already <= 2 and dis > 6):
                                self.bot_resource[unit_id] = resource_location
                                prx(PREFIX, unit_factory, 'Assigning resource', unit_task, resource_location, 'dis=', dis, 'units here', bots_already + 1)
                                break

                    elif unit_task in ['rubble']:
                        if unit_task == 'rubble':
                            distances_to_unit = get_distance_vector(unit.pos, rubble_and_opposite_lichen_locations)
                            distances_to_center = get_distance_vector(factories[unit_factory].pos, rubble_and_opposite_lichen_locations)
                            rubbles, distance_lichens = [], []
                            i = -1
                            for x in rubble_and_opposite_lichen_locations:
                                i = i + 1
                                r = self.get_rubble_amount(game_state, x)
                                rubbles.append(r)

                                if len(self.me.lichen_locations) > 0:
                                    distances_to_lichen = get_distance_vector(x, self.me.lichen_locations)
                                    min_distances_to_lichen = np.min(distances_to_lichen)
                                    if min_distances_to_lichen < distances_to_center[i]:
                                        distances_to_center[i] = min_distances_to_lichen

                            distances_kpi = (4 * distances_to_center + distances_to_unit) + np.array(rubbles)/15
                            sorted_loc = [rubble_and_opposite_lichen_locations[k] for k in np.argsort(distances_kpi)]
                            # pr(t_prefix, 'XXX distances_to_unit',distances_to_unit)
                            # pr(t_prefix, 'XXX distances_to_center',distances_to_center)
                            # pr(t_prefix, 'XXX rubble',rubbles)

                        for resource in sorted_loc:
                            # RUBBLE + OPPOSITE LICHEN
                            resource_location = (resource[0], resource[1])
                            bots_already = list(self.bot_resource.values()).count(resource_location)
                            rubble = self.get_rubble_amount(game_state, resource_location)
                            lichen = self.him.get_lichen_amount(game_state, resource_location)

                            # dis = unit.get_distance(resource_location)
                            a, dis, c = self.get_distances_info(np.array(resource_location), self.me.get_factories_areas())
                            if bots_already == 0:
                                self.bot_resource[unit_id] = resource_location
                                prx(PREFIX, unit_factory, 'Assigning resource', unit_task, resource_location, 'dis=', dis,
                                    'units here', bots_already + 1, 'rubble=', rubble, 'lichen=', lichen)

                                break

                    if unit_task != 'kill' and unit_id not in self.bot_resource:
                        prx(PREFIX, "Could not find resource for this unit", unit_task)

                # become aggressive if you need to
                assigned_task = self.bots_task[unit_id]
                if assigned_task != "kill":
                    if unit.is_heavy():
                        if distance_to_closest_opponent_heavy <= 2:
                            assigned_task = "kill"
                            target = opponent_heavy_pos_min_distance
                        elif distance_to_closest_opponent_light == 1:
                            enemy = self.him.get_unit_from_current_position(opponent_light_pos_min_distance)
                            if enemy.power < enemy.unit_cfg.MOVE_COST * 2:
                                assigned_task = "kill"
                                target = opponent_light_pos_min_distance
                    else:
                        if (self.him.get_num_lights() != 0 and distance_to_closest_opponent_light == 1):
                            assigned_task = "kill"
                            target = opponent_light_pos_min_distance

                    if assigned_task == "kill":
                        prx(PREFIX, unit.unit_id, 'from', unit_factory, unit.unit_type, unit.pos, 'temporarily tasked as', assigned_task, target)

                # reconvert ore to rubble at the end.
                if turn_left < 200 and assigned_task == "ore":
                    self.bots_task[unit_id] = 'rubble'
                    prx(PREFIX, unit_factory, unit.unit_type, unit.pos, 'permanently tasked from', assigned_task,
                        'to', self.bots_task[unit_id])
                    assigned_task = self.bots_task[unit_id]

                # position to avoid, start with the heavy to avoid, add adjacents
                positions_to_avoid = copy.copy(heavy_to_avoid)
                if unit.pos_location() in positions_to_avoid:
                    positions_to_avoid.remove(unit.pos_location())
                for p in self.me.get_unit_next_positions():
                    if get_distance(unit.pos_location(), p) == 1:
                        positions_to_avoid.append(p)
                    # if len(positions_to_avoid)>0:
                    #     prx(PREFIX," need to avoid first moves to ", positions_to_avoid)

                # if light, avoid close heavy
                if unit.is_light():
                    for p in self.him.get_heavy_positions():
                        if get_distance(unit.pos_location(), p) == 1:
                            positions_to_avoid.append(p)

                prc(PREFIX, unit.pos, 'task=' + assigned_task, unit.cargo)
                if assigned_task == "ice":
                    cost_home = self.get_cost_to(game_state, unit, turn, positions_to_avoid, closest_factory_area)
                    recharge_power = if_is_day(turn + 1, unit.charge_per_turn(), 0)
                    on_ice = self.on_ice(game_state, unit.pos)

                    # prc(PREFIX,'on ice', on_ice)
                    # prc(PREFIX, '1on ice', (on_ice and unit.cargo.ice < unit.cargo_space()) or ((not on_ice) and unit.cargo.ice == 0))
                    # prc(PREFIX, '2on ice', (unit.power + recharge_power > Queue.real_cost_dig(unit) + cost_home and actions.can_dig(unit)))

                    if ((on_ice and unit.cargo.ice < unit.cargo_space()) or ((not on_ice) and unit.cargo.ice == 0)) \
                            and ((unit.power + recharge_power > Queue.real_cost_dig(unit) + cost_home and actions.can_dig(unit)) \
                                or (not on_factory and unit.cargo.ice == 0 and cost_home > 4 * unit.dig_cost()) \
                                or (not on_factory and unit.cargo.ice < unit.cargo_space() / 2 and cost_home > 6 * unit.dig_cost()) \
                            ):

                        prc(PREFIX, 'unit.power + recharge_power > Queue.real_cost_dig(unit) + cost_home', unit.power, recharge_power, Queue.real_cost_dig(unit), cost_home)
                        self.dig_or_go_to_resouce(PREFIX, actions, game_state, positions_to_avoid, turn, unit, rubble_and_opposite_lichen_locations,
                                                  ice_locations, 'ice', drop_ice=True)

                    else:
                        if on_factory:
                            actions.dropcargo_or_recharge(unit)
                        else:
                            # GO HOME
                            self.send_unit_home(PREFIX, game_state, actions, positions_to_avoid, closest_factory_center, closest_factory_area, turn, unit)
                    continue

                elif assigned_task == 'ore':
                    # prx(PREFIX, "Looking for ore")
                    cost_home = self.get_cost_to(game_state, unit, turn, positions_to_avoid, closest_factory_area, PREFIX=PREFIX)
                    recharge_power = if_is_day(turn + 1, unit.charge_per_turn(), 0)
                    on_ore = self.on_ore(game_state, unit.pos)

                    # prc(PREFIX,'on ore', on_ore)
                    # prc(PREFIX, '1on ore', (on_ore and unit.cargo.ore < unit.cargo_space()) or ((not on_ore) and unit.cargo.ore == 0))
                    # prc(PREFIX, '2on ore', (unit.power + recharge_power > Queue.real_cost_dig(unit) + cost_home and actions.can_dig(unit)))

                    if ((on_ore and unit.cargo.ore < unit.cargo_space()) or ((not on_ore) and unit.cargo.ore == 0)) \
                            and ((unit.power + recharge_power > Queue.real_cost_dig(unit) + cost_home and actions.can_dig(unit)) \
                                or (not on_factory and unit.cargo.ore == 0 and cost_home > 4 * unit.dig_cost()) \
                                or (not on_factory and unit.cargo.ore < unit.cargo_space() / 2 and cost_home > 6 * unit.dig_cost()) \
                                or (not on_factory and unit.is_heavy() and unit.cargo.ore < 250 ) \
                                or (actions.can_dig(unit) and not actions.can_move(unit,game_state) ) \
                            ):
                        prc(PREFIX, "dig_or_go_to_resouce, cargo full =", unit.cargo.ore < unit.cargo_space(), ", estimate power=", unit.power + recharge_power,
                            'dig cost', Queue.real_cost_dig(unit), 'cost home=', cost_home)
                        self.dig_or_go_to_resouce(PREFIX, actions, game_state, positions_to_avoid, turn, unit, rubble_and_opposite_lichen_locations,
                                                  ore_locations, 'ore', drop_ore=True)

                    else:
                        if on_factory:
                            prc(PREFIX, "on base dropcargo_or_recharge")
                            actions.dropcargo_or_recharge(unit)
                        else:
                            # GO HOME
                            prc(PREFIX, "Going home, cargo full =", unit.cargo.ore < unit.cargo_space(), ", estimate power=", unit.power + recharge_power,
                                'dig cost', Queue.real_cost_dig(unit), 'cost home=', cost_home)
                            self.send_unit_home(PREFIX, game_state, actions, positions_to_avoid, closest_factory_center, closest_factory_area, turn, unit)
                    continue

                # RUBBLE
                elif assigned_task == 'rubble':
                    if actions.can_dig(unit) and np.all(self.bot_resource[unit.unit_id] == unit.pos) \
                            and (self.get_rubble_amount(game_state, unit.pos_location()) > 0 or self.him.get_lichen_amount(game_state,unit.pos) > 0):
                        prc(PREFIX, "can dig, on target ruble")
                        actions.dig(unit)
                        continue
                    elif unit.power > unit.action_queue_cost() + unit.dig_cost() + unit.rubble_dig_hurdle():
                        # if actions.can_dig(unit):
                        # compute the distance to each rubble tile from this unit and pick the closest
                        closest_rubble, sorted_rubble = get_map_distances(rubble_and_opposite_lichen_locations, unit.pos)

                        direction = 0
                        if unit.unit_id in self.bot_resource:
                            resource = self.bot_resource[unit.unit_id]
                            prc(PREFIX, "Looking for rubble actively on assigned resource", resource)
                            direction, unit_actions, new_pos, num_digs, num_steps, cost = \
                                self.get_complete_path(game_state, unit, turn, positions_to_avoid, resource, PREFIX, one_way_only_and_dig=True)
                            if np.all(resource != unit.pos) and direction == 0:
                                prc(PREFIX, "assigned resource non reachable")
                        else:
                            # get closest resource
                            resource = closest_rubble
                            prc(PREFIX, "Looking for rubble actively on closest resource", resource)

                        # if we have reached the rubble tile, start mining if possible
                        if np.all(resource == unit.pos):
                            prc(PREFIX, "can dig, on ruble")
                            if actions.can_dig(unit):
                                prc(PREFIX, "dig ruble or lichen")
                                actions.dig(unit)
                                continue

                        else:
                            prc(PREFIX, "can dig, not on ruble, move to next ruble")
                            if len(rubble_locations) != 0:
                                # try first assigned resource

                                # if the assigned resource is not reachable
                                if direction == 0:
                                    prc(PREFIX, "assigned resource non reachable")
                                    best_path = None
                                    max_range = unit.get_distance(closest_rubble) + 3
                                    for closest in sorted_rubble:
                                        if best_path is not None:
                                            if unit.get_distance(closest) > max_range:
                                                break
                                            # performance shortcut, steps * min_cost = cost, then this is the perfect path, we can exit early
                                            if best_path[4] == unit.get_distance(closest_rubble) and best_path[4] * unit.unit_cfg.MOVE_COST == best_path[5]:
                                                # prx(PREFIX, "Found perfect path ", (best_path[5], best_path[4])," to ",closest)
                                                break
                                        # direction, unit_actions, new_pos, num_digs, num_steps, cost
                                        path = self.get_complete_path(game_state, unit, turn, positions_to_avoid, closest, PREFIX,
                                                                      one_way_only_and_dig=True)
                                        this_direction, a, b, c, this_steps, this_cost = path
                                        if this_direction != 0:
                                            if best_path is None:
                                                best_path = path
                                            elif (this_cost, this_steps) < (best_path[5], best_path[4]):
                                                # prx(PREFIX, "Chosen alternative path", (this_cost, this_steps), "instead of old",(best_path[5], best_path[4])," to ",closest)
                                                best_path = path

                                    if best_path is not None:
                                        direction, unit_actions, new_pos, num_digs, num_steps, cost = best_path

                                if direction == 0:
                                    actions.clear_action(unit, PREFIX)
                                    continue

                                elif direction != 0:
                                    actions.set_new_actions(unit, unit_actions, PREFIX)
                                    self.me.set_unit_next_position(unit.unit_id, new_pos)
                                    # prx(PREFIX, "set next position ", new_pos)
                                    continue

                    else:
                        # prc(PREFIX, "cannot dig, adjacent")
                        if on_factory:
                            prc(PREFIX, "on factory recharge")
                            actions.dropcargo_or_recharge(unit)
                        else:
                            if unit.get_distance(closest_factory_area) > 3:
                                prc(PREFIX, "cannot dig, but too far away from home, stay")
                                actions.clear_action(unit, PREFIX)
                                continue
                            # GO HOME
                            self.send_unit_home(PREFIX, game_state, actions, positions_to_avoid, closest_factory_center, closest_factory_area, turn, unit)
                            continue

                elif assigned_task == 'kill':
                    if on_factory and (unit.cargo.total() > 0):
                        prc(PREFIX, "on base dropcargo_or_recharge")
                        actions.dropcargo_or_recharge(unit)
                        continue

                    # conditions in which we look for lichen rather than going after enemies
                    if (unit.is_heavy() and self.him.get_num_units() == 0) or \
                            (unit.is_heavy() and self.him.get_num_lights() == 0 and distance_to_closest_opponent_heavy > 2) or \
                            (unit.is_heavy() and distance_to_closest_opponent_heavy > 2 and self.him.is_factory_area(opponent_light_pos_min_distance)) or \
                            (unit.is_light() and self.him.get_num_lights() == 0) or \
                            (distance_to_closest_opponent > 2 and self.him.get_lichen_amount(game_state, unit.pos) > 0):

                        # no enemy we can kill
                        prc(PREFIX, "Kill no enemy to kill", 'heavy=', self.him.get_num_heavy(), 'lights=', self.him.get_num_lights())
                        if len(self.him.lichen_locations) > 0:
                            self.dig_or_go_to_resouce(PREFIX, actions, game_state, positions_to_avoid, turn, unit, [], self.him.lichen_locations,
                                                      'opponent lichen')
                            continue

                    else:
                        # TODO probably needs to chose a target not inside a city, because if it does it is not moving

                        if unit.is_heavy():
                            if self.him.get_num_heavy() > 0 and \
                                    ((distance_to_closest_opponent_heavy <= 2 or self.him.get_num_lights() == 0) or \
                                     (distance_to_closest_opponent_heavy <= 6 and distance_to_closest_opponent_light > 2 * distance_to_closest_opponent_heavy)
                                    ):
                                # if a heavy is close by, engage with him. Heavy vs heavy
                                target = (opponent_heavy_pos_min_distance[0], opponent_heavy_pos_min_distance[1])
                                distance = distance_to_closest_opponent_heavy
                            else:
                                # heavy first try to get lights
                                target = (opponent_light_pos_min_distance[0], opponent_light_pos_min_distance[1])
                                distance = distance_to_closest_opponent
                        else:
                            # lights only try to kill light
                            target = (opponent_light_pos_min_distance[0], opponent_light_pos_min_distance[1])
                            distance = distance_to_closest_opponent

                        enemy: lux.unit.Unit = self.him.get_unit_from_current_position(target)
                        prc(PREFIX, "Kill", target, enemy.unit_id, enemy.unit_type, enemy.power, 'distance', distance)
                        if on_factory and distance > 2:
                            direction, new_pos = get_straight_direction(unit, target)
                            prc(PREFIX, "Kill, on factory, going", direction, new_pos)
                            prc(PREFIX, 'position to avoid', positions_to_avoid)
                            direction = self.get_direction(game_state, unit, positions_to_avoid, new_pos)
                        elif distance == 1:
                            # NEXT TO TARGET
                            direction = self.get_direction(game_state, unit, positions_to_avoid, target)
                            prc(PREFIX, "Kill, next to target, going", direction)
                            if direction == 0:
                                # GO HOME
                                prc(PREFIX, "Kill, next to target, abort, going home")
                                self.send_unit_home(PREFIX, game_state, actions, positions_to_avoid, closest_factory_center, closest_factory_area, turn, unit)
                                continue
                            elif (enemy.power > unit.power and (unit.unit_type == enemy.unit_type)) or not actions.can_move(unit, game_state, direction):
                                # if they are both same size and he is stronger, back off
                                prc(PREFIX, "Kill, next to target, abort, he has more power:", enemy.power, ' > ', unit.power,
                                    ' or I cannot move there, can_move=', actions.can_move(unit, game_state, direction))
                                positions_to_avoid.append(enemy.pos_location())
                                direction, new_pos, unit_actions = \
                                    self.send_unit_home(PREFIX, game_state, actions, positions_to_avoid, closest_factory_center, closest_factory_area, turn,
                                                        unit)
                                continue
                            else:
                                pass
                                # going to enemy, do not continue here
                        elif distance == 2:
                            # DISTANCE 2 to enemy
                            direction = self.get_direction(game_state, unit, positions_to_avoid, target)
                            if get_next_pos(unit.pos, direction) in self.me.get_factories_areas():
                                prc(PREFIX, 'Enemy distance 2, but next step is on factory area, we cannot be smashed')
                                pass
                            elif enemy.power > unit.power - actions.move_cost(unit, game_state, direction) and (unit.unit_type == enemy.unit_type):
                                prc(PREFIX, 'Enemy distance 2, and has more power, stand still')
                                actions.clear_action(unit, PREFIX)
                                self.me.set_unit_next_position(unit_id, unit.pos_location())
                                continue
                            else:
                                pass
                                # going to enemy, do not continue here
                        else:
                            # DISTANCE greater than 2
                            if unit.power > unit.action_queue_cost():
                                prc(PREFIX, "Kill, Seeking enemy", np.array(target))
                                direction, new_pos, unit_actions = self.go_to_target(PREFIX, game_state, actions, positions_to_avoid, turn, unit,
                                                                                     np.array(target))
                            else:
                                if on_factory:
                                    actions.dropcargo_or_recharge(unit)
                                # else:
                                #     # GO HOME
                                #     prc(PREFIX, "Kill, going home")
                                #     self.send_unit_home(PREFIX, game_state, actions, positions_to_avoid, closest_factory_center, closest_factory_area, turn, unit)
                            continue

                        prc(PREFIX, "Kill, direction, resolved in direction ", direction)

                if direction == 0:
                    # prc(PREFIX, 'cannot find a path')
                    closest_center, sorted_centers = get_map_distances(factory_tiles, unit.pos)
                    if np.all(closest_center == unit.pos):
                        prx(PREFIX, 'Should not stay here on a center factory, can kill my friends..', unit.pos)
                        direction = self.get_random_direction(unit, PREFIX)
                        prx(PREFIX, 'Random got ', direction)

                # check move_cost is not None, meaning that direction is not blocked
                # check if unit has enough power to move and update the action queue.

                if direction != 0 and actions.can_move(unit, game_state, direction):
                    actions.move(unit, direction)
                    # new position
                    new_pos = np.array(unit.pos) + self.move_deltas[direction]
                    prc(PREFIX, 'move to ', direction, 'clashing with next positions=', (new_pos[0], new_pos[1]) in self.me.get_unit_next_positions())
                    self.me.set_unit_next_position(unit.unit_id, new_pos)
                else:
                    # not moving
                    prc(PREFIX, 'Not moving, remove node ', unit.pos, unit.pos_location() in self.me.get_unit_next_positions())
                    actions.clear_action(unit, PREFIX)
                    self.me.set_unit_next_position(unit_id, unit.pos_location())

        # FACTORY LOOP
        for factory_id, factory in factories.items():

            built = False
            if factory.can_build_light(game_state):

                new_task = None
                min_bots = {
                    'ice': 1,
                    'ore': 5,
                    'rubble': 5,
                    'kill': 1
                }

                # NO. BOTS PER TASK
                for task in ['ice', 'kill', 'ore', 'rubble']:
                    num_bots = sum([task in self.factory_queue[factory_id]])
                    for id in self.factory_bots[factory_id][task]:
                        if id in units and units[id].is_heavy():
                            num_bots = num_bots + 5
                        else:
                            num_bots = num_bots + 1
                    if num_bots < min_bots[task]:
                        prx(t_prefix, factory_id, "We have less bots(", num_bots, ") for", task, " than min", min_bots[task])
                        if task == 'kill' and not factory.can_build_heavy(game_state):
                            # do not build kill lights
                            prx(t_prefix, factory_id, "Cannot build a light kill, power=", factory.power, ", metal=", factory.cargo.metal)
                            continue
                        new_task = task
                        break

                if new_task is None and factory.can_build_heavy(game_state) and turn_left < 900:
                    prx(t_prefix, factory_id, "We have enough to build a Kill, light enemy (", self.him.get_num_lights())
                    if self.him.get_num_lights() > 5:
                        prx(t_prefix, factory_id, "build a new kill")
                        new_task = 'kill'
                    else:
                        prx(t_prefix, factory_id, "build a new ice")
                        new_task = 'ice'

                # toward the end of the game, build as many as rubble collector as you can
                elif turn_left < 250:
                    new_task = 'rubble'

                # BUILD ROBOT ENTRY POINT
                if new_task is not None:
                    # Check we are not building on top of another unit
                    if (factory.pos[0], factory.pos[1]) in self.me.get_unit_next_positions():
                        pr(t_prefix, factory.unit_id, "Cannot build robot, already an unit present", factory.pos)
                        continue

                    ore_ajacent = False
                    if new_task in ['kill', 'ore'] and factory.can_build_heavy(game_state):
                        # Check if there is one Ore ajacent
                        ore_tile_distances, ore_positions = get_distance_vector_from_areas_given_center(factory.pos_location(), np.argwhere(ore_map == 1))
                        i = 0
                        for ore_dist in ore_tile_distances:
                            if ore_dist == 1:
                                ore_loc = ore_positions[i]
                                # pr(type(ore_loc), ore_loc)
                                # pr(type( self.bot_resource.values()),  self.bot_resource.values())
                                if ore_loc not in self.bot_resource.values():
                                    pr(t_prefix, factory_id, factory.pos_location(), 'Ore is adjacent, can build heavy', ore_loc, ore_dist)
                                    ore_ajacent = True
                                    break
                            i = i + 1
                    if ore_ajacent and new_task == 'kill':
                        pr(t_prefix, factory_id, factory.pos_location(), 'Ore is adjacent, kill converted to ore')
                        new_task = 'ore'

                    if new_task in ['ice', 'kill']:
                        if factory.can_build_heavy(game_state):
                            self.build_heavy_robot(actions, factory, t_prefix, new_task)
                        elif factory.can_build_light(game_state):
                            self.build_light_robot(actions, factory, t_prefix, new_task)
                    elif ore_ajacent and new_task in ['ore'] and factory.can_build_heavy(game_state):
                        self.build_heavy_robot(actions, factory, t_prefix, new_task)
                    else:
                        if factory.can_build_light(game_state):
                            self.build_light_robot(actions, factory, t_prefix, new_task)

                    built = True

                    if factory_id not in self.factory_queue.keys():
                        self.factory_queue[factory_id] = [new_task]
                        prx(t_prefix, "set id ", factory_id, ' to ', self.factory_queue[factory_id])
                    else:
                        self.factory_queue[factory_id].append(new_task)
                        # prx(t_prefix, "append id ", factory_id, ' to ', self.factory_queue[factory_id])

            # if we have excess water use to grow lichen
            if not built:
                if (factory.cargo.water - factory.water_cost(game_state)) > 1:
                    # at the end, we start water if we can
                    estimated_water = factory.cargo.water + min(math.floor(factory.cargo.ice / 4), turn_left * 25)
                    if turn_left < 10 and \
                            estimated_water - factory.water_cost(game_state) > turn_left:
                        # prx(t_prefix, 'water', factory_id, "water=", factory.cargo.water, "ice=", factory.cargo.ice, "cost=", factory.water_cost(game_state),"left=", turn_left)
                        actions.water(factory)

                    # anyway, we start water if we have resource to water till the end
                    elif estimated_water > turn_left * max(1, (1 + factory.water_cost(game_state))):
                        # prx(t_prefix, 'water', factory_id, "water=", factory.cargo.water, "ice=", factory.cargo.ice, "cost=", factory.water_cost(game_state), "left=", turn_left)
                        actions.water(factory)

                    elif True and factory.water_cost(game_state)> 4 and estimated_water > 50 + 20 * max(1, (1 + factory.water_cost(game_state))):
                        prx(t_prefix, 'water boost', factory_id, "water=", factory.cargo.water, "ice=", factory.cargo.ice, "cost=", factory.water_cost(game_state), "left=", turn_left)
                        actions.water(factory)

        # if turn==205:
        #     prx(t_prefix,"next_position =====",self.unit_next_positions)
        # if turn==18:
        #     a=5/0.
        if Prop.is_local():
            actions.validate_actions(t_prefix, units, game_state, self.me)
        return actions.actions

    def get_rubble_amount(self, game_state, pos):
        return game_state.board.rubble[pos[0], pos[1]]

    def get_ice_amount(self, game_state, pos):
        return game_state.board.ice[pos[0], pos[1]]

    def get_ore_amount(self, game_state, pos):
        return game_state.board.ore[pos[0], pos[1]]

    def on_rubble(self,game_state,pos):
        return self.get_rubble_amount(game_state,pos)>0

    def on_ice(self, game_state, pos):
        return self.get_ice_amount(game_state, pos) > 0

    def on_ore(self, game_state, pos):
        return self.get_ore_amount(game_state, pos) > 0

    def get_distances_info(self, pos, position_vector):
        distances = get_distance_vector(pos, position_vector)
        min_distance = np.min(distances)
        pos_min_distance = position_vector[np.argmin(distances)]
        return distances, min_distance, pos_min_distance

    def dig_or_go_to_resouce(self, PREFIX, actions, game_state, positions_to_avoid, turn, unit, rubble_and_opposite_lichen_locations,
                             target_locations, res_name='', drop_ice=False, drop_ore=False):

        if unit.unit_id in self.bot_resource:
            resource = self.bot_resource[unit.unit_id]
            prc(PREFIX, "Looking for", res_name, "actively on assigned resource", resource)
        else:
            # get closest resource
            resource, s = get_map_distances(target_locations, unit.pos)
            prc(PREFIX, "Looking for", res_name, "actively on closest resource", resource)

        # if we have reached the ore tile, start mining if possible
        if np.all(resource == unit.pos):
            if actions.can_dig(unit):
                prc(PREFIX, "On ", res_name, ", dig,", resource)
                actions.dig(unit)
            else:
                prc(PREFIX, "On ", res_name, ", but cannot dig")
                actions.clear_action(unit, PREFIX)
        else:
            self.get_resource_and_dig(PREFIX, game_state, actions, positions_to_avoid, turn, unit, rubble_and_opposite_lichen_locations,
                                      resource, res_name, drop_ice=drop_ice, drop_ore=drop_ore)

    def go_to_target(self, PREFIX, game_state, actions, adjactent_position_to_avoid, turn, unit: lux.kit.Unit, closest_target):
        direction, unit_actions, new_pos, num_digs, num_steps, cost = self.get_complete_path(game_state, unit, turn, adjactent_position_to_avoid,
                                                                                             closest_target, PREFIX, one_way_only=True)
        prc(PREFIX, "go_to_target, found direction", direction, "to", new_pos)
        if direction != 0 and actions.can_move(unit, game_state, direction):
            prc(PREFIX, "Try to go to target, direction", direction)
            actions.set_new_actions(unit, unit_actions, PREFIX)
            self.me.set_unit_next_position(unit.unit_id, new_pos)
            # prx(PREFIX, "set next position ", new_pos)
        else:
            prc(PREFIX, "Try to go to target, aborting")
            actions.clear_action(unit, PREFIX)
        return direction, new_pos, unit_actions

    def get_resource_and_dig(self, PREFIX, game_state, actions, adjactent_position_to_avoid, turn, unit: lux.kit.Unit, rubble_and_opposite_lichen_locations,
                             closest_target, res_name='', drop_ice=False, drop_ore=False):
        direction, unit_actions, new_pos, num_digs, num_steps, cost = self.get_complete_path(game_state, unit, turn, adjactent_position_to_avoid,
                                                                                             closest_target,
                                                                                             PREFIX, drop_ice=drop_ice, drop_ore=drop_ore)

        prc(PREFIX, "get_resource_and_dig Looking for", res_name, " actively, found direction", direction, "to", new_pos, "num_digs", num_digs, 'cost', cost)

        # TODO
        # if unit.get_distance(closest_target) == 1:
        #     friend =  self.me.get_unit_from_current_position(closest_target)
        #     if self.bots_task[friend.unit_id] == self.bots_task[unit.unit_id] and self.bot_resource[friend.unit_id] == self.bot_resource[unit.unit_id]:
        #         check_can_transef_power_next_unit


        # FEATURE B
        if len(rubble_and_opposite_lichen_locations) > 0 \
                and actions.can_dig(unit) and unit.get_distance(closest_target) <= 3 and (
        closest_target[0], closest_target[1]) in self.me.get_unit_next_positions():
            closest_rubble, sorted_rubble = get_map_distances(rubble_and_opposite_lichen_locations, unit.pos)
            if np.all(closest_rubble == unit.pos):
                prc(PREFIX, "Resource is close", res_name, ", but busy,  dig, on ruble/lichen")
                actions.dig(unit)
                return

        if direction != 0 and actions.can_move(unit, game_state, direction) and \
                (num_digs > 0 or self.me.is_factory_center(unit.pos_location())):
            prc(PREFIX, "Try to go to target, direction", direction)
            actions.set_new_actions(unit, unit_actions, PREFIX)
            self.me.set_unit_next_position(unit.unit_id, new_pos)
            # prx(PREFIX, "set next position ", new_pos)
        else:
            prc(PREFIX, "Try to go to target, aborting", "direction", direction, "num_digs", num_digs)
            # FEATURE A
            if len(rubble_and_opposite_lichen_locations) > 0 and actions.can_dig(unit):
                # check if we can dig ruble while we wait
                closest_rubble, sorted_rubble = get_map_distances(rubble_and_opposite_lichen_locations, unit.pos)
                if np.all(closest_rubble == unit.pos):
                    prc(PREFIX, "Was going for", res_name, "but cannot,  dig, on ruble/lichen")
                    actions.dig(unit)
                    return
            if self.me.is_factory_area(unit.pos_location()):
                prc(PREFIX, "Was going for", res_name, "but cannot, recharge")
                actions.dropcargo_or_recharge(unit, force_recharge=True)
                return
            # else
            actions.clear_action(unit, PREFIX)
        return direction, new_pos, unit_actions

    def send_unit_home(self, PREFIX, game_state, actions, adjactent_position_to_avoid, factory_centers, factory_areas, turn, unit):

        direction, unit_actions, new_pos, num_digs, num_steps, cost = \
            self.get_complete_path(game_state, unit, turn, adjactent_position_to_avoid, factory_areas, PREFIX, one_way_only_and_recharge=True)

        if unit.on_factory(game_state):
            prc(PREFIX, "Already home")
            actions.dropcargo_or_recharge(unit, force_recharge=True)
            return direction, new_pos, unit_actions
        elif direction == 0 or not actions.can_move(unit, game_state, direction):
            prc(PREFIX, "Cannot go home via areas dir=", direction, "cost=", actions.move_cost(unit, game_state, direction), "trying centre")
            direction, unit_actions, new_pos, num_digs, num_steps, cost = \
                self.get_complete_path(game_state, unit, turn, adjactent_position_to_avoid, factory_centers, PREFIX, one_way_only_and_recharge=True)
            if direction == 0 or not actions.can_move(unit, game_state, direction):
                prc(PREFIX, "Cannot go home via centers", direction, ", aborting")
                prc(PREFIX, Queue.is_next_queue_move(unit, direction), unit.power, actions.move_cost(unit, game_state, direction),
                    unit.action_queue_cost())
                actions.clear_action(unit, PREFIX)
                return direction, new_pos, unit_actions

        if direction != 0:
            prc(PREFIX, "Go home dir=", direction, Queue.is_next_queue_move(unit, direction), actions.move_cost(unit, game_state, direction))
            actions.set_new_actions(unit, unit_actions, PREFIX)
            self.me.set_unit_next_position(unit.unit_id, new_pos)
        return direction, new_pos, unit_actions

    def build_light_robot(self, actions, factory, t_prefix, role):
        actions.build_light(factory)
        self.built_robot(factory, 'LIGHT', t_prefix, role)

    def build_heavy_robot(self, actions, factory, t_prefix, role):
        actions.build_heavy(factory)
        self.built_robot(factory, 'HEAVY', t_prefix, role)

    def built_robot(self, factory, type, t_prefix, role):
        pr(t_prefix, factory.unit_id, "Build", type, "robot in", factory.pos, 'role', role)
        self.built_robots.append(factory.pos_location())
        self.me.set_unit_next_position(factory.unit_id, factory.pos_location())

    def get_direction(self, game_state, unit, positions_to_avoid, destination, PREFIX=None):

        destination = np.array(destination)
        path = self.path_finder.get_shortest_path(unit.pos, destination, points_to_exclude=positions_to_avoid)
        direction = 0
        if len(path) > 1:
            direction = direction_to(np.array(unit.pos), path[1])

        return direction

    def get_complete_path_ice(self, game_state, unit, turn, positions_to_avoid, destination, PREFIX=None, one_way_only_and_dig=False):
        return self.get_complete_path(game_state, unit, turn, positions_to_avoid, destination, PREFIX=PREFIX, drop_ice=True,
                                      one_way_only_and_dig=one_way_only_and_dig)

    def get_complete_path_ore(self, game_state, unit, turn, positions_to_avoid, destination, PREFIX=None, one_way_only_and_dig=False):
        return self.get_complete_path(game_state, unit, turn, positions_to_avoid, destination, PREFIX=PREFIX, drop_ore=True,
                                      one_way_only_and_dig=one_way_only_and_dig)

    def get_complete_path(self, game_state, unit, turn, positions_to_avoid, destination, PREFIX=None, drop_ice=False, drop_ore=False,
                          one_way_only_and_dig=False, one_way_only_and_recharge=False, one_way_only=False):
        directions, opposite_directions, cost, cost_from, new_pos, num_steps = self.get_one_way_path(game_state, unit, positions_to_avoid, destination,
                                                                                                     PREFIX)

        # set first direction
        if len(directions) > 0:
            unit_actions, number_digs = self.get_actions_sequence(unit, turn, directions, opposite_directions, cost,
                                                                  cost_from, drop_ice=drop_ice, drop_ore=drop_ore, PREFIX=PREFIX,
                                                                  one_way_only_and_dig=one_way_only_and_dig,
                                                                  one_way_only_and_recharge=one_way_only_and_recharge,
                                                                  one_way_only=one_way_only)
            direction = directions[0]
            return direction, unit_actions, new_pos, number_digs, num_steps, cost
        else:
            return 0, [], new_pos, 0, 0, 0

    def get_one_way_path(self, game_state, unit, positions_to_avoid, destination, PREFIX=None):

        destination = np.array(destination)
        path = self.path_finder.get_shortest_path(unit.pos, destination,
                                                  points_to_exclude=positions_to_avoid)
        directions, opposite_directions = [], []
        cost_to, cost_from = 0, 0
        new_pos = unit.pos_location()

        # https://stackoverflow.com/questions/522563/accessing-the-index-in-for-loops
        # >> > l = [1, 2, 3, 4, 5, 6]
        #
        # >> > zip(l, l[1:])
        # [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        # prx(path)
        # calculate directions
        if len(path) > 1:
            new_pos = path[1]
            for p1, p2 in zip(path, path[1:]):
                # direction and opposite direction
                d = direction_to(np.array(p1), np.array(p2))
                d_opposite = opposite_direction(d)
                # append
                directions.append(d)
                opposite_directions.insert(0, d_opposite)

            for idx, p in enumerate(path):
                cost = unit.move_cost_to(game_state, p)
                if idx == 0:
                    # only cost on the way back
                    cost_from += cost
                elif idx == len(path) - 1:
                    # only cost to
                    cost_to += cost
                else:
                    cost_to += cost
                    cost_from += cost

        # calculate costs
        # prx(PREFIX, "get_complete_path from",unit.pos, destination)
        # prx(PREFIX, "get_complete_path path to",directions)
        # prx(PREFIX, "get_complete_path path from",opposite_directions)
        # prx(PREFIX, "get_complete_path path costs",cost_to,cost_from)

        steps = len(path) - 1
        # prx(PREFIX, "path", steps, path)

        return directions, opposite_directions, cost_to, cost_from, new_pos, steps

    def get_actions_sequence(self, unit, turn, directions, opposite_directions, cost_to, cost_from, drop_ore=False, drop_ice=False, PREFIX=None,
                             one_way_only_and_dig=False, one_way_only_and_recharge=False, one_way_only=False):
        DIG_COST = unit.unit_cfg.DIG_COST
        ACTION_QUEUE_COST = unit.action_queue_cost()
        CHARGE = unit.charge_per_turn()

        unit_actions = []

        walking_turn_in_day = 0
        turn_at_digging = turn + len(directions)
        for d in range(turn, turn_at_digging):
            if is_day(d):
                walking_turn_in_day += 1
        cost_to = cost_to - (walking_turn_in_day * CHARGE)

        # sequence to go
        for d in directions:
            unit_actions.append(unit.move(d))

        if one_way_only:
            return unit_actions, 0

        power_at_start_digging = unit.power - ACTION_QUEUE_COST - cost_to
        # prx(PREFIX, "1 power_at_start_digging", power_at_start_digging, "DIG_COST", DIG_COST)
        if power_at_start_digging < DIG_COST or one_way_only_and_dig:
            # if we have not to go back and forth, just go there and dig undefinetely
            unit_actions.append(unit.dig(n=50))
            return unit_actions, 0

        number_digs = 0
        if not one_way_only_and_recharge:
            # sequence to dig
            number_digs_at_least = (power_at_start_digging - cost_from) // DIG_COST
            # prx(PREFIX, "unit.power",unit.power,"cost to", cost_to, "cost from", cost_from, 'power_at_start_digging',power_at_start_digging)
            # prx(PREFIX, 'distance',len(directions), '(', power_at_start_digging, " - ", cost_from, ") //", DIG_COST, '=', number_digs_at_least)

            # we check if we can do more...
            new_number_digs = number_digs_at_least
            while (True):
                new_number_digs = new_number_digs + 1
                extra_day_sun = 0
                turn_start_digging = turn_at_digging + 1
                for d in range(turn_start_digging, turn_start_digging + new_number_digs + len(opposite_directions) - 1):
                    if is_day(d):
                        # prx(PREFIX, 'is day', d)
                        extra_day_sun += 1

                if (power_at_start_digging + (extra_day_sun * CHARGE) - (new_number_digs * DIG_COST)) >= cost_from:
                    # prx(PREFIX, 'using extra day ', extra_day_sun, 'new_number_digs_at_least', new_number_digs)
                    # prx(PREFIX, power_at_start_digging + (extra_day_sun * CHARGE) - (new_number_digs * DIG_COST), cost_from)
                    number_digs_at_least = new_number_digs
                else:
                    break

            number_digs = number_digs_at_least
            # prx(PREFIX, 'number_digs=', number_digs)
            if number_digs <= 0:
                return unit_actions, number_digs

            # if turn == 48: 5/0.

            unit_actions.append(unit.dig(n=min(number_digs, 50)))

            # sequence to return
            for d in opposite_directions:
                unit_actions.append(unit.move(d))

        # dropcargo
        if drop_ore:
            unit_actions.append(Queue.action_transfer_ore(unit))
        if drop_ice:
            unit_actions.append(Queue.action_transfer_ice(unit))

        # and recharge
        unit_actions.append(Queue.action_pickup_power(unit, unit.battery_capacity() - unit.power, repeat=True))
        return unit_actions, number_digs

    def get_cost_to(self, game_state, unit, turn, adjactent_position_to_avoid, destination, PREFIX=None):

        destination = np.array(destination)
        path = self.path_finder.get_shortest_path(unit.pos, destination, points_to_exclude=adjactent_position_to_avoid, PREFIX=PREFIX)
        # prx(PREFIX, "XXX", unit.pos, destination,'ex',adjactent_position_to_avoid, 'path', path)
        if len(path) == 1:
            return 0
        elif len(path) > 1:
            # prx(PREFIX, path)
            cost_to = 0;
            for idx, p in enumerate(path):
                cost = unit.move_cost_to(game_state, p)
                if idx >= 1:
                    # prx(PREFIX, cost_to, 'adding cost', cost)
                    cost_to += cost

            # remove charge
            for d in range(turn, len(path) - 1):
                if is_day(d):
                    # prx(PREFIX, cost_to, 'decreasing light', unit.charge_per_turn())
                    cost_to = cost_to - unit.charge_per_turn()
            return cost_to

        else:
            return 10e6

    def get_random_direction(self, unit, PREFIX=None):

        move_deltas_real = [(1, (0, -1)), (2, (1, 0)), (3, (0, 1)), (4, (-1, 0))]
        for direction, delta in move_deltas_real:
            new_pos = np.array(unit.pos) + delta
            # prc(PREFIX,"try random ",new_pos)
            if not (new_pos[0], new_pos[1]) in self.me.get_unit_next_positions():
                return direction

        return 0
