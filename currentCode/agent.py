import math
from action import *
from path_finder import *
from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import my_turn_to_place_factory
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
    if (False and (('u_9' in args[0]) or ('u_33' in args[0]))):
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

        self.bots_task = {}
        self.botpos = []
        self.bot_factory = {}
        self.factory_bots = {}
        self.factory_queue = {}
        self.move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
        self.built_robots = []
        self.unit_next_positions = {}

        self.G = nx.Graph()
        self.path_finder = Path_Finder()

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
                    ice_tile_distances = np.mean((ice_tile_locations - loc) ** 2, 1)
                    ore_tile_distances = np.mean((ore_tile_locations - loc) ** 2, 1)

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

                    area_size = (x2 - x1) * (y2 - y1)
                    density_rubble = (np.sum(area_around_factory) - np.sum(area_factory)) / area_size
                    potential_lichen = area_size * 100 - (np.sum(area_around_factory) - np.sum(area_factory))

                    # prx(area_around_factory)

                    closes_opp_factory_dist = 0
                    if len(opp_factories) >= 1:
                        closes_opp_factory_dist = np.min(np.mean((np.array(opp_factories) - loc) ** 2, 1))
                    closes_my_factory_dist = 0
                    if len(my_factories) >= 1:
                        closes_my_factory_dist = np.min(np.mean((np.array(my_factories) - loc) ** 2, 1))

                    kpi_build_factory = 0
                    if water_left > 0:
                        # if we still have water, we crate meaningful factories
                        kpi_build_factory = np.min(ice_tile_distances) * 10 \
                                            + 0.01 * np.min(ore_tile_distances) \
                                            - 0.01 * area_size \
                                            - 0.1 * potential_lichen / (DIST_RUBLE) \
                                            - closes_opp_factory_dist * 0.1 \
                                            + closes_my_factory_dist * 0.01
                    else:
                        # water is zero. Create disruptive factory near the enemy
                        kpi_build_factory = closes_opp_factory_dist

                    if kpi_build_factory < min_dist:
                        min_dist = kpi_build_factory
                        best_loc = loc
                        chosen_params = (kpi_build_factory.round(2),
                                         "ice=" + str(np.min(ice_tile_distances)),
                                         "ore=" + str(np.min(ore_tile_distances)),
                                         "are=" + str(area_size),
                                         "lic=" + str(potential_lichen),
                                         "ofc=" + str(closes_opp_factory_dist),
                                         "mfc=" + str(closes_opp_factory_dist),
                                         x1, x2, y1, y2
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
        state_obs = obs

        turn = obs["real_env_steps"]

        # prx("---------Turn number ", turn)
        t_prefix = "T_" + str(turn)
        turn_left = 1001 - turn

        # Unit locations
        self.unit_locations, self.botpos, self.botposheavy, self.opp_botpos, self.opp_bbotposheavy = self.get_unit_locations(game_state)

        # Build Robots
        factories = game_state.factories[self.player]
        factory_tiles, factory_units, factory_ids, factory_areas = [], [], [], []
        self.built_robots = []
        self.unit_next_positions = {}  # index unit id
        self.unit_current_positions = {}  # index position

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
            expand_point(factory_areas, factory.pos)
            factory_units += [factory]
            factory_ids += [factory_id]

        factory_tiles = np.array(factory_tiles)  # Factory locations (to go back to)

        # Move Robots
        # iterate over our units and have them mine the closest ice tile
        units = game_state.units[self.player]

        # Resource map and locations
        ice_map = game_state.board.ice
        ore_map = game_state.board.ore
        rubble_map = game_state.board.rubble

        # prx(t_prefix,'lichen',lichen_strain_map)
        opp_strain = [f.strain_id for _, f in game_state.factories[self.opp_player].items()]
        lichen_opposite_locations = []
        for strain in opp_strain:
            if len(lichen_opposite_locations) == 0:
                lichen_opposite_locations = np.argwhere(game_state.board.lichen_strains == strain)
            else:
                newarray = np.argwhere(game_state.board.lichen_strains == strain)
                if len(newarray) > 0:
                    lichen_opposite_locations = np.vstack((lichen_opposite_locations, newarray))

        ice_locations_all = np.argwhere(ice_map >= 1)  # numpy position of every ice tile
        ore_locations_all = np.argwhere(ore_map >= 1)  # numpy position of every ore tile
        rubble_locations_all = np.argwhere(rubble_map >= 1)  # numpy position of every rubble tile

        ice_locations = ice_locations_all
        ore_locations = ore_locations_all
        rubble_locations = rubble_locations_all

        rubble_and_opposite_lichen_locations = np.vstack((rubble_locations, lichen_opposite_locations))

        # Initial loop to set present and future locations of units
        self.unit_next_positions = {}
        self.unit_current_positions = {}
        power_transfered = {}  # index is position
        for unit_id, unit in iter(sorted(units.items())):
            # default next position to current position, we will modify then in case of movements
            self.unit_next_positions[unit.unit_id] = unit.pos_location()
            self.unit_current_positions[unit.pos_location()] = unit

        # UNIT LOOP
        for unit_id, unit in iter(sorted(units.items())):

            PREFIX = t_prefix + " " + unit.unit_type_short() + "(" + str(unit.power) + ") @" + str(unit.pos) + ' ' + unit.unit_id_short()

            if unit_id not in self.bots_task.keys():
                self.bots_task[unit_id] = ''

            if len(self.opp_botpos) != 0:
                opp_pos = np.array(self.opp_botpos).reshape(-1, 2)
                opponent_unit_distances = get_distance_vector(unit.pos, opp_pos)
                opponent_min_distance = np.min(opponent_unit_distances)
                opponent_pos_min_distance = opp_pos[np.argmin(opponent_unit_distances)]

            if len(self.opp_bbotposheavy) != 0:
                opp_heavy_pos = np.array(self.opp_bbotposheavy).reshape(-1, 2)
                opponent_heavy_unit_distances = get_distance_vector(unit.pos, opp_heavy_pos)
                opponent_heavy_min_distance = np.min(opponent_heavy_unit_distances)
                opponent_heavy_pos_min_distance = opp_heavy_pos[np.argmin(opponent_heavy_unit_distances)]

            on_factory = False
            factory_min_distance = 10000
            if len(factory_tiles) > 0:
                factory_unit_distances = get_distance_vector(unit.pos, factory_areas)
                distance_to_factory = np.min(factory_unit_distances)
                closest_factory_area = factory_areas[np.argmin(factory_unit_distances)]
                on_factory = distance_to_factory == 0
                # prx(t_prefix,'-----------------------')
                # prx(t_prefix,'factory_areas',factory_areas)
                # prx(t_prefix,'factory_unit_distances',factory_unit_distances)
                # prx(t_prefix,'distance_to_factory',distance_to_factory)
                # prx(t_prefix,'closest_factory_area',closest_factory_area)
                # prx(t_prefix,'closest_factory_area',closest_factory_area)

            if unit_id not in self.bot_factory.keys():
                factory_distances = get_distance_vector(unit.pos, factory_tiles)
                min_index = np.argmin(factory_distances)
                closest_factory_tile = factory_tiles[min_index]
                self.bot_factory[unit_id] = factory_ids[min_index]
            elif self.bot_factory[unit_id] not in factory_ids:
                factory_distances = get_distance_vector(unit.pos, factory_tiles)
                min_index = np.argmin(factory_distances)
                closest_factory_tile = factory_tiles[min_index]
                self.bot_factory[unit_id] = factory_ids[min_index]
            else:
                closest_factory_tile = factories[self.bot_factory[unit_id]].pos
            factory_belong = self.bot_factory[unit_id]

            # UNIT TASK DECISION
            if unit.power < unit.action_queue_cost():
                continue

            if len(factory_tiles) > 0:

                move_cost = None

                # Assigning task for the bot
                if self.bots_task[unit_id] == '':
                    t = 'ice'
                    if len(self.factory_queue[self.bot_factory[unit_id]]) != 0:
                        prx(PREFIX, "QUEUE", self.factory_queue[self.bot_factory[unit_id]])
                        t = self.factory_queue[self.bot_factory[unit_id]].pop(0)

                    prx(PREFIX, 'from', factory_belong, unit.unit_type, 'assigned task', t)
                    # if task =='kill' and unit.unit_type == 'LIGHT':
                    #     prx(PREFIX, 'Cannot get a light killer! Rubble instead')
                    #     task = 'rubble'

                    self.bots_task[unit_id] = t
                    self.factory_bots[factory_belong][t].append(unit_id)

                assigned_task = self.bots_task[unit_id]
                if len(self.opp_botpos) != 0 and opponent_min_distance == 1 and unit.unit_type == "HEAVY" and assigned_task != "kill":
                    assigned_task = "kill"
                    prx(PREFIX, 'from', factory_belong, unit.unit_type, unit.pos, 'temporarly tasked as', assigned_task, opponent_pos_min_distance,
                        opponent_min_distance)

                if turn_left < 200 and assigned_task == "ore":
                    self.bots_task[unit_id] = 'rubble'
                    prx(PREFIX, factory_belong, unit.unit_type, unit.pos, 'permanently tasked from', assigned_task,
                        'to', self.bots_task[unit_id])
                    assigned_task = self.bots_task[unit_id]

                positions_to_avoid = []
                for p in self.unit_next_positions.values():
                    if get_distance(unit.pos_location(), p) == 1:
                        positions_to_avoid.append(p)
                    # if len(positions_to_avoid)>0:
                    #     prx(PREFIX," need to avoid first moves to ", positions_to_avoid)

                # prc(PREFIX, unit.pos, 'task=' + assigned_task, unit.cargo)
                if assigned_task == "ice":
                    cost_home = self.get_cost_to(game_state, unit, turn, positions_to_avoid, closest_factory_area)
                    recharge_power = if_is_day(turn + 1, unit.charge_per_turn(), 0)
                    # prx(PREFIX, 'cost_home',cost_home, unit.pos, closest_factory_area, 'distance ',distance_to_factory)
                    # prx(PREFIX, 'nit.power + recharge_power',unit.power + recharge_power)
                    # prx(PREFIX, 'Queue.real_cost_dig(unit)',Queue.real_cost_dig(unit))
                    # prx(PREFIX, ' 1=== ',unit.cargo.ice < unit.cargo_space())
                    # prx(PREFIX, ' 2=== ', unit.power + recharge_power > Queue.real_cost_dig(unit) + cost_home)
                    # if turn == 49: a=5/.0

                    if unit.cargo.ice < unit.cargo_space() \
                            and unit.power + recharge_power > Queue.real_cost_dig(unit) + cost_home:

                        self.dig_or_go_to_resouce(PREFIX, actions, game_state, positions_to_avoid, turn, unit, rubble_and_opposite_lichen_locations,
                                                  ice_locations, 'ice', drop_ice=True)

                    else:
                        if on_factory:
                            actions.dropcargo_or_recharge(unit)
                        else:
                            # GO HOME
                            self.send_unit_home(PREFIX, game_state, actions, positions_to_avoid, closest_factory_tile, turn, unit)
                    continue

                elif assigned_task == 'ore':
                    # prx(PREFIX, "Looking for ore")
                    cost_home = self.get_cost_to(game_state, unit, turn, positions_to_avoid,
                                                 closest_factory_area)
                    recharge_power = if_is_day(turn + 1, unit.charge_per_turn(), 0)
                    # prx(PREFIX, 'cost_home',cost_home, unit.pos, closest_factory_area, 'distance ',distance_to_factory)
                    # prx(PREFIX, 'nit.power + recharge_power',unit.power + recharge_power)
                    # prx(PREFIX, 'Queue.real_cost_dig(unit)',Queue.real_cost_dig(unit))
                    # prx(PREFIX, ' 1=== ',unit.cargo.ore < unit.cargo_space())
                    # prx(PREFIX, ' 2=== ', unit.power + recharge_power > Queue.real_cost_dig(unit) + cost_home)
                    # if turn == 49: a=5/.0

                    if unit.cargo.ore < unit.cargo_space() and unit.power + recharge_power > Queue.real_cost_dig(unit) + cost_home:

                        self.dig_or_go_to_resouce(PREFIX, actions, game_state, positions_to_avoid, turn, unit, rubble_and_opposite_lichen_locations,
                                                  ore_locations,'ore', drop_ore=True)

                    else:
                        if on_factory:
                            prc(PREFIX, "on base dropcargo_or_recharge")
                            actions.dropcargo_or_recharge(unit)
                        else:
                            # GO HOME
                            self.send_unit_home(PREFIX, game_state, actions, positions_to_avoid, closest_factory_tile, turn, unit)
                    continue

                # RUBBLE
                elif assigned_task == 'rubble':
                    if unit.power > unit.action_queue_cost() + unit.dig_cost() + unit.rubble_dig_hurdle():
                        # if actions.can_dig(unit):

                        # compute the distance to each rubble tile from this unit and pick the closest
                        closest_rubble, sorted_rubble = get_map_distances(rubble_and_opposite_lichen_locations, unit.pos)

                        # if we have reached the rubble tile, start mining if possible
                        if np.all(closest_rubble == unit.pos):
                            prc(PREFIX, "can dig, on ruble")
                            if actions.can_dig(unit):
                                prc(PREFIX, "dig ruble or lichen")
                                actions.dig(unit)

                        else:
                            prc(PREFIX, "can dig, not on ruble, move to next ruble")
                            if len(rubble_locations) != 0:

                                # see if in the straight direction there is a friendly unit
                                # if (self.check_can_transef_power_next_unit(PREFIX, unit, actions, closest_rubble, power_transfered)):
                                #     break

                                direction = 0
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

                                # prx(PREFIX, "new ore direction ", direction)
                                if direction == 0:
                                    actions.clear_action(unit, PREFIX)
                                    continue

                                elif direction != 0:
                                    actions.set_new_actions(unit, unit_actions, PREFIX)
                                    self.unit_next_positions[unit.unit_id] = (new_pos[0], new_pos[1])
                                    # prx(PREFIX, "set next position ", new_pos)
                                    continue

                    else:
                        # prc(PREFIX, "cannot dig, adjacent")
                        if on_factory:
                            prc(PREFIX, "on factory recharge")
                            actions.dropcargo_or_recharge(unit)
                        else:
                            # GO HOME
                            self.send_unit_home(PREFIX, game_state, actions, positions_to_avoid, closest_factory_tile, turn, unit)
                            continue

                elif assigned_task == 'kill':

                    if len(self.opp_botpos) == 0:
                        if len(lichen_opposite_locations) > 0:
                            # compute the distance to each rubble tile from this unit and pick the closest
                            closest_opposite_lichen, sorted_opp_lichen = get_map_distances(lichen_opposite_locations, unit.pos)

                            # if we have reached the lichen tile, start mining if possible
                            if np.all(closest_opposite_lichen == unit.pos):
                                if unit.power >= unit.dig_cost() + \
                                        unit.action_queue_cost():
                                    actions.dig(unit)
                            else:
                                direction, move_cost = self.get_direction(game_state, unit, positions_to_avoid, closest_opposite_lichen)
                    if len(self.opp_botpos) != 0:
                        if opponent_min_distance == 1:
                            direction, move_cost = self.get_direction(game_state, unit, positions_to_avoid, np.array(opponent_pos_min_distance))

                        else:
                            if unit.power > unit.action_queue_cost():
                                direction, move_cost = self.get_direction(game_state, unit, positions_to_avoid, np.array(opponent_pos_min_distance))
                            else:
                                if on_factory:
                                    actions.dropcargo_or_recharge(unit)
                                    direction = 0
                                else:
                                    direction, move_cost = self.get_direction(game_state, unit, positions_to_avoid, closest_factory_tile)
                        prc(PREFIX, 'kill', opponent_pos_min_distance, 'd=', direction, 'cost=', move_cost)

                if move_cost is not None and direction == 0:
                    # prc(PREFIX, 'cannot find a path')
                    closest_center, sorted_centers = get_map_distances(factory_tiles, unit.pos)
                    if np.all(closest_center == unit.pos):
                        prx(PREFIX, 'Should not stay here on a center factory, can kill my friends..', unit.pos)
                        direction = self.get_random_direction(unit, PREFIX)
                        prx(PREFIX, 'Random got ', direction)

                # check move_cost is not None, meaning that direction is not blocked
                # check if unit has enough power to move and update the action queue.

                if move_cost is not None and direction != 0 and unit.power >= move_cost + unit.action_queue_cost():
                    actions.move(unit, direction)
                    # new position
                    new_pos = np.array(unit.pos) + self.move_deltas[direction]
                    # prc(PREFIX,'move to ', direction, (new_pos[0],new_pos[1]) in self.unit_next_positions.values())
                    self.unit_next_positions[unit.unit_id] = (new_pos[0], new_pos[1])
                else:
                    # not moving
                    # prx(PREFIX, 'Not moving, remove node ', unit.pos, unit.pos_location() in self.unit_next_positions.values())
                    self.unit_next_positions[unit.unit_id] = unit.pos_location()

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
                for t in ['ice', 'kill', 'ore', 'rubble']:
                    num_bots = len(self.factory_bots[factory_id][t]) + sum([t in self.factory_queue[factory_id]])
                    if num_bots < min_bots[t]:
                        prx(t_prefix, "We have less bots(", num_bots, ") for", t, " than min", min_bots[t])
                        new_task = t
                        break

                # toward the end of the game, build as many as rubble collector as you can
                # if turn_left<200:
                #     new_task = 'rubble'

                # BUILD ROBOT ENTRY POINT
                if new_task is not None:
                    # Check we are not building on top of another unit
                    if (factory.pos[0], factory.pos[1]) in self.unit_next_positions.values():
                        pr(t_prefix, factory.unit_id, "Cannot build robot, already an unit present", factory.pos)
                        continue

                    if new_task in ['kill', 'ice']:
                        if factory.can_build_heavy(game_state) or turn_left < 200:
                            self.build_heavy_robot(actions, factory, t_prefix)
                        elif factory.can_build_light(game_state):
                            self.build_light_robot(actions, factory, t_prefix)
                    else:
                        if factory.can_build_light(game_state):
                            self.build_light_robot(actions, factory, t_prefix)

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
                    if turn_left < 10 and \
                            (factory.cargo.water + math.floor(factory.cargo.ice / 4) - factory.water_cost(game_state)) > turn_left:
                        # prx(t_prefix, 'water', factory_id, "water=", factory.cargo.water, "ice=", factory.cargo.ice, "cost=", factory.water_cost(game_state),"left=", turn_left)
                        actions.water(factory)

                    # anyway, we start water if we have resource to water till the end
                    elif (factory.cargo.water + math.floor(factory.cargo.ice / 4)) > turn_left * max(1, (1 + factory.water_cost(game_state))):
                        # prx(t_prefix, 'water', factory_id, "water=", factory.cargo.water, "ice=", factory.cargo.ice, "cost=", factory.water_cost(game_state), "left=", turn_left)
                        actions.water(factory)

        # if turn==205:
        #     prx(t_prefix,"next_position =====",self.unit_next_positions)
        # if turn==18:
        #     a=5/0.

        return actions.actions

    def dig_or_go_to_resouce(self, PREFIX, actions, game_state, positions_to_avoid, turn, unit, rubble_and_opposite_lichen_locations,
                             target_locations, res_name='', drop_ice=False, drop_ore=False):
        prc(PREFIX, "Looking for",res_name,"actively")
        # get closest resource
        closest_resource, sorted_resources = get_map_distances(target_locations, unit.pos)
        # if we have reached the ore tile, start mining if possible
        if np.all(closest_resource == unit.pos):
            prc(PREFIX, "On ",res_name,", try to dig,", closest_resource)
            if actions.can_dig(unit):
                actions.dig(unit)
        else:
            self.get_resource_and_dig(PREFIX, game_state, actions, positions_to_avoid, turn, unit, rubble_and_opposite_lichen_locations,
                                      closest_resource, res_name, drop_ice=drop_ice, drop_ore=drop_ore)

    def get_resource_and_dig(self, PREFIX, game_state, actions, adjactent_position_to_avoid, turn, unit, rubble_and_opposite_lichen_locations,
                             closest_target, res_name='', drop_ice=False, drop_ore=False):
        direction, unit_actions, new_pos, num_digs, num_steps, cost = self.get_complete_path(game_state, unit, turn, adjactent_position_to_avoid,
                                                                                             closest_target,
                                                                                             PREFIX, drop_ice=drop_ice, drop_ore=drop_ore)



        prc(PREFIX, "Looking for", res_name, " actively, found direction", direction, "to", new_pos, "num_digs", num_digs)
        if direction != 0 and num_digs > 0:
            prc(PREFIX, "Try to go to target, direction", direction)
            actions.set_new_actions(unit, unit_actions, PREFIX)
            self.unit_next_positions[unit.unit_id] = (new_pos[0], new_pos[1])
            # prx(PREFIX, "set next position ", new_pos)
        else:
            prc(PREFIX, "Try to go to target, aborting")
            actions.clear_action(unit, PREFIX)
        return direction, new_pos, unit_actions

    def send_unit_home(self, PREFIX, game_state, actions, adjactent_position_to_avoid, closest_factory_tile, turn, unit):
        prc(PREFIX, "try to go home")
        direction, unit_actions, new_pos, num_digs, num_steps, cost = \
            self.get_complete_path(game_state, unit, turn, adjactent_position_to_avoid, closest_factory_tile, PREFIX,
                                   one_way_only_and_recharge=True)
        if direction == 0 or not actions.can_move(unit, game_state, direction):
            prc(PREFIX, "Cannot go home", direction)
            prc(PREFIX, Queue.is_next_queue_move(unit, direction), unit.power, unit.move_cost(game_state, direction),
                unit.action_queue_cost())
            actions.clear_action(unit, PREFIX)
        elif direction != 0:
            prc(PREFIX, "Go home", direction, Queue.is_next_queue_move(unit, direction),
                unit.move_cost(game_state, direction))
            actions.set_new_actions(unit, unit_actions, PREFIX)
            self.unit_next_positions[unit.unit_id] = (new_pos[0], new_pos[1])
            # prx(PREFIX, "set next position ", new_pos)
        return direction, new_pos, unit_actions

    def check_can_transef_power_next_unit(self, PREFIX, unit, actions, target_position, power_transfered: dict):
        if unit.get_distance(target_position) == 1:
            direction, move_to = get_straight_direction(unit, target_position)
            if direction != 0 and move_to in self.unit_current_positions.keys():
                unit_moving_on: lux.kit.Unit = self.unit_current_positions[move_to]
                # if the unit is supposed to be there next turn and has power capacity
                if unit_moving_on.battery_capacity_left() > 0 and move_to == self.unit_next_positions[
                    unit_moving_on.unit_id]:
                    # prx(PREFIX, "going on top of", unit_moving_on.unit_id)
                    # prx(PREFIX, 'me ', unit.cargo, 'power', unit.battery_info())
                    # prx(PREFIX, 'him', unit_moving_on.cargo, 'power', unit_moving_on.battery_info())
                    power_given = min(unit.power, unit_moving_on.battery_capacity_left())
                    actions.transfer_energy(unit, direction, power_given)
                    if move_to in power_transfered:
                        power_transfered[move_to] += power_given
                    else:
                        power_transfered[move_to] = power_given
                    return True
        return False

    def build_light_robot(self, actions, factory, t_prefix):
        actions.build_light(factory)
        self.built_robot(factory, 'LIGHT', t_prefix)

    def build_heavy_robot(self, actions, factory, t_prefix):
        actions.build_heavy(factory)
        self.built_robot(factory, 'HEAVY', t_prefix)

    def built_robot(self, factory, type, t_prefix):
        pr(t_prefix, factory.unit_id, "Build", type, "robot in", factory.pos)
        self.built_robots.append(factory.pos_location())
        self.unit_next_positions[factory.unit_id] = factory.pos_location()

    def get_unit_locations(self, game_state):
        bot_positions = []
        botpos = {}
        botposheavy = {}
        opp_botpos = []
        opp_botposheavy = []
        for player in [self.player, self.opp_player]:
            for unit_id, unit in game_state.units[player].items():

                if player == self.player:
                    botpos[unit_id] = str(unit.pos)
                    bot_positions.append((unit.pos[0], unit.pos[1]))
                    if unit.unit_type == "HEAVY":
                        botposheavy[unit_id] = str(unit.pos)
                else:
                    opp_botpos.append(unit.pos)
                    if unit.unit_type == "HEAVY":
                        opp_botposheavy.append(unit.pos)

        return bot_positions, botpos, botposheavy, opp_botpos, opp_botposheavy

    def get_direction(self, game_state, unit, positions_to_avoid, destination, PREFIX=None):

        destination = np.array(destination)
        path = self.path_finder.get_shortest_path(unit.pos, destination, points_to_exclude=positions_to_avoid)
        direction = 0
        if len(path) > 1:
            direction = direction_to(np.array(unit.pos), path[1])

        return direction, unit.move_cost(game_state, direction)

    def get_complete_path_ice(self, game_state, unit, turn, positions_to_avoid, destination, PREFIX=None, one_way_only_and_dig=False):
        return self.get_complete_path(game_state, unit, turn, positions_to_avoid, destination, PREFIX=PREFIX, drop_ice=True,
                                      one_way_only_and_dig=one_way_only_and_dig)

    def get_complete_path_ore(self, game_state, unit, turn, positions_to_avoid, destination, PREFIX=None, one_way_only_and_dig=False):
        return self.get_complete_path(game_state, unit, turn, positions_to_avoid, destination, PREFIX=PREFIX, drop_ore=True,
                                      one_way_only_and_dig=one_way_only_and_dig)

    def get_complete_path(self, game_state, unit, turn, positions_to_avoid, destination, PREFIX=None, drop_ice=False, drop_ore=False,
                          one_way_only_and_dig=False, one_way_only_and_recharge=False):
        directions, opposite_directions, cost, cost_from, new_pos, num_steps = self.get_one_way_path(game_state, unit, positions_to_avoid, destination,
                                                                                                     PREFIX)

        # set first direction
        if len(directions) > 0:
            unit_actions, number_digs = self.get_actions_sequence(unit, turn, directions, opposite_directions, cost,
                                                                  cost_from, drop_ice=drop_ice, drop_ore=drop_ore, PREFIX=PREFIX,
                                                                  one_way_only_and_dig=one_way_only_and_dig,
                                                                  one_way_only_and_recharge=one_way_only_and_recharge)
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
                             one_way_only_and_dig=False, one_way_only_and_recharge=None):
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
        unit_actions.append(Queue.action_pickup_power(unit, unit.battery_capacity() - unit.power))
        return unit_actions, number_digs

    def get_cost_to(self, game_state, unit, turn, adjactent_position_to_avoid, destination, PREFIX=None):

        destination = np.array(destination)
        path = self.path_finder.get_shortest_path(unit.pos, destination,
                                                  points_to_exclude=adjactent_position_to_avoid)

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
            if not (new_pos[0], new_pos[1]) in self.unit_next_positions.values():
                return direction

        return 0
