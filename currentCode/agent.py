import math

from action import *
from path_finder import *
from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import sys


import numpy as np

import networkx as nx

# create robot in the best location (especially first)
# do not bring ice to city that do not need
# do not harvest ice when all cities do not need water


def pr(*args, sep=' ', end='\n', force=False):  # print conditionally
    if (True or force): # change first parameter to False to disable logging
        print(*args, sep=sep, file=sys.stderr)

def prx(*args): pr(*args, force=True)

def prc(*args):  # print conditionally
    if (False and (('unit_10' in args[0]) or ('unit_8' in args[0]))):
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
                    #zero the centre 3x3 where we put the factory
                    area_factory = obs["board"]["rubble"][max(x - 1, 0):min(x + 1, 47),max(y - 1, 0):max(y + 1, 47)]

                    area_size = (x2-x1)*(y2-y1)
                    density_rubble = (np.sum(area_around_factory) - np.sum(area_factory)) / area_size
                    potential_lichen = area_size * 100 - (np.sum(area_around_factory) - np.sum(area_factory))


                    #prx(area_around_factory)

                    closes_opp_factory_dist = 0
                    if len(opp_factories) >= 1:
                        closes_opp_factory_dist = np.min(np.mean((np.array(opp_factories) - loc) ** 2, 1))
                    closes_my_factory_dist = 0
                    if len(my_factories) >= 1:
                        closes_my_factory_dist = np.min(np.mean((np.array(my_factories) - loc) ** 2, 1))

                    kpi_build_factory = 0
                    if (water_left > 0):
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
                                         "ice="+str(np.min(ice_tile_distances)),
                                         "ore="+str(np.min(ore_tile_distances)),
                                         "are="+str(area_size),
                                         "lic="+str(potential_lichen),
                                         "ofc="+str(closes_opp_factory_dist),
                                         "mfc="+str(closes_opp_factory_dist),
                                         x1,x2,y1,y2
                                         )



                #choose location that is the best according to the KPI above
                spawn_loc = best_loc
                actions['spawn'] = spawn_loc


                # we assign to each factory 300 or what is left (this means we build 3 factories with 300, 300, 150
                actions['metal'] = min(METAL_TO_ALLOCATE, metal_left)
                actions['water'] = min(WATER_TO_ALLOCATE, water_left)

                prx('Created factory in ',spawn_loc,'(', actions['water'], actions['metal'],')based on ', chosen_params)

        return actions



    def act(self, step: int, obs, remainingOverageTime: int = 60):
        '''
        1. Regular Phase
        2. Building Robots
        '''

        game_state = obs_to_game_state(step, self.env_cfg, obs)
        self.path_finder.build_path(game_state,self.player, self.opp_player)
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
        self.unit_next_positions = {}

        # FACTORY LOOP
        for factory_id, factory in factories.items():

            if factory_id not in self.factory_bots.keys():
                self.factory_bots[factory_id] = {
                    'ice': [],
                    'ore': [],
                    'rubble': [],
                    'kill': [],
                }

                self.factory_queue[factory_id] = []

            for task in ['ice', 'ore', 'rubble', 'kill']:
                for bot_unit_id in self.factory_bots[factory_id][task]:
                    if bot_unit_id not in self.botpos.keys():
                        self.factory_bots[factory_id][task].remove(bot_unit_id)

            minbot_task = None
            min_bots = {
                'ice': 1,
                'ore': 5,
                'rubble': 5,
                'kill': 1
            }
            # NO. BOTS PER TASK
            for task in ['ice', 'kill', 'ore', 'rubble']:
                num_bots = len(self.factory_bots[factory_id][task]) + sum([task in self.factory_queue[factory_id]])
                if num_bots < min_bots[task]:
                    minbots = num_bots
                    minbot_task = task
                    break

            #BUILD ROBOT ENTRY POINT
            if minbot_task is not None:
                # TODO THIS SHOULD BE MOVED AFTER ROBOT MOVEMENT, SO TO CHECK IF ROBOTS ARE STILL THERE
                #
                #     if ((factory.pos[0],factory.pos[1])) in self.unit_locations:
                #         pr(t_prefix, factory.unit_id, "Cannot build robot, already an unit present",factory.pos)
                #
                if minbot_task in ['kill', 'ice']:
                    if factory.can_build_heavy(game_state):
                        self.build_heavy_robot(actions, factory, t_prefix)
                    elif factory.can_build_light(game_state):
                        self.build_light_robot(actions, factory, t_prefix)
                else:
                    if factory.can_build_light(game_state):
                        self.build_light_robot(actions, factory, t_prefix)
                    elif factory.can_build_heavy(game_state):
                        self.build_heavy_robot(actions, factory, t_prefix)

                if factory_id not in self.factory_queue.keys():
                    self.factory_queue[factory_id] = [minbot_task]
                    # prx(t_prefix, "set id ",factory_id,' to ', self.factory_queue[factory_id])
                else:
                    self.factory_queue[factory_id].append(minbot_task)
                    # prx(t_prefix, "append id ", factory_id, ' to ', self.factory_queue[factory_id])

            factory_tiles += [factory.pos]
            Path_Finder.expand_point(factory_areas, factory.pos)
            factory_units += [factory]
            factory_ids += [factory_id]

            #if we have excess water use to grow lichen
            if (factory.cargo.water - factory.water_cost(game_state)) > 1:
                # at the end, we start water if we can
                if turn_left<10 and \
                        (factory.cargo.water + math.floor(factory.cargo.ice / 4) - factory.water_cost(game_state)) > turn_left:
                    # prx(t_prefix, 'water', factory_id, "water=", factory.cargo.water, "ice=", factory.cargo.water, "cost=", factory.water_cost(game_state),"left=", turn_left)
                    actions.water(factory)

                # anyway, we start water if we have resource to water till the end
                elif (factory.cargo.water + math.floor(factory.cargo.ice / 4)) > turn_left * max(1,(1 + factory.water_cost(game_state))):
                    # prx(t_prefix, 'water', factory_id, "water=", factory.cargo.water, "ice=", factory.cargo.water, "cost=", factory.water_cost(game_state), "left=", turn_left)
                    actions.water(factory)


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
                if len(newarray)>0:
                    lichen_opposite_locations = np.vstack((lichen_opposite_locations,newarray))

        ice_locations_all = np.argwhere(ice_map >= 1)  # numpy position of every ice tile
        ore_locations_all = np.argwhere(ore_map >= 1)  # numpy position of every ore tile
        rubble_locations_all = np.argwhere(rubble_map >= 1)  # numpy position of every rubble tile

        ice_locations = ice_locations_all
        ore_locations = ore_locations_all
        rubble_locations = rubble_locations_all

        rubble_and_opposite_lichen_locations = np.vstack((rubble_locations,lichen_opposite_locations))

        for unit_id, unit in iter(sorted(units.items())):
            # default next position to current position, we will modify then in case of movements
            self.unit_next_positions[unit.unit_id] = unit.pos_location()

        # UNIT LOOP
        for unit_id, unit in iter(sorted(units.items())):

            PREFIX = t_prefix+" "+unit_id+ " "+ unit.unit_type+"("+str(unit.power)+")"

            if unit_id not in self.bots_task.keys():
                self.bots_task[unit_id] = ''

            if len(self.opp_botpos) != 0:
                opp_pos = np.array(self.opp_botpos).reshape(-1, 2)
                opponent_unit_distances = self.get_distance_vector(unit.pos, opp_pos)
                opponent_min_distance = np.min(opponent_unit_distances)
                opponent_pos_min_distance = opp_pos[np.argmin(opponent_unit_distances)]

            if len(self.opp_bbotposheavy) != 0:
                opp_heavy_pos = np.array(self.opp_bbotposheavy).reshape(-1, 2)
                opponent_heavy_unit_distances = self.get_distance_vector(unit.pos, opp_heavy_pos)
                opponent_heavy_min_distance = np.min(opponent_heavy_unit_distances)
                opponent_heavy_pos_min_distance = opp_heavy_pos[np.argmin(opponent_heavy_unit_distances)]

            adjacent_to_factory = False
            factory_min_distance = 10000
            if len(factory_tiles) > 0:
                factory_unit_distances = self.get_distance_vector(unit.pos, factory_areas)
                distance_to_factory = np.min(factory_unit_distances)
                factory_pos_min_distance = factory_areas[np.argmin(factory_unit_distances)]
                # prx(PREFIX,unit.pos,'distance', factory_min_distance, factory_pos_min_distance)
                adjacent_to_factory = distance_to_factory == 0
                closest_factory_tile = factory_tiles[0]

            if unit_id not in self.bot_factory.keys():
                factory_distances = self.get_distance_vector(unit.pos,factory_tiles)
                min_index = np.argmin(factory_distances)
                closest_factory_tile = factory_tiles[min_index]
                self.bot_factory[unit_id] = factory_ids[min_index]
            elif self.bot_factory[unit_id] not in factory_ids:
                factory_distances = self.get_distance_vector(unit.pos,factory_tiles)
                min_index = np.argmin(factory_distances)
                closest_factory_tile = factory_tiles[min_index]
                self.bot_factory[unit_id] = factory_ids[min_index]
            else:
                closest_factory_tile = factories[self.bot_factory[unit_id]].pos
            factory_belong = self.bot_factory[unit_id]

            PREFIX = PREFIX + " " + factory_belong

            sorted_factory = [factory_pos_min_distance]



            # UNIT TASK DECISION
            if unit.power < unit.action_queue_cost(game_state):
                continue

            if len(factory_tiles) > 0:

                move_cost = None

                ## Assigning task for the bot
                if self.bots_task[unit_id] == '':
                    task = 'ice'
                    if len(self.factory_queue[self.bot_factory[unit_id]]) != 0:
                        prx(PREFIX, "QUEUE", self.factory_queue[self.bot_factory[unit_id]])
                        task = self.factory_queue[self.bot_factory[unit_id]].pop(0)

                    prx(PREFIX,'from',factory_belong,unit.unit_type,'assigned task',task)
                    self.bots_task[unit_id] = task
                    self.factory_bots[factory_belong][task].append(unit_id)

                assigned_task = self.bots_task[unit_id]
                if len(self.opp_botpos) != 0 and opponent_min_distance == 1 and unit.unit_type == "HEAVY" and assigned_task != "kill":
                    assigned_task = "kill"
                    prx(PREFIX, 'from', factory_belong, unit.unit_type, unit.pos, 'temporarly tasked as', assigned_task, opponent_pos_min_distance, opponent_min_distance)


                if turn_left<200 and assigned_task == "ore":
                   self.bots_task[unit_id] = 'rubble'
                   prx(PREFIX, factory_belong, unit.unit_type, unit.pos, 'permanently tasked from ', assigned_task,
                       'to',self.bots_task[unit_id])
                   assigned_task = self.bots_task[unit_id]


                adjactent_position_to_avoid = []
                for p in self.unit_next_positions.values():
                    if self.get_distance(unit.pos_location(),p) == 1:
                        adjactent_position_to_avoid.append(p)
                    # if len(adjactent_position_to_avoid)>0:
                    #     prx(PREFIX," need to avoid first moves to ", adjactent_position_to_avoid)

                prc(PREFIX, unit.pos, 'task=' + assigned_task, unit.cargo)
                if assigned_task == "ice":

                    if unit.cargo.ice < unit.cargo_space() and unit.power > unit.action_queue_cost(game_state) + unit.dig_cost(
                            game_state) + unit.def_move_cost() * distance_to_factory:

                        # get closest ice
                        closest_ice, sorted_ice = self.get_map_distances(ice_locations, unit.pos)

                        # if we have reached the ice tile, start mining if possible
                        if np.all(closest_ice == unit.pos):
                            if actions.can_dig(unit):
                                actions.dig(unit)
                        else:
                            direction, move_cost = self.get_direction(game_state, unit, adjactent_position_to_avoid,closest_ice, sorted_ice)

                    elif unit.cargo.ice >= unit.cargo_space() or unit.power <= unit.action_queue_cost(
                            game_state) + unit.dig_cost(game_state) + unit.def_move_cost() * distance_to_factory:

                        if adjacent_to_factory:
                            actions.dropcargo_or_recharge(unit)
                        else:
                            direction, move_cost = self.get_direction(game_state, unit, adjactent_position_to_avoid,closest_factory_tile, sorted_factory)

                elif assigned_task == 'ore':
                    if unit.cargo.ore < unit.cargo_space() and unit.power > unit.action_queue_cost(game_state) + unit.dig_cost(
                            game_state) + unit.def_move_cost() * distance_to_factory:

                        # get closest ore
                        closest_ore, sorted_ore = self.get_map_distances(ore_locations, unit.pos)

                        # if we have reached the ore tile, start mining if possible
                        if np.all(closest_ore == unit.pos):
                            if actions.can_dig(unit):
                                actions.dig(unit)
                        else:
                            direction, move_cost = self.get_direction(game_state, unit, adjactent_position_to_avoid,closest_ore, sorted_ore)

                    elif unit.cargo.ore >= unit.cargo_space() or unit.power <= unit.action_queue_cost(
                            game_state) + unit.dig_cost(game_state) + unit.def_move_cost() * distance_to_factory:

                        if adjacent_to_factory:
                            actions.dropcargo_or_recharge(unit)
                        else:
                            direction, move_cost = self.get_direction(game_state, unit, adjactent_position_to_avoid,closest_factory_tile, sorted_factory)                      
                # RUBBLE
                elif assigned_task == 'rubble':
                    # if actions.can_dig(unit): THIS SEEMS WRONG BUT DECREASE PERFORMANCE
                    if unit.can_dig(game_state):

                        # compute the distance to each rubble tile from this unit and pick the closest
                        closest_rubble, sorted_rubble = self.get_map_distances(rubble_and_opposite_lichen_locations, unit.pos)

                        # if we have reached the rubble tile, start mining if possible
                        if np.all(closest_rubble == unit.pos) or rubble_map[unit.pos[0], unit.pos[1]] != 0:
                            #prc(PREFIX,"can dig, on ruble")
                            if actions.can_dig(unit):
                                #prc(PREFIX,"dig ",rubble_map[unit.pos[0], unit.pos[1]])
                                actions.dig(unit)
                        else:
                            #prc(PREFIX, "can dig, not on ruble, move to next ruble")
                            if len(rubble_locations) != 0:
                                direction, move_cost = self.get_direction(game_state, unit, adjactent_position_to_avoid,closest_rubble, sorted_rubble)                        

                    elif unit.power <= unit.action_queue_cost(game_state) + unit.dig_cost(game_state) + unit.rubble_dig_cost():
                        #prc(PREFIX, "cannot dig, adjacent")
                        if adjacent_to_factory:
                            actions.dropcargo_or_recharge(unit)
                        else:
                            direction, move_cost = self.get_direction(game_state, unit, adjactent_position_to_avoid,closest_factory_tile, sorted_factory)          
                elif assigned_task == 'kill':

                    if len(self.opp_botpos) == 0:
                        if len(lichen_opposite_locations) >0:
                            # compute the distance to each rubble tile from this unit and pick the closest
                            closest_opposite_lichen, sorted_opp_lichen = self.get_map_distances(lichen_opposite_locations, unit.pos)

                            # if we have reached the lichen tile, start mining if possible
                            if np.all(closest_opposite_lichen == unit.pos):
                                if unit.power >= unit.dig_cost(game_state) + \
                                        unit.action_queue_cost(game_state):
                                    actions.dig(unit)
                            else:
                                direction, move_cost = self.get_direction(game_state, unit, adjactent_position_to_avoid,closest_opposite_lichen, sorted_opp_lichen)
                    if len(self.opp_botpos) != 0:
                        if opponent_min_distance == 1:
                            direction, move_cost = self.get_direction(game_state, unit, adjactent_position_to_avoid,np.array(opponent_pos_min_distance),
                                                           [np.array(opponent_pos_min_distance)])
                            
                        else:
                            if unit.power > unit.action_queue_cost(game_state):
                                direction, move_cost = self.get_direction(game_state, unit, adjactent_position_to_avoid,np.array(opponent_pos_min_distance),
                                                               [np.array(opponent_pos_min_distance)])
                            else:
                                if adjacent_to_factory:
                                    actions.dropcargo_or_recharge(unit)
                                    direction=0
                                else:
                                    direction, move_cost = self.get_direction(game_state, unit, adjactent_position_to_avoid,closest_factory_tile, sorted_factory)
                        prc(PREFIX,'kill',opponent_pos_min_distance,'d=',direction,'cost=',move_cost)


                if move_cost is not None and direction == 0:
                    prc(PREFIX, 'cannot find a path')
                    if unit.pos in factory_tiles:
                        prc(PREFIX, 'Should not stay here on a center factory, can kill my friends..',unit.pos)
                        direction = self.get_random_direction(unit, PREFIX)
                        prc(PREFIX, 'Random got ', direction)

                # check move_cost is not None, meaning that direction is not blocked
                # check if unit has enough power to move and update the action queue.
                if move_cost is not None and direction != 0 and unit.power >= move_cost + unit.action_queue_cost(game_state):

                    actions.move(unit,direction)
                    # new position
                    new_pos = np.array(unit.pos) + self.move_deltas[direction]
                    prc(PREFIX,'move to ', direction, (new_pos[0],new_pos[1]) in self.unit_next_positions.values())
                    self.unit_next_positions[unit.unit_id] = (new_pos[0], new_pos[1])
                else:
                    #not moving
                    prc(PREFIX, 'Not moving, remove node ', unit.pos, unit.pos_location() in self.unit_next_positions.values())
                    self.unit_next_positions[unit.unit_id] = unit.pos_location()

        return actions.actions

    def build_light_robot(self, actions, factory, t_prefix):
        actions.build_light(factory)
        self.built_robot(factory, 'LIGHT' ,t_prefix)

    def build_heavy_robot(self, actions, factory, t_prefix):
        actions.build_heavy(factory)
        self.built_robot(factory, 'HEAVY' ,t_prefix)

    def built_robot(self, factory, type, t_prefix):
        pr(t_prefix, factory.unit_id, "Build",type,"robot", factory.pos)
        self.built_robots.append(factory.pos_location())
        self.unit_next_positions[factory.unit_id] = factory.pos_location()

    def get_distance(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

    def get_distance_vector(self, pos, points):
        return 2 * np.mean(np.abs(points - pos), 1)

    def get_map_distances(self, locations, pos, rubble_map=[] ):
        if len(rubble_map)>0:
            rubbles = np.array([rubble_map[pos[0]][pos[1]] for pos in locations])
        distances = self.get_distance_vector(pos, locations)  # - (rubbles)*10
        sorted_loc = [locations[k] for k in np.argsort(distances)]
        closest_loc = sorted_loc[0]
        return closest_loc, sorted_loc

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
                    bot_positions.append((unit.pos[0],unit.pos[1]))
                    if unit.unit_type == "HEAVY":
                        botposheavy[unit_id] = str(unit.pos)
                else:
                    opp_botpos.append(unit.pos)
                    if unit.unit_type == "HEAVY":
                        opp_botposheavy.append(unit.pos)

        return bot_positions, botpos, botposheavy, opp_botpos, opp_botposheavy

    def get_direction(self, game_state, unit, adjactent_position_to_avoid, closest_tile, sorted_tiles, PREFIX=None):

        closest_tile = np.array(closest_tile)
        path = self.path_finder.get_shortest_path(unit.pos, closest_tile, points_to_exclude=adjactent_position_to_avoid)
        direction = 0
        if len(path) > 1:
            direction = direction_to(np.array(unit.pos), path[1])

        return direction, unit.move_cost(game_state, direction)

    def get_random_direction(self, unit, PREFIX=None):

        move_deltas_real = [(1,(0, -1)), (2,(1, 0)), (3,(0, 1)), (4,(-1, 0))]
        for direction,delta in move_deltas_real:
            new_pos = np.array(unit.pos) + delta
            # prc(PREFIX,"try random ",new_pos)
            if not (new_pos[0],new_pos[1]) in self.unit_next_positions.values():
                return direction

        return 0





