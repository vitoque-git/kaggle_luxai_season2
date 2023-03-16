import math

import numpy

from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys

# create robot in the best location (especially first)
# do not bring ice to city that do not need
# do not harvest ice when all cities do not need water

def pr(*args, sep=' ', end='\n', f=False):  # known special case of print
    if False:
        print(*args, sep=sep, file=sys.stderr)
    elif f:
        print(*args, sep=sep, file=sys.stderr)


def prx(*args): pr(*args, f=True)

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
        self.early_setup_steps = 0;

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

    def check_collision(self, pos, direction, unitpos, unit_type='LIGHT'):
        move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
        #         move_deltas = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])

        new_pos = pos + move_deltas[direction]

        if unit_type == "LIGHT":
            return str(new_pos) in unitpos or str(new_pos) in self.botposheavy.values()
        else:
            return str(new_pos) in unitpos

    def get_direction(self, unit, closest_tile, sorted_tiles):

        closest_tile = np.array(closest_tile)
        direction = direction_to(np.array(unit.pos), closest_tile)
        k = 0
        all_unit_positions = set(self.botpos.values())
        unit_type = unit.unit_type
        while self.check_collision(np.array(unit.pos), direction, all_unit_positions, unit_type) and k < min(
                len(sorted_tiles) - 1, 500):
            k += 1
            closest_tile = sorted_tiles[k]
            closest_tile = np.array(closest_tile)
            direction = direction_to(np.array(unit.pos), closest_tile)

        if self.check_collision(unit.pos, direction, all_unit_positions, unit_type):
            for direction_x in np.arange(4, -1, -1):
                if not self.check_collision(np.array(unit.pos), direction_x, all_unit_positions, unit_type):
                    direction = direction_x
                    break

        if self.check_collision(np.array(unit.pos), direction, all_unit_positions, unit_type):
            direction = np.random.choice(np.arange(5))

        move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

        self.botpos[unit.unit_id] = str(np.array(unit.pos) + move_deltas[direction])

        return direction

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        '''
        1. Regular Phase
        2. Building Robots
        '''
        actions = dict()
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        state_obs = obs


        if self.early_setup_steps == 0:
            self.early_setup_steps = step
            prx('Initial step is ', self.early_setup_steps)

        turn = step - self.early_setup_steps
        pr("---------Turn number ", turn)
        t_prefix = "T_" + str(turn)
        turn_left = 1001 - turn

        # Unit locations
        self.botpos, self.botposheavy, self.opp_botpos, self.opp_bbotposheavy = self.get_unit_locations(game_state)

        # Build Robots
        factories = game_state.factories[self.player]
        factory_tiles, factory_units, factory_ids = [], [], []

        for factory_id, factory in factories.items():

            # custom method that change variables on turn changing
            factory.newTurn(turn_left,prx,t_prefix)

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

            if minbot_task is not None:
                if minbot_task in ['kill', 'ice']:
                    if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
                            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                        actions[factory_id] = factory.build_heavy()
                    elif factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and \
                            factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST:
                        actions[factory_id] = factory.build_light()
                else:
                    if factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and \
                            factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST:
                        actions[factory_id] = factory.build_light()
                    elif factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
                            factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                        actions[factory_id] = factory.build_heavy()

                if factory_id not in self.factory_queue.keys():
                    self.factory_queue[factory_id] = [minbot_task]
                else:
                    self.factory_queue[factory_id].append(minbot_task)

            factory_tiles += [factory.pos]
            factory_units += [factory]
            factory_ids += [factory_id]

            #if we have excess water use to grow lichen
            if (factory.cargo.water - factory.water_cost(game_state)) > 1:
                # at the end, we start water if we can
                if turn_left<10 and \
                        (factory.cargo.water + math.floor(factory.cargo.ice / 4) - factory.water_cost(game_state)) > turn_left:
                    # prx(t_prefix, 'water', factory_id, "water=", factory.cargo.water, "ice=", factory.cargo.water, "cost=", factory.water_cost(game_state),"left=", turn_left)
                    actions[factory_id] = factory.water()

                # anyway, we start water if we have resource to water till the end
                elif (factory.cargo.water + math.floor(factory.cargo.ice / 4)) > turn_left * max(1,(1 + factory.water_cost(game_state))):
                    # prx(t_prefix, 'water', factory_id, "water=", factory.cargo.water, "ice=", factory.cargo.water, "cost=", factory.water_cost(game_state), "left=", turn_left)
                    actions[factory_id] = factory.water()


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

        for unit_id, unit in iter(sorted(units.items())):
            PREFIX = t_prefix+" u:"+unit_id
            #prx(PREFIX)
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

            if len(factory_tiles) > 0:
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

            distance_to_factory = np.mean((np.array(closest_factory_tile) - np.array(unit.pos)) ** 2)
            adjacent_to_factory = False
            sorted_factory = [closest_factory_tile]

            if unit.power < unit.action_queue_cost(game_state):
                continue

            if len(factory_tiles) > 0:

                move_cost = None
                try:
                    adjacent_to_factory = np.mean((np.array(closest_factory_tile) - np.array(unit.pos)) ** 2) <= 1
                except:
                    print(closest_factory_tile, unit.pos)
                    assert False

                factory_belong = self.bot_factory[unit_id]
                ## Assigning task for the bot
                if self.bots_task[unit_id] == '':
                    task = 'ice'
                    if len(self.factory_queue[self.bot_factory[unit_id]]) != 0:
                        task = self.factory_queue[self.bot_factory[unit_id]].pop(0)
                    prx(PREFIX,'from',factory_belong,unit.unit_type,'assigned task',task)
                    self.bots_task[unit_id] = task
                    self.factory_bots[factory_belong][task].append(unit_id)

                assigned_task = self.bots_task[unit_id]
                if len(self.opp_botpos) != 0 and opponent_min_distance == 1 and unit.unit_type == "HEAVY" and assigned_task != "kill":
                    assigned_task = "kill"
                    prx(PREFIX, 'from', factory_belong, unit.unit_type, unit.pos, 'temporarly tasked as', assigned_task, opponent_pos_min_distance, opponent_min_distance)


                #if turn_left<150 and assigned_task != "ore":
                #    self.bots_task[unit_id] = 'rubble'
                #    assigned_task = self.bots_task[unit_id]

                if assigned_task == "ice":
                    if unit.cargo.ice < unit.cargo_space() and unit.power > unit.action_queue_cost(game_state) + unit.dig_cost(
                            game_state) + unit.def_move_cost() * distance_to_factory:

                        # get closest ice
                        closest_ice, sorted_ice = self.get_map_distances(ice_locations, unit.pos)

                        # if we have reached the ice tile, start mining if possible
                        if np.all(closest_ice == unit.pos):
                            if unit.power >= unit.dig_cost(game_state) + \
                                    unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.dig(repeat=False)]
                        else:
                            direction = self.get_direction(unit, closest_ice, sorted_ice)
                            move_cost = unit.move_cost(game_state, direction)

                    elif unit.cargo.ice >= unit.cargo_space() or unit.power <= unit.action_queue_cost(
                            game_state) + unit.dig_cost(game_state) + unit.def_move_cost() * distance_to_factory:

                        if adjacent_to_factory:
                            if unit.cargo.ice > 0:
                                actions[unit_id] = [unit.transfer(0, 0, unit.cargo.ice, repeat=False)]
                            elif unit.cargo.ore > 0:
                                actions[unit_id] = [unit.transfer(0, 1, unit.cargo.ore, repeat=False)]
                            elif unit.power < unit.battery_capacity() * 0.1:
                                actions[unit_id] = [unit.pickup(4, unit.battery_capacity() - unit.power)]
                        else:
                            direction = self.get_direction(unit, closest_factory_tile, sorted_factory)
                            move_cost = unit.move_cost(game_state, direction)

                elif assigned_task == 'ore':
                    if unit.cargo.ore < unit.cargo_space() and unit.power > unit.action_queue_cost(game_state) + unit.dig_cost(
                            game_state) + unit.def_move_cost() * distance_to_factory:

                        # get closest ore
                        closest_ore, sorted_ore = self.get_map_distances(ore_locations, unit.pos)

                        # if we have reached the ore tile, start mining if possible
                        if np.all(closest_ore == unit.pos):
                            if unit.power >= unit.dig_cost(game_state) + \
                                    unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.dig(repeat=False)]
                        else:
                            direction = self.get_direction(unit, closest_ore, sorted_ore)
                            move_cost = unit.move_cost(game_state, direction)

                    elif unit.cargo.ore >= unit.cargo_space() or unit.power <= unit.action_queue_cost(
                            game_state) + unit.dig_cost(game_state) + unit.def_move_cost() * distance_to_factory:

                        if adjacent_to_factory:
                            if unit.cargo.ore > 0:
                                actions[unit_id] = [unit.transfer(0, 1, unit.cargo.ore, repeat=False)]
                            elif unit.cargo.ice > 0:
                                actions[unit_id] = [unit.transfer(0, 0, unit.cargo.ice, repeat=False)]
                            elif unit.power < unit.battery_capacity() * 0.1:
                                actions[unit_id] = [unit.pickup(4, unit.battery_capacity() - unit.power)]
                        else:
                            direction = self.get_direction(unit, closest_factory_tile, sorted_factory)
                            move_cost = unit.move_cost(game_state, direction)
                # RUBBLE
                elif assigned_task == 'rubble':
                    if unit.can_dig(game_state):

                        # compute the distance to each rubble tile from this unit and pick the closest
                        closest_rubble, sorted_rubble = self.get_map_distances(rubble_locations, unit.pos)

                        # if we have reached the rubble tile, start mining if possible
                        if np.all(closest_rubble == unit.pos) or rubble_map[unit.pos[0], unit.pos[1]] != 0:
                            if unit.power >= unit.dig_cost(game_state) + \
                                    unit.action_queue_cost(game_state):
                                actions[unit_id] = [unit.dig(repeat=False)]
                        else:
                            if len(rubble_locations) != 0:
                                direction = self.get_direction(unit, closest_rubble, sorted_rubble)
                                move_cost = unit.move_cost(game_state, direction)

                    elif unit.power <= unit.action_queue_cost(game_state) + unit.dig_cost(game_state) + unit.rubble_dig_cost():

                        if adjacent_to_factory:
                            if unit.cargo.ore > 0:
                                actions[unit_id] = [unit.transfer(0, 1, unit.cargo.ore, repeat=False)]
                            elif unit.cargo.ice > 0:
                                actions[unit_id] = [unit.transfer(0, 0, unit.cargo.ice, repeat=False)]
                            elif unit.power < unit.battery_capacity() * 0.1:
                                actions[unit_id] = [unit.pickup(4, unit.battery_capacity() - unit.power)]
                        else:
                            direction = self.get_direction(unit, closest_factory_tile, sorted_factory)
                            move_cost = unit.move_cost(game_state, direction)
                elif assigned_task == 'kill':

                    if len(self.opp_botpos) == 0:
                        if len(lichen_opposite_locations) >0:
                            # compute the distance to each rubble tile from this unit and pick the closest
                            closest_opposite_lichen, sorted_opp_lichen = self.get_map_distances(lichen_opposite_locations, unit.pos)

                            # if we have reached the lichen tile, start mining if possible
                            if np.all(closest_opposite_lichen == unit.pos):
                                if unit.power >= unit.dig_cost(game_state) + \
                                        unit.action_queue_cost(game_state):
                                    actions[unit_id] = [unit.dig(repeat=False)]
                            else:
                                direction = self.get_direction(unit, closest_opposite_lichen, sorted_opp_lichen)
                                move_cost = unit.move_cost(game_state, direction)

                    if len(self.opp_botpos) != 0:
                        if opponent_min_distance == 1:
                            direction = self.get_direction(unit, np.array(opponent_pos_min_distance),
                                                           [np.array(opponent_pos_min_distance)])
                            move_cost = unit.move_cost(game_state, direction)
                        else:
                            if unit.power > unit.action_queue_cost(game_state):
                                direction = self.get_direction(unit, np.array(opponent_pos_min_distance),
                                                               [np.array(opponent_pos_min_distance)])
                                move_cost = unit.move_cost(game_state, direction)
                            else:
                                if adjacent_to_factory:
                                    if unit.cargo.ore > 0:
                                        actions[unit_id] = [unit.transfer(0, 1, unit.cargo.ore, repeat=False)]
                                    elif unit.cargo.ice > 0:
                                        actions[unit_id] = [unit.transfer(0, 0, unit.cargo.ice, repeat=False)]
                                    elif unit.power < unit.battery_capacity() * 0.1:
                                        actions[unit_id] = [unit.pickup(4, unit.battery_capacity() - unit.power)]
                                else:
                                    direction = self.get_direction(unit, closest_factory_tile, sorted_factory)
                                    move_cost = unit.move_cost(game_state, direction)

                # check move_cost is not None, meaning that direction is not blocked
                # check if unit has enough power to move and update the action queue.
                if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
                    actions[unit_id] = [unit.move(direction, repeat=False)]

        return actions

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
        botpos = {}
        botposheavy = {}
        opp_botpos = []
        opp_botposheavy = []
        for player in [self.player, self.opp_player]:
            for unit_id, unit in game_state.units[player].items():

                if player == self.player:
                    botpos[unit_id] = str(unit.pos)
                    if unit.unit_type == "HEAVY":
                        botposheavy[unit_id] = str(unit.pos)
                else:
                    opp_botpos.append(unit.pos)
                    if unit.unit_type == "HEAVY":
                        opp_botposheavy.append(unit.pos)

        return botpos, botposheavy, opp_botpos, opp_botposheavy
