import numpy as np

from utils import *


class PlayerHelper():
    def __init__(self) -> None:
        # Initial loop to set present and future locations of units
        self.unit_next_positions = {}  # index unit id, payload position
        self.unit_current_positions = {}  # index position, payload unit object
        self.light_current_positions = {}  # index position, payload unit object
        self.heavy_current_positions = {}  # index position, payload unit object

        self.factory_positions = {}  # index location, payload factory
        self.factory_areas = [] #list of positions
        self.lichen_locations = {} # NP of loc

    def set_player(self,game_state, player):
        units = game_state.units[player]
        factories = game_state.factories[player]

        self.__init__() #reset everything

        for unit_id, factory in factories.items():
            self.factory_positions[factory.pos_location()] = factory
            expand_point(self.factory_areas, factory.pos)
            if len(self.lichen_locations) == 0:
                self.lichen_locations = np.argwhere(game_state.board.lichen_strains == factory.strain_id)
            else:
                newarray = np.argwhere(game_state.board.lichen_strains == factory.strain_id)
                if len(newarray) > 0:
                    self.lichen_locations = np.vstack((self.lichen_locations, newarray))

        for unit_id, unit in iter(sorted(units.items())):
            # default next position to current position, we will modify then in case of movements
            location = unit.pos_location()
            self.unit_next_positions[unit.unit_id] = location
            self.unit_current_positions[location] = unit

            if unit.unit_type == "HEAVY":
                self.heavy_current_positions[location] = unit
            else:
                self.light_current_positions[location] = unit


    def get_factories_areas(self):
        return self.factory_areas


    def get_unit_positions(self):
        return self.unit_current_positions.keys()

    def get_light_positions(self):
        return self.light_current_positions.keys()

    def get_heavy_positions(self):
        return self.heavy_current_positions.keys()

    def get_unit_next_positions(self):
        return self.unit_next_positions.values()

    def get_unit_from_current_position(self, pos):
        return self.unit_current_positions[(pos[0],pos[1])]

    def set_unit_next_position(self,unit_id, new_pos):
        self.unit_next_positions[unit_id] = (new_pos[0], new_pos[1])

    def get_num_lights(self):
        return len(self.get_light_positions())

    def get_num_heavy(self):
        return len(self.get_heavy_positions())

    def get_num_units(self):
        return len(self.get_unit_positions())

    def is_factory_center(self, pos):
        return (pos[0],pos[1]) in self.factory_positions

    def is_factory_area(self, pos):
        return (pos[0],pos[1]) in self.factory_areas

    def get_lichen_amount(self, game_state, pos):
        if pos in self.lichen_locations:
            return game_state.board.lichen[pos[0], pos[1]]
        else:
            return 0
