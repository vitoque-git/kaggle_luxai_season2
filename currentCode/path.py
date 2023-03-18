import networkx as nx
from luxai_s2.env import LuxAI_S2
import matplotlib.pyplot as plt
import numpy as np


def pr(*args, sep=' ', end='\n', force=False):  # print conditionally
    if (True or force): # change first parameter to False to disable logging
        print(*args, sep=sep, file=sys.stderr)

def prx(*args): pr(*args, force=True)

class Path_Finder():
    def __init__(self) -> None:
        self.G = nx.Graph()

    def get_opp_factories_areas(self, game_state, opp_player):
        opp_factories = [np.array(f.pos) for _, f in game_state.factories[opp_player].items()]
        opp_factories_areas = []
        for pos in opp_factories:
            # prx('city',pos)
            x = pos[0]
            y = pos[1]
            opp_factories_areas.append((x - 1, y - 1))
            opp_factories_areas.append((x - 1, y))
            opp_factories_areas.append((x - 1, y + 1))
            opp_factories_areas.append((x, y - 1))
            opp_factories_areas.append((x, y))
            opp_factories_areas.append((x, y + 1))
            opp_factories_areas.append((x + 1, y - 1))
            opp_factories_areas.append((x + 1, y))
            opp_factories_areas.append((x + 1, y + 1))
        return opp_factories_areas

    def build_path(self,game_state,opp_player):
        opp_factories_areas = self.get_opp_factories_areas(game_state,opp_player)
        rubbles = game_state.board.rubble
        return self._build_path(rubbles, opp_factories_areas)

    def _build_path(self, rubbles, prohibited_locations=[]):

        self.G = nx.Graph()

        add_delta = lambda a: tuple(np.array(a[0]) + np.array(a[1]))

        for x in range(rubbles.shape[0]):
            for y in range(rubbles.shape[1]):
                self.G.add_node((x, y), rubble=rubbles[x, y])

        # prx("Nodes created.")

        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for g1 in self.G.nodes:
            x1, y1 = g1
            for delta in deltas:
                g2 = add_delta((g1, delta))
                if g2 in prohibited_locations:
                    self.G.add_edge(g1, g2, cost=10e6)
                elif self.G.has_node(g2) :
                    self.G.add_edge(g1, g2, cost=20 + rubbles[g2])
        # prx("Edges created.")

    def get_shortest_path(self, from_pt, to_pt):
        ptA = (from_pt[0], from_pt[1])
        ptB = (to_pt[0], to_pt[1])
        path = nx.shortest_path(self.G, source=ptA, target=ptB, weight="cost")
        return path;

def test():
    print("TEST MODE FOR PATH_FINDER")
    PIC_SIZE = 1024

    env = LuxAI_S2()
    obs = env.reset(seed=22)
    rubbles = obs["player_0"]["board"]["rubble"]
    img = env.render("rgb_array", width=PIC_SIZE, height=PIC_SIZE)
    #
    # plt.figure(figsize=(6,6))
    # plt.imshow(img)
    # plt.show()

    PF = Path_Finder()
    PF._build_path(rubbles)

    #random points
    all_pts = [(x, y) for x in range(0, 48) for y in range(0, 48)]
    np.random.shuffle(all_pts)
    ptA, ptB = all_pts[0], all_pts[1]

    ptA, ptB = (14, 24), (17, 26)

    path = PF.get_shortest_path(ptA,ptA);

    print('from', ptA,'to',ptB)
    print(path)
    # output graph
    scale = lambda a: (a + .5) / 48 * PIC_SIZE

    plt.figure(figsize=(6, 6))
    cA = plt.Circle((scale(ptA[0]), scale(ptA[1])), scale(0), color="white")
    plt.gca().add_patch(cA)
    cB = plt.Circle((scale(ptB[0]), scale(ptB[1])), scale(0), color="red")
    plt.gca().add_patch(cB)
    plt.plot([scale(p[0]) for p in path], [scale(p[1]) for p in path], c="lime")

    plt.imshow(img, alpha=0.9)
    plt.show()

    print("Finished")

test()