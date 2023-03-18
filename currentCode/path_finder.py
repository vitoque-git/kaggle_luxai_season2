import sys

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
        self.rubbles = []
        self.prohibited_locations=[]
        self.deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]


    def get_opp_factories_areas(self, game_state, opp_player):
        opp_factories = [np.array(f.pos) for _, f in game_state.factories[opp_player].items()]
        opp_factories_areas = []
        for pos in opp_factories:
            # prx('city',pos)
            Path_Finder.expand_point(opp_factories_areas, pos)
        return opp_factories_areas

    def expand_point(opp_factories_areas, pos):
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

    def build_path(self,game_state,player,opp_player):
        opp_factories_areas = self.get_opp_factories_areas(game_state,opp_player)
        rubbles = game_state.board.rubble
        return self._build_path(rubbles, opp_factories_areas)

    def _build_path(self, rubbles, prohibited_locations=[]):

        self.G = nx.Graph()
        self.rubbles = rubbles
        self.prohibited_locations = prohibited_locations

        for x in range(self.rubbles.shape[0]):
            for y in range(self.rubbles.shape[1]):
                self._add_node(x, y)

        # prx("Nodes created.")

        add_delta = lambda a: tuple(np.array(a[0]) + np.array(a[1]))
        for g1 in self.G.nodes:
            self._add_edge(add_delta, g1)
        # prx("Edges created.")

    def _add_node(self, x, y):
        if (x, y) in self.prohibited_locations:
            # this makes impossible direction to enemy with prohibited_locations
            pass
        else:
            self.G.add_node((x, y), rubble=self.rubbles[x, y])

    def _add_edge(self, add_delta, g1):
        if self.G.has_node(g1):
            for delta in self.deltas:
                g2 = add_delta((g1, delta))
                if self.G.has_node(g2):
                    self.G.add_edge(g1, g2, cost=20 + self.rubbles[g2])


    def _exclude_nodes(self,points_to_exclude=[]):
        for l in points_to_exclude:
            if self.G.has_node(l):
                self.G.remove_node(l)

    def _re_add_nodes(self,points_to_re_add=[]):
        add_delta = lambda a: tuple(np.array(a[0]) + np.array(a[1]))
        for g1 in points_to_re_add:
            self._add_node(g1[0], g1[1])
            self._add_edge(add_delta, g1)


    def get_shortest_path(self, ptA, ptB, points_to_exclude=[]):
        ptA = (ptA[0], ptA[1])
        ptB = (ptB[0], ptB[1])

        # we remove nodes that only for this path we do not want
        # used for the adjacent collision that we want to remove
        self._exclude_nodes(points_to_exclude)
        try:
            path = nx.shortest_path(self.G, source=ptA, target=ptB, weight="cost")
        except:
            self._re_add_nodes(points_to_exclude)
            return [ptA[0]]
        self._re_add_nodes(points_to_exclude)

        return path;

    def has_node(self, pos):
        return self.G.has_node((pos[0],pos[1]))


import time

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
    pf = Path_Finder()

    opp_factories_areas = []

    # prx('city',pos)
    Path_Finder.expand_point(opp_factories_areas, (16, 21))
    Path_Finder.expand_point(opp_factories_areas, (24, 24))
    st = time.time()
    pf._build_path(rubbles,opp_factories_areas)
    print('_build_path Execution time:', -1000*(st-time.time()), 'ms')


    #random points
    all_pts = [(x, y) for x in range(0, 48) for y in range(0, 48)]
    np.random.shuffle(all_pts)
    ptA, ptB = all_pts[0], all_pts[1]

    ptA, ptB = (14, 24), (27, 26)

    st = time.time()
    path = nx.shortest_path(pf.G, source=ptA, target=ptB, weight="cost")
    print('shortest_path Execution time:', -1000*(st-time.time()), 'ms')

    print('from', ptA,'to',ptB)
    print(path)
    # output graph
    scale = lambda a: (a + .5) / 48 * PIC_SIZE

    plt.figure(figsize=(6, 6))

    plt.plot([scale(p[0]) for p in path], [scale(p[1]) for p in path], c="lime")
    plt.plot([scale(p[0]) for p in path], [scale(p[1]) for p in path], c="lime")
    cA = plt.Circle((scale(ptA[0]), scale(ptA[1])), scale(0), color="white")
    plt.gca().add_patch(cA)
    cB = plt.Circle((scale(ptB[0]), scale(ptB[1])), scale(0), color="red")
    plt.gca().add_patch(cB)
    for l in opp_factories_areas:
        cB = plt.Circle((scale(l[0]), scale(l[1])), scale(0), color="blue")
        plt.gca().add_patch(cB)

    plt.imshow(img, alpha=0.9)
    plt.show()

    print("Finished")

def test2():
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
    pf = Path_Finder()



    st = time.time()
    build_path = pf._build_path(rubbles)
    print('_build_path Execution time:', -1000*(st-time.time()), 'ms')


    #random points
    all_pts = [(x, y) for x in range(0, 48) for y in range(0, 48)]
    np.random.shuffle(all_pts)
    ptA, ptB = all_pts[0], all_pts[1]

    ptA, ptB = (3, 19), (30, 26)

    ptC, ptD = (19, 5), (19, 37)

    points_to_exclude = [(16, 13)]

    st = time.time()



    pf._exclude_nodes(points_to_exclude)
    path = nx.shortest_path(pf.G, source=ptA, target=ptB, weight="cost")
    pf._re_add_nodes(points_to_exclude)

    print('shortest_path Execution time:', -1000*(st-time.time()), 'ms')
    print('from', ptA,'to',ptB)
    print(path)

    st = time.time()
    path2 = nx.shortest_path(pf.G, source=ptC, target=ptD, weight="cost")

    print('shortest_path Execution time:', -1000*(st-time.time()), 'ms')
    print('from', ptC,'to',ptD)
    print(path2)

    # output graph
    scale = lambda a: (a + .5) / 48 * PIC_SIZE

    plt.figure(figsize=(6, 6))

    plt.plot([scale(p[0]) for p in path], [scale(p[1]) for p in path], c="lime")
    plt.plot([scale(p[0]) for p in path], [scale(p[1]) for p in path], c="lime")
    plt.plot([scale(p[0]) for p in path2], [scale(p[1]) for p in path2], c="lime")
    plt.plot([scale(p[0]) for p in path2], [scale(p[1]) for p in path2], c="lime")
    cA = plt.Circle((scale(ptA[0]), scale(ptA[1])), scale(0), color="white")
    plt.gca().add_patch(cA)
    cB = plt.Circle((scale(ptB[0]), scale(ptB[1])), scale(0), color="red")
    plt.gca().add_patch(cB)
    cC = plt.Circle((scale(ptC[0]), scale(ptC[1])), scale(0), color="white")
    plt.gca().add_patch(cC)
    cD = plt.Circle((scale(ptD[0]), scale(ptD[1])), scale(0), color="red")
    plt.gca().add_patch(cD)
    for l in points_to_exclude:
        cB = plt.Circle((scale(l[0]), scale(l[1])), scale(0), color="blue")
        plt.gca().add_patch(cB)

    plt.imshow(img, alpha=0.9)
    plt.show()

    print("Finished")
# test2()