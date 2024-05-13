from collections import deque

import os
import numpy as np
import numpy.ma as ma
from numpy.lib.stride_tricks import sliding_window_view
import heapq
import time
from time import time as tik
import numba as nb
from numba import prange
import networkx as nx
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkAgg')

# @nb.jit(nopython=True, nogil=True, parallel=True, cache=True)
def min(data):
    x, y, z = data.shape
    mins = np.inf * np.ones((x, y))
    for i in prange(x):
        for j in prange(y):
            for k in prange(z):
                if mins[i, j] > data[i, j, k]:
                    mins[i, j] = data[i, j, k]
    return mins


def euclidean(t):
    return np.sqrt(np.sum(np.power(t, 2), axis=-1))


def manhattan(t):
    return np.sum(np.abs(t), axis=-1)


def precompute_distances(goals, to, distance):
    to_sh = list(to.shape)
    goals_sh = list(goals.shape)
    to_n_addr_dims = len(to.shape) - 1
    goals_n_addr_dims = len(goals.shape) - 1
    n_dims = to_sh[-1]
    to = to.reshape(to_sh[:-1] + [1] * to_n_addr_dims + [n_dims])
    goals = goals.reshape([1] * goals_n_addr_dims + goals_sh[:-1] + [n_dims])
    t = goals - to
    distances = distance(t).reshape(to_sh[:-1] + goals_sh[:-1])
    return np.round(distances)


def goals_from_mask(mask):
    return np.vstack(np.where(mask))


SQRT2 = np.sqrt(2)


def d_cost(fr, to):
    x1, y1 = fr
    x2, y2 = to
    return 1 if x1 == x2 or y1 == y2 else SQRT2  # 1.378225873596


def precompute_moves():
    move = np.vstack([np.repeat([-1, 0, 1], 3), np.tile([-1, 0, 1], 3)]).T
    move = np.delete(move, 4, axis=0)
    return move


def get_neighbours(cells, bounds):
    dx = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dy = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    x, y = cells
    x, y = x[..., None] + dx, y[..., None] + dy
    x_in_bounds = np.logical_and(x < bounds[0], x > -1)
    y_in_bounds = np.logical_and(y < bounds[1], y > -1)
    in_bounds = np.logical_and(x_in_bounds, y_in_bounds)
    return (x, y), in_bounds


def make_lookup(goals, distance_func):
    #all2all = precompute_distances(goals.shape, distance_func)
    #all2all_padded = np.pad(all2all, ((0,0), (0,0), (1, 1), (1, 1)))
    #all2all_window = sliding_window_view(all2all_padded, (3, 3), axis=(-2, -1))
    h, w = goals.shape
    goals_padded = np.pad(goals, ((1, 1), (1, 1)))
    goals = goals_padded[1:-1, 1:-1]
    goals_windowed = sliding_window_view(goals_padded, (3, 3))
    visited_padded = np.zeros((h + 2, w + 2), dtype=bool)
    visited = visited_padded[1:-1, 1:-1]
    visited_windowed = sliding_window_view(visited_padded, (3, 3))
    visited[goals] = True
    closest_padded = np.zeros((h + 2, w + 2, 2), dtype=int)
    closest = closest_padded[1:-1, 1:-1]
    closest_windowed = sliding_window_view(closest_padded, (3, 3), axis=(0, 1))
    print(closest_windowed.shape)
    closest[goals, 0], closest[goals, 1] = np.where(goals)
    distances = np.ones(goals.shape) * np.inf
    distances[goals] = 0
    front_padded = np.zeros((h + 2, w + 2), dtype=bool)
    front = front_padded[1:-1, 1:-1]
    front_windowed = sliding_window_view(front_padded, (3, 3), writeable=True)

    while visited.sum() < visited.size:
        #print(front_windowed.shape)
        front_windowed[goals] = True
        front[visited] = False
        '''cl = closest_windowed[front]
        temp = np.ones((front.sum(), 3, 3)) * np.inf
        print(temp.shape, cl.shape, visited_windowed[front].shape, temp[visited_windowed[front]].shape, all2all[front].shape)
        temp[visited_windowed[front]] = all2all[front, cl]
        print(temp.shape)'''

        # front necomes new goals
        goals = np.zeros_like(goals, dtype=bool)
        goals[front] = True
        goals[visited] = False
        visited[goals] = True
        cv2.imshow("front", (front*255).astype(np.uint8))
        time.sleep(.03)
        if cv2.waitKey(1) == ord('q'):
            # press q to terminate the loop
            cv2.destroyAllWindows()
            break
        front_padded[:] = False
    return None


STATE_EMPTY = 0
STATE_WALL = 1
STATE_FRONT = 2
STATE_SEARCH = 3
STATE_PATH = 4
STATE_START = 5
STATE_REACHED_GOAL = 6
STATE_GOAL_BARRIER = 7


class MapSearch:

    def __init__(self, map):
        self.map_state = None
        self.came_from = None
        self.goals = None
        self.h_lookup = None
        self.heuristic = None
        self.heap = None
        self.g_cost = None
        self.shortest_path = (None, None)
        self.reached_goal = None
        self.start_pos = None
        self.map_view = map
        self.d_funcs = {"euclidian": euclidean, "manhattan": manhattan}
        self.moves = precompute_moves()
        self.heuristic = "euclidian"
        self.reset()

    def reset(self):
        self.map_state = self.map_view.copy().astype(int)
        self.came_from = -np.ones(list(self.map_view.shape) + [2]).astype(int)
        self.g_cost = np.ones(self.map_view.shape) + float("Inf")
        self.heap = []

    def set_start_goals(self, start, goals, precompute=True):
        self.goals = goals
        self.start_pos = start
        heapq.heappush(self.heap, (float("Inf"), 0, start))

        if precompute or self.h_lookup is None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            goals = goals - (cv2.erode(goals * 1., kernel, iterations=1))
            goals = goals == 1
            ds_to_goals = self.precompute_distance_lookup(goals).min(axis=-1)
            self.h_lookup = ds_to_goals

    def precompute_distance_lookup(self, goals):
        goals = goals_from_mask(goals).T
        sample = np.random.choice(goals.shape[0], int(goals.shape[0]*1.))
        goals = goals[sample]
        to_all = np.indices(self.map_view.shape).T
        h, w = self.map_view.shape
        return precompute_distances(goals, to_all, self.d_funcs[self.heuristic]).reshape(h, w, -1)

    def f_cost(self, fr, to):
        _, g_cost, fr_pos = fr
        f_cost = (g_cost + d_cost(fr_pos, to) + self.h_lookup[fr_pos])
        return f_cost

    def generate_neighbours(self, from_node):
        _, g_cost, from_pos = from_node
        x, y = from_pos
        n, s, w, e = (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)
        nw, sw, sw, ne = (x - 1, y - 1), (x + 1, y - 1), (x + 1, y - 1), (x + 1, y + 1)
        possible = [n, s, w, e, nw, sw, sw, ne]
        possible = [(self.f_cost(from_node, to), g_cost + d_cost(from_pos, to), to) for to in possible]
        generated = [node for node in possible if self.do_generate(node, from_pos)]
        return generated

    def do_generate(self, node, from_pos):
        f_cost, g_cost, (x, y) = node
        xm, ym = self.map_view.shape
        generate = -1 < x < xm and -1 < y < ym
        generate = generate and (self.map_view[x, y] == STATE_EMPTY or self.map_view[x, y] == STATE_GOAL_BARRIER or
                                 (self.goals[x, y] == 1)) #and self.map_view[from_pos] != STATE_GOAL_BARRIER
        generate = generate and self.g_cost[x, y] > g_cost
        return generate

    def search(self):
        while len(self.heap) > 0:
            current_node = heapq.heappop(self.heap)
            _, _, pos = current_node
            self.map_state[pos] = STATE_SEARCH
            if self.goals[pos] == 1:
                self.map_state[pos] = STATE_REACHED_GOAL
                self.reached_goal = pos
                self.backtrace(pos)
                return self.shortest_path
            for neighbour in self.generate_neighbours(current_node):
                f_cost, g_cost, (x, y) = neighbour
                self.map_state[x, y] = STATE_FRONT
                self.came_from[x, y] = list(pos)
                self.g_cost[x, y] = g_cost
                heapq.heappush(self.heap, neighbour)
        return None, None

    def img_mask(self, id, color):
        mask = np.repeat(self.map_state[..., None], 3, axis=-1) == id
        img = mask * color
        return mask, img

    def search_result_image(self):
        _, bg = self.img_mask(0, [245, 245, 245])
        search_mask, search = self.img_mask(STATE_SEARCH, [175, 238, 238])
        bg[search_mask] = search[search_mask]
        path_mask, path = self.img_mask(STATE_PATH, [255, 255, 0])
        bg[path_mask] = path[path_mask]
        front_mask, front = self.img_mask(STATE_FRONT, [152, 251, 152])
        bg[front_mask] = front[front_mask]
        start_mask, start = self.img_mask(STATE_START, [0, 221, 0])
        bg[start_mask] = start[start_mask]
        reached_goal_mask, reached_goal = self.img_mask(STATE_REACHED_GOAL, [238, 68, 0])
        bg[reached_goal_mask] = reached_goal[reached_goal_mask]
        goal_barrier_mask, goal_barrier = self.img_mask(STATE_GOAL_BARRIER, [255, 88, 20])
        bg[goal_barrier_mask] = goal_barrier[goal_barrier_mask]
        '''cv2.imwrite("test.png", cv2.cvtColor(bg.astype("uint8"), cv2.COLOR_RGB2BGR))
        plt.imshow(bg)
        plt.show()'''
        return bg.astype(np.uint8)

    def backtrace(self, prev):
        cost = 0
        path = []
        while True:
            if prev == self.start_pos:
                self.map_state[prev] = STATE_START
                break
            path.append(prev)
            now = tuple(self.came_from[prev])
            self.map_state[now] = 4
            cost += d_cost(prev, now)
            prev = now
        self.shortest_path = (path, cost)


def node2coords(node, shape):
    h, w = shape
    x = int(node / h)
    y = int(node - h * x)
    return x, y


def coords2node(x, y, shape):
    h, w = shape
    return h * x + y


def gen_connections(x, y, map):
    h, w = map.shape
    sqrt2 = np.sqrt(2)
    cost = np.array([sqrt2, 1, sqrt2, 1, 1, sqrt2, 1, sqrt2])
    dx = np.array([-1, -1, -1, 0, 0, 1, 1, 1]) + x
    dy = np.array([-1, 0, 1, -1, 1, -1, 0, 1]) + y
    x_in_bounds = np.logical_and(dx < h, dx > -1)
    y_in_bounds = np.logical_and(dy < w, dy > -1)
    in_bounds = np.logical_and(x_in_bounds, y_in_bounds)
    cnn = np.dstack((dx, dy))[0, in_bounds]
    barrier = map[cnn[:, 0].astype(int), cnn[:, 1]]
    return cnn[barrier], cost[in_bounds][barrier]


if __name__ == "__main__":
    map = np.zeros((250, 250), dtype=bool)

    xi, yi = np.indices(map.shape)
    goals = np.zeros_like(map)
    circle = np.sqrt((xi-125)**2 + (yi-120)**2) < 100
    map[circle] = True
    circle = np.sqrt((xi-25)**2 + (yi-25)**2) < 15
    map[circle] = True
    circle = np.sqrt((xi-25)**2 + (yi-150)**2) < 35
    map[circle] = True
    circle = np.sqrt((xi-200)**2 + (yi-200)**2) < 45
    map[circle] = True
    map[circle] = True
    s = np.ones_like(map)

    ms = MapSearch(s)
    ms.set_start_goals([0,0], map)

    exit(9)
    t = tik()
    make_lookup(map, None)
    print(tik() - t)


    ''' G = nx.grid_graph(dim=map.shape)
    
    xs, ys = np.where(np.ones_like(map, dtype=bool))
    G.add_nodes_from(zip(xs, ys))
    
    t = tik()
    for x, y in zip(xs, ys):
        cnn, cost = gen_connections(x, y, map)
        for (cx, cy), c in zip(cnn, cost):
            G.add_edge((x, y), (cx, cy), cost=c)
    print(tik() - t)
    
    t = tik()
    nx.astar_path(G, (0, 0), (499, 499), heuristic=lambda a, b: 0, weight="cost")
    print(tik() - t)'''




