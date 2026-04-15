# algorithms/standard_plain.py
# Fully self-contained Standard Visibility-Graph A* planner.
# NO penalties, NO predecessor logic, NO imports from fundamental.py or standard.py.
# Matches the temp folder's standard-mode behavior: random corner offsets,
# simple line-of-sight, plain shortest-path A*.

import math
import heapq
import random
from config import *


# ---------------------------------------------------------------------------
# Local utilities (no import from fundamental)
# ---------------------------------------------------------------------------
def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def _clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))


# ---------------------------------------------------------------------------
# Self-contained A* core
# ---------------------------------------------------------------------------
def _astar_core(start, goal, get_neighbors, heuristic):
    open_set = [(0, 0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    counter = 1

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for neighbor, cost in get_neighbors(current):
            new_g = g_score[current] + cost
            if neighbor not in g_score or new_g < g_score[neighbor]:
                g_score[neighbor] = new_g
                f = new_g + heuristic(neighbor)
                heapq.heappush(open_set, (f, counter, neighbor))
                counter += 1
                came_from[neighbor] = current

    return None


def _astar_plain(start_idx, goal_idx, graph, nodes, goal_pos):
    """Plain A* over a visibility graph — no penalties."""
    def get_neighbors(idx):
        return graph[idx]

    def heuristic(idx):
        return _dist(nodes[idx], goal_pos)

    path_indices = _astar_core(start_idx, goal_idx, get_neighbors, heuristic)
    if path_indices is None:
        return None
    return [nodes[i] for i in path_indices]


# ---------------------------------------------------------------------------
# Standard Plain A* Planner (self-contained visibility-graph)
# ---------------------------------------------------------------------------
class StandardPlainPlanner:
    """
    Pure shortest-path visibility-graph A*.
    No penalty logic, no predecessor awareness.
    Random corner offsets for slight path variation (matches temp folder behavior).
    """

    def __init__(self, obstacles):
        self.obstacles = obstacles
        self._bloat(obstacles)

    def _bloat(self, obstacles):
        inflation = AGENT_DIAMETER + 6
        self.bloated_obstacles = [obs.inflate(inflation, inflation) for obs in obstacles]

    def update_obstacles(self, obstacles):
        self.obstacles = obstacles
        self._bloat(obstacles)

    def _get_corners(self):
        """Generate visibility-graph corner nodes with random offsets."""
        margin = 10.0
        nodes = []

        for obs in self.bloated_obstacles:
            left_ok   = obs.left   >= 0
            right_ok  = obs.right  <= MAP_WIDTH
            top_ok    = obs.top    >= 0
            bottom_ok = obs.bottom <= MAP_HEIGHT

            # Random offsets per corner (matching temp folder behavior)
            m_tl = random.uniform(2.0, WIDE_CORNER_MAX_MARGIN)
            m_tr = random.uniform(2.0, WIDE_CORNER_MAX_MARGIN)
            m_br = random.uniform(2.0, WIDE_CORNER_MAX_MARGIN)
            m_bl = random.uniform(2.0, WIDE_CORNER_MAX_MARGIN)

            candidate_corners = []
            if left_ok and top_ok:
                candidate_corners.append((obs.left  - 2.0,  obs.top    - 2.0))
                candidate_corners.append((obs.left  - m_tl, obs.top    - m_tl))
            if right_ok and top_ok:
                candidate_corners.append((obs.right + 2.0,  obs.top    - 2.0))
                candidate_corners.append((obs.right + m_tr, obs.top    - m_tr))
            if right_ok and bottom_ok:
                candidate_corners.append((obs.right + 2.0,  obs.bottom + 2.0))
                candidate_corners.append((obs.right + m_br, obs.bottom + m_br))
            if left_ok and bottom_ok:
                candidate_corners.append((obs.left  - 2.0,  obs.bottom + 2.0))
                candidate_corners.append((obs.left  - m_bl, obs.bottom + m_bl))

            for (x, y) in candidate_corners:
                cx = _clamp(x, margin, MAP_WIDTH  - margin)
                cy = _clamp(y, margin, MAP_HEIGHT - margin)
                if all(not o.collidepoint(cx, cy) for o in self.bloated_obstacles):
                    nodes.append((cx, cy))

        return nodes

    def _is_line_clear(self, start, end, agent_start_pos):
        p1 = (float(start[0]), float(start[1]))
        p2 = (float(end[0]),   float(end[1]))

        is_start_edge = (start == agent_start_pos or end == agent_start_pos)

        for obs in self.bloated_obstacles:
            if obs.clipline(p1, p2):
                if not is_start_edge:
                    return False
                if is_start_edge and not obs.collidepoint(agent_start_pos):
                    return False

            if start != agent_start_pos and obs.collidepoint(p1): return False
            if end   != agent_start_pos and obs.collidepoint(p2): return False

        return True

    def find_path(self, agent, end_pos, predecessor_paths=None):
        """Find shortest obstacle-avoiding path.
        predecessor_paths is accepted for interface compatibility but ignored."""
        start_pos = agent.pos

        corner_nodes = self._get_corners()
        nodes = [start_pos, end_pos] + corner_nodes

        valid_nodes = []
        for n in nodes:
            safe = True
            for obs in self.bloated_obstacles:
                if obs.collidepoint(n):
                    safe = False
                    break
            if safe or n == start_pos or n == end_pos:
                valid_nodes.append(n)

        nodes = valid_nodes
        graph = {i: [] for i in range(len(nodes))}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self._is_line_clear(nodes[i], nodes[j], start_pos):
                    d = _dist(nodes[i], nodes[j])
                    graph[i].append((j, d))
                    graph[j].append((i, d))

        return _astar_plain(0, 1, graph, nodes, end_pos)


# ---------------------------------------------------------------------------
# Standard Plain Central Manager
# ---------------------------------------------------------------------------
class StandardPlainManager:
    """
    Assigns each agent a shortest obstacle-avoiding path.
    No priority ordering, no predecessor penalties.
    """

    def __init__(self, planner, exit_manager):
        self.planner = planner
        self.exit_manager = exit_manager

    def plan_all_paths(self, agents):
        if not self.planner or not agents:
            return

        goal_pos = self.exit_manager.center_pixel

        for i, agent in enumerate(agents):
            if not agent.active or agent.spot_reserved:
                continue

            agent.index = i + 1
            path = self.planner.find_path(agent, goal_pos)
            if path:
                agent.path = path
                agent.path_valid = True
                agent.current_wp_index = 0
            else:
                agent.path_valid = False
                agent.path = []
