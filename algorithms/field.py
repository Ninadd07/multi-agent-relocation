# algorithms/field.py
# Fully self-contained Electric Field planner.
# This module has its own visibility-graph A* pathfinding, its own manager,
# and its own per-tick update logic — NO imports from standard.py or
# fundamental.py (except config constants).

import pygame
import math
import heapq
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
    """Plain A* — no penalties, no predecessor logic."""
    def get_neighbors(idx):
        return graph[idx]

    def heuristic(idx):
        return _dist(nodes[idx], goal_pos)

    path_indices = _astar_core(start_idx, goal_idx, get_neighbors, heuristic)
    if path_indices is None:
        return None
    return [nodes[i] for i in path_indices]


# ---------------------------------------------------------------------------
# Electric Field Planner (self-contained visibility-graph A*)
# ---------------------------------------------------------------------------
class ElectricFieldPlanner:
    """
    Visibility-graph A* planner for the Electric Field algorithm.
    Generates obstacle-avoiding waypoint paths that agents then follow
    using electric-field force steering.
    
    No penalty logic — every agent gets the same shortest obstacle-free path.
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
        """Generate visibility-graph corner nodes with fixed offsets."""
        offsets = [2.0, float(WIDE_CORNER_MAX_MARGIN)]
        margin = 10.0
        nodes = []

        for obs in self.bloated_obstacles:
            left_ok   = obs.left   >= 0
            right_ok  = obs.right  <= MAP_WIDTH
            top_ok    = obs.top    >= 0
            bottom_ok = obs.bottom <= MAP_HEIGHT

            raw_corners = []
            if left_ok  and top_ok:    raw_corners.append((obs.left,  obs.top,    -1, -1))
            if right_ok and top_ok:    raw_corners.append((obs.right, obs.top,    +1, -1))
            if right_ok and bottom_ok: raw_corners.append((obs.right, obs.bottom, +1, +1))
            if left_ok  and bottom_ok: raw_corners.append((obs.left,  obs.bottom, -1, +1))

            for (cx, cy, dx, dy) in raw_corners:
                for m in offsets:
                    x = _clamp(cx + dx * m, margin, MAP_WIDTH  - margin)
                    y = _clamp(cy + dy * m, margin, MAP_HEIGHT - margin)
                    if all(not o.collidepoint(x, y) for o in self.bloated_obstacles):
                        nodes.append((x, y))

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
        """Find an obstacle-avoiding path. predecessor_paths is accepted
        for interface compatibility but ignored (no penalties)."""
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
# Electric Field Central Manager
# ---------------------------------------------------------------------------
class ElectricFieldManager:
    """
    Assigns each agent an obstacle-avoiding path via the ElectricFieldPlanner.
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


# ---------------------------------------------------------------------------
# Per-agent physics tick
# ---------------------------------------------------------------------------
def update_electric(agent, agents, obstacles, end_rect):
    """
    Steer the agent using electric-field forces:
      • Attraction toward the NEXT WAYPOINT on the A*-planned path
      • Coulomb repulsion from other agents
      • Coulomb repulsion from obstacle walls

    The agent follows A*-planned waypoints (so paths avoid obstacles),
    but steering is force-based rather than direct velocity-to-waypoint.
    """
    if not agent.active:
        return

    # --- Parking hand-off -------------------------------------------------
    agent._check_parking_logic(end_rect)

    if agent.spot_reserved:
        agent.update(end_rect)
        return

    if not agent.path_valid:
        agent.resolve_collision(agents, obstacles)
        return

    # --- Attraction toward current waypoint -------------------------------
    if agent.current_wp_index < len(agent.path):
        target = pygame.Vector2(agent.path[agent.current_wp_index])
        desired = target - agent.pos
        attraction = desired.normalize() * K_ATTRACTION if desired.length() > 0 else pygame.Vector2(0, 0)
    else:
        attraction = pygame.Vector2(0, 0)

    # --- Agent–agent repulsion --------------------------------------------
    repulsion = pygame.Vector2(0, 0)
    for other in agents:
        if other is agent:
            continue
        diff = agent.pos - other.pos
        d = diff.length()
        if 0.1 < d < 60:
            repulsion += diff.normalize() * (K_AGENT / (d ** 2))

    # --- Wall repulsion ---------------------------------------------------
    wall_repulsion = pygame.Vector2(0, 0)
    for obs in obstacles:
        cx = _clamp(agent.pos.x, obs.left, obs.right)
        cy = _clamp(agent.pos.y, obs.top,  obs.bottom)
        diff = agent.pos - pygame.Vector2(cx, cy)
        d = diff.length()
        if 0.1 < d < 40:
            wall_repulsion += diff.normalize() * (K_WALL / (d ** 2))

    # --- Combine forces ---------------------------------------------------
    total = attraction + repulsion * 1.5 + wall_repulsion * 2.0
    if total.length() > MAX_FORCE:
        total = total.normalize() * MAX_FORCE

    agent.velocity += total
    if agent.velocity.length() > AGENT_SPEED:
        agent.velocity = agent.velocity.normalize() * AGENT_SPEED
    agent.pos += agent.velocity

    # --- Waypoint progression ---------------------------------------------
    if agent.current_wp_index < len(agent.path):
        if agent.pos.distance_to(pygame.Vector2(agent.path[agent.current_wp_index])) < AGENT_RADIUS * 1.5:
            agent.current_wp_index += 1

    # --- Collision resolution (uses the shared Agent.resolve_collision) ----
    agent.resolve_collision(agents, obstacles)
