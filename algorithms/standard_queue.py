# algorithms/standard_queue.py
# Explicit-queue planner.
# - Every agent receives the identical shortest path (no predecessor penalties).
# - Agent N is held (waiting=True) until agent N-1 has travelled
#   QUEUE_SPACING pixels from its own release position.
# - tick() is called every simulation frame by the GUI to update release flags.
# Completely independent of GlobalPlanner / astar_standard / penalty system.

import math
import heapq
import pygame
from config import *
from fundamental import clamp

# How far the predecessor must travel before the next agent is released.
QUEUE_SPACING = AGENT_DIAMETER
QUEUE_RELEASE_RADIUS = AGENT_DIAMETER * 4


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class QueuePlanner:
    """
    Visibility-graph A* with explicit predecessor-release queueing.

    find_path()  — pure geometric A*, same path for every agent.
    tick(agents) — called every frame; sets agent.waiting based on
                   whether the predecessor has moved far enough.
    reset_queue()— call after a fresh plan_all_paths.
    """

    def __init__(self, obstacles):
        self._released    : set   = set()   # indices already released
        self._release_pos : dict  = {}      # idx → Vector2 at release moment
        self.update_obstacles(obstacles)

    # ------------------------------------------------------------------
    # Obstacle management
    # ------------------------------------------------------------------
    def update_obstacles(self, obstacles):
        self.obstacles = obstacles
        inflation  = AGENT_DIAMETER + 6
        self.bloated = [obs.inflate(inflation, inflation) for obs in obstacles]

    # ------------------------------------------------------------------
    # Queue state
    # ------------------------------------------------------------------
    def reset_queue(self, agents=None):
        self._released = set()
        self._release_pos = {}

        if not agents:
            return

        sorted_agents = sorted(
            [a for a in agents if a.active and not a.spot_reserved and hasattr(a, "index")],
            key=lambda a: a.index,
        )

        for agent in sorted_agents:
            if agent.index == 1:
                agent.waiting = False
                self._released.add(agent.index)
                self._release_pos[agent.index] = pygame.Vector2(agent.pos)
            else:
                agent.waiting = True

    def tick(self, agents):
        sorted_agents = sorted(
            [a for a in agents if a.active and not a.spot_reserved and hasattr(a, "index")],
            key=lambda a: a.index,
        )

        if not sorted_agents:
            return

        idx_map = {a.index: a for a in sorted_agents}

        first = sorted_agents[0]
        if first.index not in self._released:
            self._released.add(first.index)
            self._release_pos[first.index] = pygame.Vector2(first.pos)
            first.waiting = False

        for agent in sorted_agents:
            # Relax queueing once agent is near / inside destination region
            if hasattr(agent, "exit_manager") and agent.exit_manager is not None:
                expanded_end = agent.exit_manager.rect.inflate(40, 40)
                if expanded_end.collidepoint(agent.pos.x, agent.pos.y):
                    agent.waiting = False
                    continue

            if agent.index == 1:
                agent.waiting = False
                continue

            pred_idx = agent.index - 1
            pred = idx_map.get(pred_idx)

            # IMPORTANT FIX:
            # If predecessor is missing from idx_map, it most likely already entered
            # the exit zone and became spot_reserved. In that case, do NOT block.
            if pred is None:
                if agent.index not in self._released:
                    self._released.add(agent.index)
                    self._release_pos[agent.index] = pygame.Vector2(agent.pos)
                agent.waiting = False
                continue

            # If predecessor is near destination, do not keep strict queueing
            if hasattr(agent, "exit_manager") and agent.exit_manager is not None:
                expanded_end = agent.exit_manager.rect.inflate(40, 40)
                if expanded_end.collidepoint(pred.pos.x, pred.pos.y):
                    agent.waiting = False
                    continue

            if agent.index not in self._released:
                pred_release_pos = self._release_pos.get(pred.index)
                if pred_release_pos is None:
                    agent.waiting = True
                    continue

                dist_from_pred_release = pred.pos.distance_to(pred_release_pos)

                if dist_from_pred_release >= QUEUE_SPACING:
                    self._released.add(agent.index)
                    self._release_pos[agent.index] = pygame.Vector2(agent.pos)
                    agent.waiting = False
                else:
                    agent.waiting = True
                    continue

            distance_to_pred = agent.pos.distance_to(pred.pos)
            agent.waiting = distance_to_pred < QUEUE_SPACING
    def enforce_queue(self, agents):
        sorted_agents = sorted(
            [a for a in agents if a.active and not a.spot_reserved and hasattr(a, "index")],
            key=lambda a: a.index,
        )
        idx_map = {a.index: a for a in sorted_agents}

        for agent in sorted_agents:
            if agent.index == 1:
                continue

            # Do not enforce queue spacing near the destination
            if hasattr(agent, "exit_manager") and agent.exit_manager is not None:
                goal = pygame.Vector2(agent.exit_manager.center_pixel)
                if agent.pos.distance_to(goal) < QUEUE_RELEASE_RADIUS:
                    continue

            pred = idx_map.get(agent.index - 1)
            if pred is None:
                continue

            # Also stop queue enforcement if predecessor is already near the destination
            if hasattr(agent, "exit_manager") and agent.exit_manager is not None:
                goal = pygame.Vector2(agent.exit_manager.center_pixel)
                if pred.pos.distance_to(goal) < QUEUE_RELEASE_RADIUS:
                    continue

            distance = agent.pos.distance_to(pred.pos)

            if distance < QUEUE_SPACING:
                if distance > 0.01:
                    push_dir = (agent.pos - pred.pos).normalize()
                else:
                    push_dir = pygame.Vector2(-1, 0)

                agent.pos = pygame.Vector2(pred.pos + push_dir * QUEUE_SPACING)
                agent.velocity = pygame.Vector2(0, 0)
                agent.waiting = True
    # ------------------------------------------------------------------
    # Visibility-graph helpers (self-contained, no fundamental.astar_*)
    # ------------------------------------------------------------------
    def _get_corners(self):
        nodes  = []
        off    = 20.0
        margin = 10.0

        for obs in self.bloated:
            left_ok   = obs.left   >= 0
            right_ok  = obs.right  <= MAP_WIDTH
            top_ok    = obs.top    >= 0
            bottom_ok = obs.bottom <= MAP_HEIGHT

            cands = []
            if left_ok  and top_ok:    cands.append((obs.left  - off, obs.top    - off))
            if right_ok and top_ok:    cands.append((obs.right + off, obs.top    - off))
            if right_ok and bottom_ok: cands.append((obs.right + off, obs.bottom + off))
            if left_ok  and bottom_ok: cands.append((obs.left  - off, obs.bottom + off))

            for (x, y) in cands:
                cx = clamp(x, margin, MAP_WIDTH  - margin)
                cy = clamp(y, margin, MAP_HEIGHT - margin)
                if all(not o.collidepoint(cx, cy) for o in self.bloated):
                    nodes.append((cx, cy))
        return nodes

    def _line_clear(self, p1, p2, agent_start):
        fp1 = (float(p1[0]), float(p1[1]))
        fp2 = (float(p2[0]), float(p2[1]))
        is_start_edge = (p1 == agent_start or p2 == agent_start)
        for obs in self.bloated:
            if obs.clipline(fp1, fp2):
                if not is_start_edge:
                    return False
                if not obs.collidepoint(*agent_start):
                    return False
            if p1 != agent_start and obs.collidepoint(fp1): return False
            if p2 != agent_start and obs.collidepoint(fp2): return False
        return True

    # ------------------------------------------------------------------
    # Path planning
    # ------------------------------------------------------------------
    def find_path(self, agent, end_pos, predecessor_paths=None):
        """
        Pure shortest-path A*.  predecessor_paths is **intentionally ignored**.
        Queue ordering is enforced by tick(), not by path cost.
        """
        sp = (float(agent.pos.x), float(agent.pos.y))
        ep = (
            (float(end_pos.x), float(end_pos.y))
            if hasattr(end_pos, 'x')
            else (float(end_pos[0]), float(end_pos[1]))
        )

        all_nodes = [sp, ep] + self._get_corners()
        nodes = [
            n for n in all_nodes
            if all(not o.collidepoint(*n) for o in self.bloated) or n in (sp, ep)
        ]

        graph = {i: [] for i in range(len(nodes))}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self._line_clear(nodes[i], nodes[j], sp):
                    d = _dist(nodes[i], nodes[j])
                    graph[i].append((j, d))
                    graph[j].append((i, d))

        open_set  = [(0.0, 0, 0)]
        came_from = {0: None}
        g_score   = {0: 0.0}
        counter   = 1

        while open_set:
            _, _, curr = heapq.heappop(open_set)
            if curr == 1:
                path, c = [], 1
                while c is not None:
                    path.append(nodes[c])
                    c = came_from[c]
                path.reverse()
                return path
            for nxt, w in graph[curr]:
                ng = g_score[curr] + w
                if nxt not in g_score or ng < g_score[nxt]:
                    g_score[nxt] = ng
                    heapq.heappush(open_set, (ng + _dist(nodes[nxt], ep), counter, nxt))
                    counter += 1
                    came_from[nxt] = curr

        return None
