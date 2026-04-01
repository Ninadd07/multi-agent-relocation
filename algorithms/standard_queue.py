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
        """Call after a fresh plan_all_paths. All agents start moving freely."""
        if agents:
            for agent in agents:
                agent.waiting = False

    def tick(self, agents):
        """
        Exact copy of reference algorithms.py local_safety_check,
        applied to all agents each frame by the planner.

        - Resets waiting=False each frame
        - Skips if velocity == 0 (reference returns early)
        - Checks ALL other active agents in the forward heading cone
          (dot > 0.7, within VIEW_DISTANCE) — first match wins
        """
        for agent in agents:
            agent.waiting = False
            if not agent.active or agent.spot_reserved:
                continue
            if agent.velocity.length() <= 0:
                continue                          # reference returns early here
            heading = agent.velocity.normalize()
            for other in agents:
                if other is agent or not other.active or other.spot_reserved:
                    continue
                d_vec    = other.pos - agent.pos
                distance = d_vec.length()
                if distance < 0.1:
                    continue
                if distance < VIEW_DISTANCE:
                    d_norm = d_vec.normalize()
                    if heading.dot(d_norm) > 0.7:
                        agent.waiting = True
                        break                     # reference returns on first match


    def enforce_queue(self, agents):
        """
        Hard-constraint enforcer — call AFTER resolve_collision each frame.
        Processes agents front-to-back (index 1 = front).  If any agent has
        been pushed within QUEUE_SPACING of its predecessor by the physics, it
        is repositioned and its velocity zeroed so the gap is strictly maintained.
        """
        sorted_agents = sorted(
            [a for a in agents if a.active and not a.spot_reserved and hasattr(a, 'index')],
            key=lambda a: a.index,
        )
        idx_map = {a.index: a for a in sorted_agents}

        for agent in sorted_agents:
            pred_idx = agent.index - 1
            if pred_idx < 1:
                continue
            pred = idx_map.get(pred_idx)
            if pred is None:
                continue

            distance = agent.pos.distance_to(pred.pos)
            if distance < QUEUE_SPACING:
                # Push agent back to restore spacing and flag it as waiting.
                # We set waiting=True instead of zeroing velocity so that tick()
                # still has a valid heading reference on the next frame.
                if distance > 0.01:
                    push_dir = (agent.pos - pred.pos).normalize()
                    agent.pos = pygame.Vector2(pred.pos + push_dir * QUEUE_SPACING)
                agent.waiting = True

    # ------------------------------------------------------------------
    # Visibility-graph helpers (self-contained, no fundamental.astar_*)
    # ------------------------------------------------------------------
    def _get_corners(self):
        nodes  = []
        off    = 2.0
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
