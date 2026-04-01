# algorithms/standard.py
# Standard visibility-graph A* path planner

from config import *
from fundamental import dist, clamp, astar_standard, astar_tbc
import random

class GlobalPlanner:
    def __init__(self, obstacles, mode='standard'):
        """
        mode: 'standard'   - no predecessor penalty
              'penalized'  - proximity-based queueing penalty
              'tbc'        - crossing-based (Traffic-Based Crossing) penalty
        """
        self.mode = mode
        self.update_obstacles(obstacles)

    def update_obstacles(self, obstacles):
        self.obstacles = obstacles
        inflation = AGENT_DIAMETER + 6 
        self.bloated_obstacles = [obs.inflate(inflation, inflation) for obs in obstacles]

    def _corner_clearance(self, cx, cy, dx, dy, own_obs):
        """Diagonal clearance from (cx,cy) in direction (±1,±1) before
        hitting another bloated obstacle or the map boundary.  Returns px.
        """
        x_lim = cx          if dx < 0 else MAP_WIDTH  - cx
        y_lim = cy          if dy < 0 else MAP_HEIGHT - cy
        limit = min(x_lim, y_lim)
        probe = 4.0
        t     = probe
        while t < limit:
            tx = cx + dx * t
            ty = cy + dy * t
            for obs in self.bloated_obstacles:
                if obs is own_obs:
                    continue
                if obs.collidepoint(tx, ty):
                    return t
            t += probe
        return limit

    def get_corners(self, corner_offsets=None):
        """Generate visibility-graph corner nodes.

        corner_offsets=None  → intelligent mode: fan count derived per-corner
                               from diagonal clearance (penalized mode).
        corner_offsets=list  → fixed offsets applied uniformly (standard mode).
        """
        step    = float(AGENT_DIAMETER + 2)   # 18 px — above proximity_r
        max_fan = 6
        margin  = 10.0
        nodes   = []

        for obs in self.bloated_obstacles:
            left_ok   = obs.left   >= 0
            right_ok  = obs.right  <= MAP_WIDTH
            top_ok    = obs.top    >= 0
            bottom_ok = obs.bottom <= MAP_HEIGHT

            # Enumerate corners with their outward diagonal direction
            raw_corners = []
            if left_ok  and top_ok:    raw_corners.append((obs.left,  obs.top,    -1, -1))
            if right_ok and top_ok:    raw_corners.append((obs.right, obs.top,    +1, -1))
            if right_ok and bottom_ok: raw_corners.append((obs.right, obs.bottom, +1, +1))
            if left_ok  and bottom_ok: raw_corners.append((obs.left,  obs.bottom, -1, +1))

            for (cx, cy, dx, dy) in raw_corners:
                if corner_offsets is None:
                    # Intelligent: measure how much diagonal space exists
                    clearance = self._corner_clearance(cx, cy, dx, dy, obs)
                    n_off     = max(1, min(max_fan, int(clearance / step)))
                    offsets   = [float(2 + i * step) for i in range(n_off)]
                else:
                    offsets = corner_offsets

                for m in offsets:
                    x = clamp(cx + dx * m, margin, MAP_WIDTH  - margin)
                    y = clamp(cy + dy * m, margin, MAP_HEIGHT - margin)
                    if all(not o.collidepoint(x, y) for o in self.bloated_obstacles):
                        nodes.append((x, y))

        return nodes

    def is_line_clear(self, start, end, agent_start_pos):
        p1 = (float(start[0]), float(start[1]))
        p2 = (float(end[0]), float(end[1]))
        
        # Check if this edge originates from the agent's current position
        is_start_edge = (start == agent_start_pos or end == agent_start_pos)
        
        for obs in self.bloated_obstacles:
            # If the line intersects the obstacle boundary
            if obs.clipline(p1, p2): 
                if not is_start_edge:
                    return False
                # If it is the start edge, allow it to cross the wall ONLY IF 
                # the agent is currently stuck inside that exact wall.
                if is_start_edge and not obs.collidepoint(agent_start_pos):
                    return False 
                    
            # Check if endpoints are stuck inside a wall (ignore the agent's start pos)
            if start != agent_start_pos and obs.collidepoint(p1): return False
            if end != agent_start_pos and obs.collidepoint(p2): return False
            
        return True

    def find_path(self, agent, end_pos, predecessor_paths=None):
        start_pos = agent.pos
        
        # Removed the strict end_pos check just in case the goal is slightly inside a margin
        
        # Penalized mode: fan count derived intelligently from per-corner clearance.
        # Standard mode: fixed two nodes per corner (tight + wide offset).
        if self.mode == 'penalized' and predecessor_paths:
            corner_nodes = self.get_corners()                              # intelligent
        else:
            corner_nodes = self.get_corners([2.0, float(WIDE_CORNER_MAX_MARGIN)])

        nodes = [start_pos, end_pos] + corner_nodes
        valid_nodes = []
        for n in nodes:
            safe = True
            for obs in self.bloated_obstacles:
                if obs.collidepoint(n): safe = False; break
            if safe or n == start_pos or n == end_pos: valid_nodes.append(n)
        
        nodes = valid_nodes
        graph = {i: [] for i in range(len(nodes))}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Pass the start_pos into the line-clear check
                if self.is_line_clear(nodes[i], nodes[j], start_pos):
                    d = dist(nodes[i], nodes[j])
                    graph[i].append((j, d))
                    graph[j].append((i, d))

        if self.mode == 'tbc':
            return astar_tbc(0, 1, graph, nodes, end_pos, predecessor_paths)
        elif self.mode == 'penalized':
            return astar_standard(0, 1, graph, nodes, end_pos, predecessor_paths)
        else:
            return astar_standard(0, 1, graph, nodes, end_pos, None)