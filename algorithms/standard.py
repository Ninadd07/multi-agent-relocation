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

    def get_corners(self):
        nodes = []
        margin = 10.0
        
        for obs in self.bloated_obstacles:
            # Check which edges of this obstacle extend off-screen.
            # If an edge is off-screen, corners on that side are phantom —
            # they don't represent a real navigable path around an obstacle edge.
            left_ok  = obs.left  >= 0
            right_ok = obs.right <= MAP_WIDTH
            top_ok   = obs.top   >= 0
            bottom_ok = obs.bottom <= MAP_HEIGHT
            
            m_tl = random.uniform(2.0, WIDE_CORNER_MAX_MARGIN)
            m_tr = random.uniform(2.0, WIDE_CORNER_MAX_MARGIN)
            m_br = random.uniform(2.0, WIDE_CORNER_MAX_MARGIN)
            m_bl = random.uniform(2.0, WIDE_CORNER_MAX_MARGIN)
            
            # Each corner is only valid if BOTH of its edges are on-screen.
            # E.g. top-left corner is only useful if both left AND top edges are on-screen.
            candidate_corners = []
            if left_ok and top_ok:
                candidate_corners.append((obs.left - 2.0, obs.top - 2.0))
                candidate_corners.append((obs.left - m_tl, obs.top - m_tl))
            if right_ok and top_ok:
                candidate_corners.append((obs.right + 2.0, obs.top - 2.0))
                candidate_corners.append((obs.right + m_tr, obs.top - m_tr))
            if right_ok and bottom_ok:
                candidate_corners.append((obs.right + 2.0, obs.bottom + 2.0))
                candidate_corners.append((obs.right + m_br, obs.bottom + m_br))
            if left_ok and bottom_ok:
                candidate_corners.append((obs.left - 2.0, obs.bottom + 2.0))
                candidate_corners.append((obs.left - m_bl, obs.bottom + m_bl))
            
            for (x, y) in candidate_corners:
                cx = clamp(x, margin, MAP_WIDTH - margin)
                cy = clamp(y, margin, MAP_HEIGHT - margin)
                is_safe = True
                for test_obs in self.bloated_obstacles:
                    if test_obs.collidepoint(cx, cy):
                        is_safe = False; break
                if is_safe: nodes.append((cx, cy))
        return nodes

    def is_line_clear(self, start, end):
        p1 = (float(start[0]), float(start[1]))
        p2 = (float(end[0]), float(end[1]))
        for obs in self.bloated_obstacles:
            if obs.clipline(p1, p2): return False
            if obs.collidepoint(p1) or obs.collidepoint(p2): return False
        return True

    def find_path(self, agent, end_pos, predecessor_paths=None):
        start_pos = agent.pos
        for obs in self.bloated_obstacles:
            if obs.collidepoint(start_pos) or obs.collidepoint(end_pos): return None
        
        nodes = [start_pos, end_pos] + self.get_corners()
        valid_nodes = []
        for n in nodes:
            safe = True
            for obs in self.bloated_obstacles:
                if obs.collidepoint(n): safe = False; break
            if safe or n == start_pos or n == end_pos: valid_nodes.append(n)
        
        nodes = valid_nodes
        # Build visibility graph
        graph = {i: [] for i in range(len(nodes))}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self.is_line_clear(nodes[i], nodes[j]):
                    d = dist(nodes[i], nodes[j])
                    graph[i].append((j, d))
                    graph[j].append((i, d))

        if self.mode == 'tbc':
            return astar_tbc(0, 1, graph, nodes, end_pos, predecessor_paths)
        elif self.mode == 'penalized':
            return astar_standard(0, 1, graph, nodes, end_pos, predecessor_paths)
        else:
            # 'standard' — no penalties
            return astar_standard(0, 1, graph, nodes, end_pos, None)
