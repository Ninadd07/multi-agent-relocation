# algorithms/discrete_grid.py
# Adaptive-grid A* path planner (Discretisation)

import math
import pygame
import random
from config import (
    MAP_WIDTH as WORLD_WIDTH,
    MAP_HEIGHT as WORLD_HEIGHT,
    AGENT_RADIUS,
    AGENT_DIAMETER,
)
from fundamental import _astar_core


def _build_path_segments(predecessor_paths):
    """
    We store the line segments of the path instead of points
    so we can evaluate distance to the entire path corridor.
    """
    segments = []
    if not predecessor_paths:
        return segments
    for path in predecessor_paths:
        if not path or len(path) < 2:
            continue
        for i in range(len(path) - 1):
            segments.append((path[i], path[i + 1]))
    return segments

class Discretisation:
    def __init__(self, obstacles=None, base_grid=64.0, min_grid=8.0):
        """
        base_grid: maximum size for a cell (coarse resolution)
        min_grid: minimum size for a cell (fine resolution near obstacles)
        """
        self.base_grid = base_grid
        self.min_grid = min_grid
        self.obstacles = obstacles if obstacles else []
        self.agents = []
        self.cached_grid = None

    def update_obstacles(self, obstacles):
        self.obstacles = obstacles
        self.cached_grid = None

    def set_agents(self, agents):
        self.agents = agents

    # --- Distance from point to segment ---
    @staticmethod
    def point_to_segment_distance(px, py, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy
        return math.hypot(px - nearest_x, py - nearest_y)

    # --- Adaptive occupancy grid ---
    def build_occupancy_grid(self, robot_radius):
        """
        Returns a list of tuples:
        (x0, y0, w, h, is_free)
        """
        grid_cells = []

        def subdivide(x0, y0, w, h):
            near_obs = False

            # We check if the cell, inflated by the robot's radius, collides with any obstacle.
            # This determines if we need to subdivide for higher resolution.
            cell_rect_inflated = pygame.Rect(
                x0 - robot_radius, 
                y0 - robot_radius, 
                w + 2 * robot_radius, 
                h + 2 * robot_radius
            )

            # Check for collision against the true shape of the obstacle, not its AABB.
            # This provides more accurate grid discretisation around non-rectangular obstacles.
            for obs in self.obstacles:
                if obs.colliderect(cell_rect_inflated):
                    near_obs = True
                    break

            should_subdivide = False
            if near_obs:
                if w > self.min_grid:
                    should_subdivide = True
            else:
                if w > self.base_grid:
                    should_subdivide = True

            if not should_subdivide:
                is_free = not near_obs
                grid_cells.append((x0, y0, w, h, is_free))
                return

            hw, hh = w / 2, h / 2
            subdivide(x0, y0, hw, hh)
            subdivide(x0 + hw, y0, hw, hh)
            subdivide(x0, y0 + hh, hw, hh)
            subdivide(x0 + hw, y0 + hh, hw, hh)

        subdivide(0, 0, WORLD_WIDTH, WORLD_HEIGHT)
        return grid_cells

    # --- Helper to find cell index ---
    def get_cell_index(self, x, y, grid_cells):
        best_idx = -1
        min_dist = float('inf')
        
        for i, (cx, cy, w, h, is_free) in enumerate(grid_cells):
            if cx <= x <= cx + w and cy <= y <= cy + h:
                return i
            center_x, center_y = cx + w / 2, cy + h / 2
            d = math.hypot(center_x - x, center_y - y)
            if d < min_dist:
                min_dist = d
                best_idx = i
                
        return best_idx

    # --- Path Smoothing ---
    def smooth_path(self, path, radius):
        if not path or len(path) < 3:
            return path
            
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            next_idx = current_idx + 1
            for i in range(len(path) - 1, current_idx + 1, -1):
                if self.is_segment_safe(smoothed[-1], path[i], radius):
                    next_idx = i
                    break
            smoothed.append(path[next_idx])
            current_idx = next_idx
            
        return smoothed

    def is_segment_safe(self, p1, p2, radius):
        x1, y1 = p1
        x2, y2 = p2
        seg_dist = math.hypot(x2 - x1, y2 - y1)
        if seg_dist < 1e-3:
            return True
            
        step_size = radius * 0.5
        steps = int(math.ceil(seg_dist / step_size))
        safety_margin = 0.2
        check_radius = radius + safety_margin
        for i in range(steps + 1):
            t = i / steps
            px = x1 + (x2 - x1) * t
            py = y1 + (y2 - y1) * t
            
            # Check for collision against the true shape of the obstacle.
            # This ensures the smoothed path correctly avoids the actual obstacle boundaries.
            for obs in self.obstacles:
                inflated_obs = obs.inflate(check_radius * 2, check_radius * 2)
                if inflated_obs.collidepoint(px, py):
                    return False
        return True

    def find_path(self, agent, end, predecessor_paths=None):
        start = (agent.pos.x, agent.pos.y)
        class Robot:
            def __init__(self, x, y, r):
                self.x = x; self.y = y; self.radius = r
        robot = Robot(start[0], start[1], AGENT_RADIUS)
        return self.plan_path(robot, end, agent, predecessor_paths)

    # --- Path planning via fundamental.astar_discrete ---
    def plan_path(self, robot, target, owner_agent=None, predecessor_paths=None):
        if self.cached_grid is None:
            self.cached_grid = self.build_occupancy_grid(robot.radius)
        grid_cells = self.cached_grid
        
        start_idx = self.get_cell_index(robot.x, robot.y, grid_cells)
        end_idx = self.get_cell_index(target[0], target[1], grid_cells)

        if start_idx == -1 or end_idx == -1:
            return None

        end_cell = grid_cells[end_idx]
        end_center = (end_cell[0] + end_cell[2] / 2, end_cell[1] + end_cell[3] / 2)

        # Pre-extract agent positions (exclude self)
        agent_positions = []
        for agent in self.agents:
            if not agent.active or agent is owner_agent:
                continue
            agent_positions.append((agent.pos.x, agent.pos.y))

        # --- A* logic moved from fundamental.py to allow penalty modification ---
        path_segments = _build_path_segments(predecessor_paths)
        inner_r_sq = (robot.radius * 2.5) ** 2
        outer_r_sq = (robot.radius * 6.0) ** 2
        radius_6 = robot.radius * 6.0

        def get_neighbors(current_idx):
            current_cell = grid_cells[current_idx]
            cx = current_cell[0] + current_cell[2] / 2
            cy = current_cell[1] + current_cell[3] / 2
            cw, ch = current_cell[2], current_cell[3]
            neighbors = []

            for i, cell in enumerate(grid_cells):
                if i == current_idx or not cell[4]:  # cell[4] is is_free
                    continue

                nx = cell[0] + cell[2] / 2
                ny = cell[1] + cell[3] / 2
                nw, nh = cell[2], cell[3]

                # Quick bounding-box adjacency reject
                if abs(nx - cx) > (cw + nw) / 2 + 0.1: continue
                if abs(ny - cy) > (ch + nh) / 2 + 0.1: continue

                # Geometric adjacency check
                dx_val, dy_val = abs(nx - cx), abs(ny - cy)
                sum_w, sum_h = (cw + nw) / 2, (ch + nh) / 2
                EPS = 0.1

                touch_x = (abs(dx_val - sum_w) < EPS) and (dy_val < sum_h - EPS)
                touch_y = (abs(dy_val - sum_h) < EPS) and (dx_val < sum_w - EPS)
                touch_diag = (abs(dx_val - sum_w) < EPS) and (abs(dy_val - sum_h) < EPS)

                if not (touch_x or touch_y or touch_diag):
                    continue

                edge_dist = math.hypot(nx - cx, ny - cy)
                penalty = 0

                # --- Dynamic agent penalty ---
                for ax, ay in agent_positions:
                    if abs(nx - ax) > radius_6 or abs(ny - ay) > radius_6:
                        continue
                    d_sq = (nx - ax) ** 2 + (ny - ay) ** 2
                    if d_sq < inner_r_sq: penalty += 60
                    elif d_sq < outer_r_sq: penalty += 10

                # --- Predecessor path penalty ---
                # This is the key change. The penalty for following a previous path
                # was too low, causing agents to queue. We increase it significantly
                # to encourage finding alternative routes.
                for seg in path_segments:
                    if self.point_to_segment_distance(nx, ny, seg[0], seg[1]) < AGENT_DIAMETER * 1.5:
                        penalty += AGENT_DIAMETER * 10.0  # Was AGENT_DIAMETER
                        break

                # --- Symmetry-breaking noise ---
                noise = random.uniform(0, 5.0)
                neighbors.append((i, edge_dist + penalty + noise))

            return neighbors

        def heuristic(idx):
            cell = grid_cells[idx]
            nx = cell[0] + cell[2] / 2
            ny = cell[1] + cell[3] / 2
            return math.hypot(end_center[0] - nx, end_center[1] - ny)

        # Delegate A* to fundamental's core engine
        path_indices = _astar_core(start_idx, end_idx, get_neighbors, heuristic)

        if path_indices is None:
            return None

        # --- Path Generation Improvement ---
        # The original implementation created a path by connecting the centers of the grid cells.
        # This leads to suboptimal, jagged paths, especially in areas with large, open cells,
        # as agents would travel far out of their way to hit the center of a large square.
        #
        # The improved approach generates waypoints at the midpoint of the shared edge between
        # consecutive cells in the A* path. This creates a much more direct "channel"
        # for the path smoother to work with, resulting in shorter and more natural paths.

        # Start with the agent's actual starting position.
        raw_path = [(robot.x, robot.y)]

        # For each pair of consecutive cells, find the midpoint of their shared edge.
        if len(path_indices) > 1:
            for i in range(len(path_indices) - 1):
                cell1 = grid_cells[path_indices[i]]
                cell2 = grid_cells[path_indices[i+1]]
                
                rect1 = pygame.Rect(cell1[0], cell1[1], cell1[2], cell1[3])
                rect2 = pygame.Rect(cell2[0], cell2[1], cell2[2], cell2[3])
                
                # The 'clip' method gives the intersection of two rects. For adjacent
                # grid cells, this intersection is their shared edge (a thin rectangle).
                shared_edge = rect1.clip(rect2)
                
                # The center of this shared edge is a much better waypoint than the cell center.
                raw_path.append(shared_edge.center)

        # Finally, add the actual target position.
        raw_path.append(target)

        # The path smoother can now operate on this much higher-quality raw path.
        return self.smooth_path(raw_path, robot.radius)
