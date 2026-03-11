# algorithms/discrete_grid.py
# Adaptive-grid A* path planner (Discretisation)

import math
import pygame
from config import MAP_WIDTH as WORLD_WIDTH, MAP_HEIGHT as WORLD_HEIGHT, AGENT_RADIUS
from fundamental import astar_discrete


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

            cell_aabb = (
                x0 - robot_radius, 
                y0 - robot_radius, 
                w + 2 * robot_radius, 
                h + 2 * robot_radius
            )

            for obs in self.obstacles:
                if isinstance(obs, pygame.Rect):
                    if (obs.x < cell_aabb[0] + cell_aabb[2] and obs.right > cell_aabb[0] and
                        obs.y < cell_aabb[1] + cell_aabb[3] and obs.bottom > cell_aabb[1]):
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
            
            for obs in self.obstacles:
                if isinstance(obs, pygame.Rect):
                    if (obs.x - check_radius < px < obs.right + check_radius and
                        obs.y - check_radius < py < obs.bottom + check_radius):
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
            if not agent.active:
                continue
            if agent is owner_agent:
                continue
            agent_positions.append((agent.pos.x, agent.pos.y))

        # Delegate A* to fundamental
        path_indices = astar_discrete(
            start_idx, end_idx, grid_cells, end_center,
            agent_positions, robot.radius, predecessor_paths
        )

        if path_indices is None:
            return None

        # Convert cell indices → coordinates
        raw_path = [(robot.x, robot.y)]
        for idx in path_indices[1:]:
            cell = grid_cells[idx]
            raw_path.append((cell[0] + cell[2] / 2, cell[1] + cell[3] / 2))
        raw_path.append(target)

        return self.smooth_path(raw_path, robot.radius)
