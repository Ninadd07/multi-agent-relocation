import math
import heapq
import pygame
import random
from config import MAP_WIDTH as WORLD_WIDTH, MAP_HEIGHT as WORLD_HEIGHT, AGENT_RADIUS


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
            mid_x, mid_y = x0 + w / 2, y0 + h / 2
            near_obs = False

            # 1. Broad phase: Check if any obstacle is within the cell's bounding box (expanded by radius)
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

            # 2. Decision to subdivide or stop
            should_subdivide = False
            
            if near_obs:
                # If near obstacle, subdivide unless we reached min resolution
                if w > self.min_grid:
                    should_subdivide = True
            else:
                # If free, subdivide unless we are small enough (base_grid)
                if w > self.base_grid:
                    should_subdivide = True

            if not should_subdivide:
                # Leaf node
                # If near_obs is True here, it means we hit min_grid, so it's occupied.
                is_free = not near_obs
                grid_cells.append((x0, y0, w, h, is_free))
                return

            # Subdivide into 4
            hw, hh = w / 2, h / 2
            subdivide(x0, y0, hw, hh)
            subdivide(x0 + hw, y0, hw, hh)
            subdivide(x0, y0 + hh, hw, hh)
            subdivide(x0 + hw, y0 + hh, hw, hh)

        # Start with the full world dimensions
        subdivide(0, 0, WORLD_WIDTH, WORLD_HEIGHT)
        return grid_cells

    # --- Helper to find cell index ---
    def get_cell_index(self, x, y, grid_cells):
        best_idx = -1
        min_dist = float('inf')
        
        for i, (cx, cy, w, h, is_free) in enumerate(grid_cells):
            # We check all cells, even occupied ones, so robots can pathfind OUT of obstacles
            # Check if point is strictly inside
            if cx <= x <= cx + w and cy <= y <= cy + h:
                return i
            
            # Fallback: find nearest center
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
            # Try to connect current point to the furthest possible point
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
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1e-3:
            return True
            
        # Check points along the segment
        step_size = radius * 0.5
        steps = int(math.ceil(dist / step_size))
        
        # Add a safety margin to prevent grazing corners which causes the robot to get stuck
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

    def find_path(self, agent, end):
        start = (agent.pos.x, agent.pos.y)
        # Adapter for gui.py Agent
        class Robot:
            def __init__(self, x, y, r):
                self.x = x; self.y = y; self.radius = r
        robot = Robot(start[0], start[1], AGENT_RADIUS)
        return self.plan_path(robot, end, agent)

    # --- A* path planning over adaptive grid ---
    def plan_path(self, robot, target, owner_agent=None):
        if self.cached_grid is None:
            self.cached_grid = self.build_occupancy_grid(robot.radius)
        grid_cells = self.cached_grid
        
        start_idx = self.get_cell_index(robot.x, robot.y, grid_cells)
        end_idx = self.get_cell_index(target[0], target[1], grid_cells)

        if start_idx == -1 or end_idx == -1:
            return None

        # A* Initialization
        # Node state is the index in grid_cells
        start_cell = grid_cells[start_idx]
        start_center = (start_cell[0] + start_cell[2]/2, start_cell[1] + start_cell[3]/2)
        
        end_cell = grid_cells[end_idx]
        end_center = (end_cell[0] + end_cell[2]/2, end_cell[1] + end_cell[3]/2)

        # OPTIMIZATION: Pre-extract agent positions to avoid object attribute lookups in inner loop
        # and filter out self.
        agent_positions = []
        for agent in self.agents:
            if not agent.active: continue
            # Robust exclusion: Don't treat myself as an obstacle
            if agent is owner_agent: continue
            agent_positions.append((agent.pos.x, agent.pos.y))

        # Priority Queue: (f_score, cell_index, path_points)
        open_set = []
        heapq.heappush(open_set, (0, start_idx, [(robot.x, robot.y)]))
        
        g_score = {start_idx: 0}
        visited = set()

        while open_set:
            _, current_idx, path = heapq.heappop(open_set)
            
            if current_idx in visited:
                continue
            visited.add(current_idx)

            if current_idx == end_idx:
                # Path found
                raw_path = path + [target]
                return self.smooth_path(raw_path, robot.radius)

            current_cell = grid_cells[current_idx]
            cx, cy = current_cell[0] + current_cell[2]/2, current_cell[1] + current_cell[3]/2
            cw, ch = current_cell[2], current_cell[3]

            # Find neighbors
            for i, cell in enumerate(grid_cells):
                if i == current_idx or not cell[4]: # Skip self or occupied
                    continue
                if i in visited:
                    continue

                nx, ny = cell[0] + cell[2]/2, cell[1] + cell[3]/2
                nw, nh = cell[2], cell[3]

                # Optimization: Bounding box check for adjacency
                # If centers are too far, they can't be neighbors
                if abs(nx - cx) > (cw + nw) / 2 + 0.1: continue
                if abs(ny - cy) > (ch + nh) / 2 + 0.1: continue

                # Geometric Adjacency Check (Touching edges)
                dx = abs(nx - cx)
                dy = abs(ny - cy)
                sum_w = (cw + nw) / 2
                sum_h = (ch + nh) / 2
                EPS = 0.1

                # Touching in X (vertical edge shared)
                touch_x = (abs(dx - sum_w) < EPS) and (dy < sum_h - EPS)
                # Touching in Y (horizontal edge shared)
                touch_y = (abs(dy - sum_h) < EPS) and (dx < sum_w - EPS)
                # Touching Corner (Diagonal)
                touch_diag = (abs(dx - sum_w) < EPS) and (abs(dy - sum_h) < EPS)

                if touch_x or touch_y or touch_diag:
                    dist = math.hypot(nx - cx, ny - cy)
                    
                    # Dynamic Obstacle Penalty: Discourage paths near other agents
                    penalty = 0
                    
                    # Two-Tier Penalty System to force flow around bottlenecks
                    # 1. Inner Zone: High cost to prevent physical overlap
                    # 2. Outer Zone: Moderate cost to discourage congestion (Congestion Repulsion)
                    inner_r_sq = (robot.radius * 2.5) ** 2
                    outer_r_sq = (robot.radius * 6.0) ** 2
                    
                    for ax, ay in agent_positions:
                        # Fast Bounding Box rejection (based on outer radius)
                        if abs(nx - ax) > robot.radius * 6.0 or abs(ny - ay) > robot.radius * 6.0:
                            continue
                        
                        d_sq = (nx - ax)**2 + (ny - ay)**2
                        if d_sq < inner_r_sq:
                            penalty += 60 # Reduced from 500: High cost, but passable if necessary (prevents retreating)
                        elif d_sq < outer_r_sq:
                            penalty += 10  # Reduced from 50: Slight preference for open space

                    # Add Random Noise: Makes pathfinding unique per robot/frame to break symmetry and clumping
                    noise = random.uniform(0, 5.0)

                    new_g = g_score[current_idx] + dist + penalty + noise
                    
                    if i not in g_score or new_g < g_score[i]:
                        g_score[i] = new_g
                        h = math.hypot(end_center[0] - nx, end_center[1] - ny)
                        heapq.heappush(open_set, (new_g + h, i, path + [(nx, ny)]))

        # Fallback
        return None
