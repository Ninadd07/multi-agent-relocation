# algorithms.py
import pygame
import math
import heapq
import random
from collections import deque
from config import *

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

# --- SMART EXIT (Controller) ---
class SmartExit:
    def __init__(self, rect):
        self.rect = rect
        self.cols = int(rect.width // (AGENT_DIAMETER + 4))
        self.rows = int(rect.height // (AGENT_DIAMETER + 4))
        self.cell_w = rect.width / self.cols
        self.cell_h = rect.height / self.rows
        
        self.parking_queue = [] 
        self.initialized = False
        
        # We no longer store a static grid; we check dynamic positions
        self.active_agents = []

    def register_agents(self, agents):
        """Keep a reference to all agents for proximity checks"""
        self.active_agents = agents

    def is_cell_physically_blocked(self, r, c, self_agent):
        """
        Returns True if ANY agent is physically too close to this cell's center.
        This prevents 'running over' even during transitions.
        """
        cell_center = self.get_pixel_center(r, c)
        safe_radius = AGENT_DIAMETER * 0.9 # Slightly generous buffer
        
        for agent in self.active_agents:
            if agent is self_agent: continue
            if not agent.spot_reserved: continue # Ignore agents outside box
            
            # If an agent is physically overlapping this cell, it's blocked
            if agent.pos.distance_to(cell_center) < safe_radius:
                return True
        return False

    def get_grid_coords(self, pos):
        local_x = pos[0] - self.rect.x
        local_y = pos[1] - self.rect.y
        c = int(clamp(local_x // self.cell_w, 0, self.cols - 1))
        r = int(clamp(local_y // self.cell_h, 0, self.rows - 1))
        return r, c

    def get_pixel_center(self, r, c):
        px = self.rect.x + (c * self.cell_w) + (self.cell_w / 2)
        py = self.rect.y + (r * self.cell_h) + (self.cell_h / 2)
        return pygame.Vector2(px, py)

    def initialize_queue(self, entry_velocity):
        self.parking_queue = []
        all_slots = []
        for r in range(self.rows):
            for c in range(self.cols):
                all_slots.append((r, c))

        # Sort: Deepest First
        if abs(entry_velocity.x) > abs(entry_velocity.y):
            if entry_velocity.x > 0: all_slots.sort(key=lambda x: (-x[1], x[0])) 
            else: all_slots.sort(key=lambda x: (x[1], x[0]))
        else:
            if entry_velocity.y > 0: all_slots.sort(key=lambda x: (-x[0], x[1]))
            else: all_slots.sort(key=lambda x: (x[0], x[1]))

        self.parking_queue = all_slots
        self.initialized = True

    def request_spot(self, agent_vel):
        if not self.initialized: self.initialize_queue(agent_vel)
        if self.parking_queue: return self.parking_queue.pop(0)
        return None

# --- GLOBAL PLANNER ---
class GlobalPlanner:
    def __init__(self, obstacles):
        self.update_obstacles(obstacles)

    def update_obstacles(self, obstacles):
        self.obstacles = obstacles
        inflation = AGENT_DIAMETER + 6 
        self.bloated_obstacles = [obs.inflate(inflation, inflation) for obs in obstacles]

    def get_corners(self):
        nodes = []
        margin = 2.0
        for obs in self.bloated_obstacles:
            corners = [
                (obs.left - margin, obs.top - margin),
                (obs.right + margin, obs.top - margin),
                (obs.right + margin, obs.bottom + margin),
                (obs.left - margin, obs.bottom + margin)
            ]
            for (x, y) in corners:
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

    def find_path(self, agent, end_pos):
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
        graph = {i: [] for i in range(len(nodes))}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if self.is_line_clear(nodes[i], nodes[j]):
                    d = dist(nodes[i], nodes[j])
                    graph[i].append((j, d))
                    graph[j].append((i, d))

        pq = [(0, 0)]; came_from = {0: None}; cost_so_far = {0: 0}; target_idx = 1 
        while pq:
            current_cost, current_idx = heapq.heappop(pq)
            if current_idx == target_idx: break
            for next_idx, weight in graph[current_idx]:
                new_cost = cost_so_far[current_idx] + weight
                if next_idx not in cost_so_far or new_cost < cost_so_far[next_idx]:
                    cost_so_far[next_idx] = new_cost
                    priority = new_cost + dist(nodes[next_idx], end_pos)
                    heapq.heappush(pq, (priority, next_idx))
                    came_from[next_idx] = current_idx

        if target_idx not in came_from: return None
        path = []; curr = target_idx
        while curr is not None:
            path.append(nodes[curr]); curr = came_from[curr]
        path.reverse()
        return path

# --- AGENT ---
class Agent:
    def __init__(self, start, exit_manager, planner):
        self.pos = pygame.Vector2(start)
        self.exit_manager = exit_manager
        self.planner = planner
        self.target_pos = pygame.Vector2(exit_manager.rect.center)
        
        self.active = True
        self.waiting = False
        self.spot_reserved = False 
        self.target_grid_coord = None
        
        self.path_valid = False
        self.path = []
        self.current_wp_index = 0
        self.velocity = pygame.Vector2(0, 0)
        self.color = RED
        self.patience = 0
        self.patience_threshold = random.randint(30, 60)
        self.push_state = 0 # Frames to ignore safety checks (break deadlocks)
        self.prev_pos = pygame.Vector2(start)
        self.recalc_path()

    def recalc_path(self):
        if not self.active: return
        new_path = self.planner.find_path(self, self.target_pos)
        if new_path:
            self.path = new_path
            self.current_wp_index = 0
            self.path_valid = True
        else:
            self.path_valid = False 

    def local_safety_check(self, agents):
        self.waiting = False
        if not self.active: return
        
        # If in push mode, ignore safety checks to force movement
        if self.push_state > 0:
            self.push_state -= 1
            return

        # Determine heading: use velocity if moving, otherwise use path target
        heading = None
        if self.velocity.length() > 0.1:
            heading = self.velocity.normalize()
        elif self.path and self.current_wp_index < len(self.path):
            tgt = pygame.Vector2(self.path[self.current_wp_index])
            diff = tgt - self.pos
            if diff.length() > 0.1: heading = diff.normalize()
            
        if heading is None: return

        for other in agents:
            if other is self or not other.active: continue
            d_vec = other.pos - self.pos
            distance = d_vec.length()
            if distance < 0.1: continue 
            if distance < VIEW_DISTANCE * 0.75: # Reduced view distance to prevent freezing in crowds
                d_norm = d_vec.normalize()
                angle = heading.dot(d_norm)
                if angle > 0.9: 
                    # Smart Following: Don't wait if the agent ahead is moving fast enough in the same direction
                    if other.velocity.length() > 0.5:
                        # Check if other is moving in roughly the same direction as my intended heading
                        move_alignment = heading.dot(other.velocity.normalize())
                        if move_alignment > 0.7:
                             # If they are moving away/along path, don't wait unless we are very close
                             if distance > AGENT_DIAMETER * 1.1:
                                 continue
                    self.waiting = True; return 

    def resolve_collision(self, agents, obstacles):
        for obs in obstacles:
            closest_x = clamp(self.pos.x, obs.left, obs.right)
            closest_y = clamp(self.pos.y, obs.top, obs.bottom)
            diff_x = self.pos.x - closest_x
            diff_y = self.pos.y - closest_y
            dist = math.hypot(diff_x, diff_y)
            if dist < AGENT_RADIUS:
                if dist == 0: self.pos.x += 1
                else:
                    overlap = AGENT_RADIUS - dist
                    self.pos.x += (diff_x / dist) * overlap
                    self.pos.y += (diff_y / dist) * overlap
        
        if self.pos.x < AGENT_RADIUS: self.pos.x = AGENT_RADIUS
        if self.pos.x > MAP_WIDTH - AGENT_RADIUS: self.pos.x = MAP_WIDTH - AGENT_RADIUS
        if self.pos.y < AGENT_RADIUS: self.pos.y = AGENT_RADIUS
        if self.pos.y > MAP_HEIGHT - AGENT_RADIUS: self.pos.y = MAP_HEIGHT - AGENT_RADIUS

        # Agent-Agent (ALWAYS ON)
        for other in agents:
            if other is self: continue
            
            diff = self.pos - other.pos
            dist = diff.length()
            min_dist = AGENT_DIAMETER + 1 
            if dist < min_dist:
                if dist == 0: correction = pygame.Vector2(1, 0)
                else:
                    overlap = min_dist - dist
                    correction = diff.normalize() * (overlap / 2)
                self.pos += correction

    def find_next_grid_step(self, curr_r, curr_c, target_r, target_c):
        """
        BFS to find next move, treating occupied cells as walls.
        """
        if curr_r == target_r and curr_c == target_c: return (curr_r, curr_c)

        queue = deque([(curr_r, curr_c, [])]) 
        visited = set(); visited.add((curr_r, curr_c))
        
        MAX_DEPTH = 10 

        while queue:
            r, c, path = queue.popleft()
            if len(path) > MAX_DEPTH: continue
            if r == target_r and c == target_c: return path[0] if path else (r, c)

            neighbors = [
                (r+1,c), (r-1,c), (r,c+1), (r,c-1),
                (r+1,c+1), (r+1,c-1), (r-1,c+1), (r-1,c-1)
            ]
            neighbors.sort(key=lambda n: abs(n[0]-target_r) + abs(n[1]-target_c))

            for nr, nc in neighbors:
                if (nr, nc) not in visited:
                    # STRICT: Is this cell physically clear?
                    if not self.exit_manager.is_cell_physically_blocked(nr, nc, self):
                        visited.add((nr, nc))
                        new_path = path + [(nr, nc)]
                        if nr == target_r and nc == target_c: return new_path[0]
                        queue.append((nr, nc, new_path))
        return None 

    def update_grid_step(self):
        if not self.target_grid_coord: return

        curr_r, curr_c = self.exit_manager.get_grid_coords(self.pos)
        target_r, target_c = self.target_grid_coord
        
        # 1. ARRIVAL CHECK
        if curr_r == target_r and curr_c == target_c:
            final_center = self.exit_manager.get_pixel_center(target_r, target_c)
            dist = self.pos.distance_to(final_center)
            if dist < 3:
                self.pos = final_center; self.active = False
            else:
                self.velocity = (final_center - self.pos).normalize() * AGENT_SPEED
                self.pos += self.velocity
            return

        # 2. PATHFINDING
        next_step = self.find_next_grid_step(curr_r, curr_c, target_r, target_c)

        if next_step:
            self.waiting = False
            next_pixel = self.exit_manager.get_pixel_center(*next_step)
            direction = next_pixel - self.pos
            if direction.length() > 0:
                self.velocity = direction.normalize() * AGENT_SPEED
                self.pos += self.velocity
        else:
            # STOP AND WAIT FOR DECONFLICTION
            self.waiting = True
            self.velocity = pygame.Vector2(0,0)

    def update(self, end_rect):
        if not self.active: return
        
        # RECOVERY: If path is invalid (stuck), try to find one again periodically
        if not self.path_valid:
            self.patience += 1
            if self.patience > self.patience_threshold:
                self.recalc_path()
                self.patience = 0
                self.patience_threshold = random.randint(30, 60)
            return

        self._check_parking_logic(end_rect)
        
        if self.spot_reserved:
            self.update_grid_step()
        else:
            # Calculate actual physical movement to detect if stuck
            moved_dist = (self.pos - self.prev_pos).length()
            self.prev_pos = pygame.Vector2(self.pos)

            # Check patience: Only if NOT in push mode AND physically stuck
            if self.push_state == 0 and self.path_valid and moved_dist < 0.5:
                self.patience += 1
                if self.patience > self.patience_threshold: # Re-plan after random interval
                    self.recalc_path()
                    self.patience = 0
                    self.patience_threshold = random.randint(30, 60)
                    self.push_state = 45 # Force movement for ~0.75s to break deadlock
            else:
                self.patience = 0

            if self.current_wp_index < len(self.path):
                target = pygame.Vector2(self.path[self.current_wp_index])
                direction = target - self.pos
                if direction.length() < AGENT_RADIUS:
                    self.pos = target 
                    self.current_wp_index += 1
                else:
                    self.velocity = direction.normalize() * AGENT_SPEED
                    self.pos += self.velocity

    def update_electric(self, agents, obstacles, end_rect):
        if not self.active: return
        self._check_parking_logic(end_rect)
        
        if self.spot_reserved:
            self.update_grid_step()
            self.resolve_collision(agents, obstacles)
            return

        if not self.path_valid: 
            self.resolve_collision(agents, obstacles); return

        if self.current_wp_index < len(self.path):
            target = pygame.Vector2(self.path[self.current_wp_index])
            desired = target - self.pos
            attraction = desired.normalize() * K_ATTRACTION if desired.length() > 0 else pygame.Vector2(0,0)
        else: attraction = pygame.Vector2(0,0)

        repulsion = pygame.Vector2(0, 0)
        for other in agents:
            if other is self: continue
            diff = self.pos - other.pos
            d = diff.length()
            if 0.1 < d < 60: repulsion += diff.normalize() * (K_AGENT / (d**2))

        wall_repulsion = pygame.Vector2(0, 0)
        for obs in obstacles:
            cx = clamp(self.pos.x, obs.left, obs.right)
            cy = clamp(self.pos.y, obs.top, obs.bottom)
            diff = self.pos - pygame.Vector2(cx, cy)
            d = diff.length()
            if 0.1 < d < 40: wall_repulsion += diff.normalize() * (K_WALL / (d**2))

        total = attraction + repulsion * 1.5 + wall_repulsion * 2.0
        if total.length() > MAX_FORCE: total = total.normalize() * MAX_FORCE
        
        self.velocity += total
        if self.velocity.length() > AGENT_SPEED: self.velocity = self.velocity.normalize() * AGENT_SPEED
        self.pos += self.velocity
        
        if self.current_wp_index < len(self.path):
            if self.pos.distance_to(self.path[self.current_wp_index]) < AGENT_RADIUS * 1.5:
                self.current_wp_index += 1
        
        self.resolve_collision(agents, obstacles)

    def _check_parking_logic(self, end_rect):
        if not self.spot_reserved:
            if end_rect.collidepoint(self.pos.x, self.pos.y):
                vel_check = self.velocity if self.velocity.length() > 0.1 else (self.target_pos - self.pos)
                spot = self.exit_manager.request_spot(vel_check)
                if spot:
                    self.target_grid_coord = spot
                    self.target_pos = self.exit_manager.get_pixel_center(*spot)
                    self.spot_reserved = True
                    self.path = []

    def get_color(self):
        if not self.path_valid: return (100, 100, 100)
        if not self.active: return PARKED_GREEN
        if self.waiting: return ORANGE
        return RED