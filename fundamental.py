# fundamental.py
# Shared primitives, A* algorithms, and core controllers used by all planners

import math
import pygame
import heapq
import random
from config import *


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))


# ---------------------------------------------------------------------------
# Generic A* core (internal)
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


# ---------------------------------------------------------------------------
# Predecessor path penalty computation
# ---------------------------------------------------------------------------
def _point_to_segment_dist(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

def _build_path_segments(predecessor_paths):
    segments = []
    if not predecessor_paths:
        return segments
    for path in predecessor_paths:
        if not path or len(path) < 2:
            continue
        for i in range(len(path) - 1):
            segments.append((path[i], path[i+1]))
    return segments

# ---------------------------------------------------------------------------
# A* for Standard Visibility-Graph planner
# ---------------------------------------------------------------------------
def astar_standard(start_idx, goal_idx, graph, nodes, goal_pos, predecessor_paths=None):
    path_segments = _build_path_segments(predecessor_paths)
    proximity_r = AGENT_DIAMETER * 0.75

    def get_neighbors(idx):
        neighbors = []
        for next_idx, weight in graph[idx]:
            penalty = 0
            nx, ny = nodes[next_idx]
            cx, cy = nodes[idx]
            
            mid_x, mid_y = (cx + nx) / 2, (cy + ny) / 2
            for seg in path_segments:
                d1 = _point_to_segment_dist(nx, ny, seg[0][0], seg[0][1], seg[1][0], seg[1][1])
                d2 = _point_to_segment_dist(mid_x, mid_y, seg[0][0], seg[0][1], seg[1][0], seg[1][1])
                if min(d1, d2) < proximity_r:
                    penalty += AGENT_DIAMETER * 4.0 * PRIORITY_PENALTY_MULTIPLIER
            
            neighbors.append((next_idx, weight + penalty))
        return neighbors

    def heuristic(idx):
        return dist(nodes[idx], goal_pos)

    path_indices = _astar_core(start_idx, goal_idx, get_neighbors, heuristic)
    if path_indices is None: return None
    return [nodes[i] for i in path_indices]


# ---------------------------------------------------------------------------
# Line-segment crossing test (for TBC algorithm)
# ---------------------------------------------------------------------------
def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def _segments_cross(p1, p2, p3, p4):
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _build_per_path_segments(predecessor_paths):
    per_path = []
    if not predecessor_paths: return per_path
    for path in predecessor_paths:
        if not path or len(path) < 2: continue
        segs = []
        for i in range(len(path) - 1):
            segs.append((path[i], path[i+1]))
        per_path.append(segs)
    return per_path

# ---------------------------------------------------------------------------
# A* for TBC (Traffic-Based Crossing) planner
# ---------------------------------------------------------------------------
def astar_tbc(start_idx, goal_idx, graph, nodes, goal_pos, predecessor_paths=None):
    per_path_segs = _build_per_path_segments(predecessor_paths)
    convergence_r_sq = (AGENT_DIAMETER * 3) ** 2
    gx, gy = goal_pos
    CROSSING_PENALTY = 200.0

    def get_neighbors(idx):
        neighbors = []
        for next_idx, weight in graph[idx]:
            nx, ny = nodes[next_idx]
            cx, cy = nodes[idx]
            
            if (nx - gx)**2 + (ny - gy)**2 < convergence_r_sq:
                noise = random.uniform(0, 5.0)
                neighbors.append((next_idx, weight + noise))
                continue
            
            crossings = 0
            for path_segs in per_path_segs:
                path_crossed = False
                for seg in path_segs:
                    if _segments_cross((cx, cy), (nx, ny), seg[0], seg[1]):
                        path_crossed = True
                        break
                if path_crossed: crossings += 1
            
            penalty = crossings * CROSSING_PENALTY * PRIORITY_PENALTY_MULTIPLIER
            noise = random.uniform(0, 5.0)
            neighbors.append((next_idx, weight + penalty + noise))
        return neighbors

    def heuristic(idx):
        return dist(nodes[idx], goal_pos)

    path_indices = _astar_core(start_idx, goal_idx, get_neighbors, heuristic)
    if path_indices is None: return None
    return [nodes[i] for i in path_indices]


# ---------------------------------------------------------------------------
# CENTRAL CONTROLLER (The Brain)
# ---------------------------------------------------------------------------
def path_length(path):
    if not path or len(path) < 2:
        return float("inf")
    total = 0.0
    for i in range(len(path) - 1):
        total += math.hypot(
            path[i + 1][0] - path[i][0],
            path[i + 1][1] - path[i][1]
        )
    return total
class CentralManager:
    """
    Handles global pathfinding, prioritizes agents by distance to goal, 
    and applies path penalties to trailing agents.
    """
    def __init__(self, planner, exit_manager):
        self.planner = planner
        self.exit_manager = exit_manager

    def plan_all_paths(self, agents, force_replace=False):
        if not self.planner or not agents:
            return

        goal_pos = self.exit_manager.center_pixel

        # 1. Rank active, unparked agents by proximity to the goal
        active_agents = [a for a in agents if a.active and not a.spot_reserved]
        active_agents.sort(key=lambda a: math.hypot(a.pos.x - goal_pos.x, a.pos.y - goal_pos.y))

        predecessor_paths = []

        # 2. Sequential A* with Penalty Accumulation
        for i, agent in enumerate(active_agents):
            agent.index = i + 1  # 1 = closest to goal (highest priority)

            old_path = list(agent.path) if agent.path else []
            old_wp_index = agent.current_wp_index

            # The planner uses predecessor_paths to heavily penalize overlap
            new_path = self.planner.find_path(agent, goal_pos, predecessor_paths)

            if new_path:
                use_new = True

                # If old path exists, only switch if the new path is clearly better
                if (not force_replace) and old_path and old_wp_index < len(old_path):
                    new_cost = path_length(new_path)
                    old_remaining = old_path[old_wp_index:]
                    old_cost = path_length(old_remaining)

                    # Only switch if new path is at least 10% better
                    if new_cost >= old_cost * 0.90:
                        use_new = False

                if use_new:
                    agent.path = new_path
                    agent.path_valid = True
                    agent.current_wp_index = 1 if len(new_path) > 1 else 0

                    if len(new_path) > 1:
                        predecessor_paths.append(list(new_path))
                else:
                    agent.path = old_path
                    agent.path_valid = True
                    agent.current_wp_index = min(old_wp_index, len(old_path) - 1)

                    if len(old_path) > 1:
                        predecessor_paths.append(list(old_path))
            else:
                # If replanning fails, keep the old path if it is still usable
                if old_path and old_wp_index < len(old_path):
                    agent.path = old_path
                    agent.path_valid = True
                    agent.current_wp_index = min(old_wp_index, len(old_path) - 1)

                    if len(old_path) > 1:
                        predecessor_paths.append(list(old_path))
                else:
                    agent.path = []
                    agent.path_valid = False
                    agent.current_wp_index = 0


# ---------------------------------------------------------------------------
# LOCAL CONTROLLER (The Muscle)
# ---------------------------------------------------------------------------
class Agent:
    def __init__(self, start, exit_manager):
        self.pos = pygame.Vector2(start)
        self.exit_manager = exit_manager
        
        self.active = True
        self.waiting = False
        self.spot_reserved = False 
        self.target_grid_coord = None
        
        # Stuck / Deadlock Detection
        self.stuck_timer = 0
        self.wait_timer  = 0   # how long we've been held in 'waiting'
        self.is_stuck = False
        
        # Parking state
        self.dfs_current_cell = None
        self.dfs_settled = False
        self.grid_path = []
        self.grid_patience = 0
        self.grid_step_cooldown = 0
        self.parking_blend = 0.0

        # Local Path Tracking State
        self.path_valid = False
        self.path = []
        self.current_wp_index = 0
        self.velocity = pygame.Vector2(0, 0)
        self.push_state = 0 
        self.prev_pos = pygame.Vector2(start)

    def _update_waypoint(self):
        """
        Projects agent position onto path segments. Ensures smooth path re-joining
        without backtracking if pushed off course.
        """
        if not self.path or self.current_wp_index >= len(self.path):
            return

        # Snap to the closest valid forward segment
        best_index = self.current_wp_index
        min_dist_sq = float('inf')
        
        for i in range(max(1, self.current_wp_index), len(self.path)):
            prev_wp = pygame.Vector2(self.path[i-1])
            curr_wp = pygame.Vector2(self.path[i])
            
            seg_vec = curr_wp - prev_wp
            seg_len_sq = seg_vec.length_squared()
            
            if seg_len_sq == 0:
                d_sq = self.pos.distance_squared_to(curr_wp)
            else:
                t = clamp((self.pos - prev_wp).dot(seg_vec) / seg_len_sq, 0, 1)
                proj = prev_wp + t * seg_vec
                d_sq = self.pos.distance_squared_to(proj)
            
            if d_sq < min_dist_sq - 1.0: 
                min_dist_sq = d_sq
                best_index = i
                
        self.current_wp_index = best_index

        # Normal progression check for the active segment
        if self.current_wp_index < len(self.path):
            target = pygame.Vector2(self.path[self.current_wp_index])
            direction = target - self.pos
            
            if direction.length_squared() < AGENT_RADIUS**2:
                self.current_wp_index += 1
            elif self.current_wp_index > 0:
                prev = pygame.Vector2(self.path[self.current_wp_index - 1])
                segment = target - prev
                if segment.length_squared() > 0 and direction.dot(segment) < 0:
                    self.current_wp_index += 1

    def local_safety_check(self, agents):
        self.waiting = False
        if not self.active: return
        
        if self.push_state > 0:
            self.push_state -= 1
            return

        heading = None
        if self.velocity.length() > 0.1:
            heading = self.velocity.normalize()
        elif self.path and self.current_wp_index < len(self.path):
            tgt = pygame.Vector2(self.path[self.current_wp_index])
            diff = tgt - self.pos
            if diff.length() > 0.1: heading = diff.normalize()
        elif self.grid_path:
            tgt = self.exit_manager.get_pixel_center(*self.grid_path[0])
            diff = tgt - self.pos
            if diff.length() > 0.1: heading = diff.normalize()
            
        if heading is None: return

        for other in agents:
            if other is self or not other.active: continue
            d_vec = other.pos - self.pos
            distance = d_vec.length()
            if distance < 0.1: continue 
            
            if distance < AGENT_DIAMETER * 1.2: 
                d_norm = d_vec.normalize()
                angle = heading.dot(d_norm)
                if angle > 0.8:
                    # Corner-turn check: compute where the other agent is actually heading.
                    # If their intended direction differs substantially from ours (crossing
                    # paths / one of us just turned a corner), don't freeze — let the
                    # physics collision resolver handle the overlap instead.
                    other_heading = None
                    if other.velocity.length() > 0.5:
                        other_heading = other.velocity.normalize()
                    elif other.path and other.current_wp_index < len(other.path):
                        tgt2 = pygame.Vector2(other.path[other.current_wp_index])
                        d2   = tgt2 - other.pos
                        if d2.length() > 0.1:
                            other_heading = d2.normalize()
                    if other_heading is not None and heading.dot(other_heading) < 0.4:
                        continue   # crossing / turning — not a queue; skip the freeze

                    if other.velocity.length() > 0.5:
                        move_alignment = heading.dot(other.velocity.normalize())
                        if move_alignment > 0.7:
                            if distance > AGENT_DIAMETER * 1.05: continue
                    self.waiting = True; return 

    def resolve_collision(self, agents, obstacles):
        if not self.active: return

        from algorithms.obstacles import CircleObstacle, FreehandObstacle

        for obs in obstacles:
            if isinstance(obs, CircleObstacle):
                diff = self.pos - pygame.Vector2(obs.cx, obs.cy)
                dist = diff.length()
                min_dist = AGENT_RADIUS + obs.r
                if dist < min_dist:
                    if dist == 0: self.pos.x += 1
                    else:
                        overlap = min_dist - dist
                        self.pos += diff.normalize() * overlap
            elif isinstance(obs, FreehandObstacle):
                closest_point_on_shape = None
                min_dist_sq = float('inf')
                for i in range(len(obs.points) - 1):
                    p1, p2 = pygame.Vector2(obs.points[i]), pygame.Vector2(obs.points[i+1])
                    p1_to_agent = self.pos - p1
                    seg_vec = p2 - p1
                    seg_len_sq = seg_vec.length_squared()
                    t = p1_to_agent.dot(seg_vec) / seg_len_sq if seg_len_sq > 0 else 0
                    t = clamp(t, 0, 1)
                    closest_on_segment = p1 + t * seg_vec
                    dist_sq = self.pos.distance_squared_to(closest_on_segment)
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        closest_point_on_shape = closest_on_segment
                if closest_point_on_shape and min_dist_sq < (AGENT_RADIUS + obs.thickness / 2)**2:
                    dist = math.sqrt(min_dist_sq)
                    overlap = (AGENT_RADIUS + obs.thickness / 2) - dist
                    push_vec = self.pos - closest_point_on_shape
                    if push_vec.length() > 0: self.pos += push_vec.normalize() * overlap
                    else: self.pos.x += overlap
            else:
                closest_point = pygame.Vector2(clamp(self.pos.x, obs.left, obs.right), clamp(self.pos.y, obs.top, obs.bottom))
                diff = self.pos - closest_point
                dist = diff.length()
                if dist < AGENT_RADIUS:
                    if dist == 0: self.pos.x += 1
                    else:
                        overlap = AGENT_RADIUS - dist
                        self.pos += diff.normalize() * overlap
        
        if self.pos.x < AGENT_RADIUS: self.pos.x = AGENT_RADIUS
        if self.pos.x > MAP_WIDTH - AGENT_RADIUS: self.pos.x = MAP_WIDTH - AGENT_RADIUS
        if self.pos.y < AGENT_RADIUS: self.pos.y = AGENT_RADIUS
        if self.pos.y > MAP_HEIGHT - AGENT_RADIUS: self.pos.y = MAP_HEIGHT - AGENT_RADIUS

        for other in agents:
            if other is self: continue
            
            if self.spot_reserved and other.spot_reserved:
                 self.exit_manager.resolve_collision(self, other)
                 continue

            if self.spot_reserved and not other.active:
                if self.pos.distance_to(other.pos) < AGENT_DIAMETER:
                    self.velocity.update(0, 0)
                    self.active = False
                    self.exit_manager.park_agent(self)
                    return

            min_dist = AGENT_DIAMETER + 1 
            push_factor = 0.5
            
            diff = self.pos - other.pos
            dist = diff.length()
            if dist < min_dist:
                if dist == 0: correction = pygame.Vector2(1, 0)
                else:
                    overlap = min_dist - dist
                    correction = diff.normalize() * (overlap * push_factor)
                self.pos += correction

    def _check_parking_logic(self, end_rect):
        self.exit_manager.check_entry(self)

    def update(self, end_rect):
        if not self.active: return
        
        if not self.path_valid:
            self.velocity = pygame.Vector2(0, 0)
            return

        self._update_waypoint()

        astar_velocity = pygame.Vector2(0, 0)
        if self.current_wp_index < len(self.path):
            target = pygame.Vector2(self.path[self.current_wp_index])
            direction = target - self.pos
            if direction.length() > 0:
                astar_velocity = direction.normalize() * AGENT_SPEED

        funnel_velocity = pygame.Vector2(0, 0)
        if self.spot_reserved:
            funnel_target = self.exit_manager.center_pixel
            diff = funnel_target - self.pos
            if diff.length() > 2.0:
                pull_strength = AGENT_SPEED * 0.8
                funnel_velocity = diff.normalize() * pull_strength

        blend = self.parking_blend 

        if blend >= 1.0:
            self.exit_manager.update_agent(self)
        elif blend > 0.0:
            blended_vel = astar_velocity * (1.0 - blend) + funnel_velocity * blend
            if blended_vel.length() > AGENT_SPEED:
                blended_vel.scale_to_length(AGENT_SPEED)
            self.velocity = blended_vel
            self.pos += self.velocity
        else:
            if not self.waiting:
                self.velocity = astar_velocity
                self.pos += self.velocity

        # STUCK DETECTION: moving very slowly outside parking / waiting state
        if blend == 0.0 and self.velocity.length() < 0.2 and not self.waiting:
            self.stuck_timer += 1
        else:
            self.stuck_timer = 0

        # DEADLOCK DETECTION: continuously waiting ⟹ probably a circular block
        if blend == 0.0 and self.waiting:
            self.wait_timer += 1
        else:
            self.wait_timer = 0

        # 60 frames ≈ 1 s stuck;  90 frames ≈ 1.5 s deadlocked
        if self.stuck_timer > 60 or self.wait_timer > 90:
            self.is_stuck = True
            self.stuck_timer = 0
            self.wait_timer  = 0

        self._check_parking_logic(end_rect)

    def get_color(self):
        if not self.path_valid: return (100, 100, 100)
        if not self.active: return PARKED_GREEN
        if self.waiting: return ORANGE
        return RED