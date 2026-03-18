# fundamental.py
# Shared primitives and A* algorithms used by all planners

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
    """
    Generic A* search. Returns list of nodes (start → goal) or None.
    """
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
            segments.append((path[i], path[i+1]))
    return segments

# ---------------------------------------------------------------------------
# A* for Standard Visibility-Graph planner
# ---------------------------------------------------------------------------
def astar_standard(start_idx, goal_idx, graph, nodes, goal_pos,
                   predecessor_paths=None):
    """
    A* over a pre-built visibility graph with predecessor path deprioritization.

    Parameters:
        start_idx / goal_idx : indices into `nodes`
        graph                : adjacency dict  {idx: [(neighbor_idx, weight), ...]}
        nodes                : list of (x, y) coordinates
        goal_pos             : (x, y) goal for heuristic
        predecessor_paths    : list of coordinate-path lists from earlier agents

    Returns:
        Coordinate path [(x,y), ...] or None.
    """
    path_segments = _build_path_segments(predecessor_paths)
    # Penalty zone radius: must be SMALLER than WIDE_CORNER_MAX_MARGIN so that
    # wide-corner paths can genuinely escape the penalty zone of tight-corner predecessors.
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
                    # Penalties now STACK for every predecessor path overlapping this segment.
                    # This strongly penalizes "queueing" behind multiple agents on the same path.
                    penalty += AGENT_DIAMETER * 4.0 * PRIORITY_PENALTY_MULTIPLIER
            
            # Inject symmetry-breaking noise (0 to 5.0) just like the Discrete Grid
            noise = random.uniform(0, 5.0)
            neighbors.append((next_idx, weight + penalty + noise))
        return neighbors

    def heuristic(idx):
        return dist(nodes[idx], goal_pos)

    path_indices = _astar_core(start_idx, goal_idx, get_neighbors, heuristic)
    if path_indices is None:
        return None
    return [nodes[i] for i in path_indices]


# ---------------------------------------------------------------------------
# Line-segment crossing test (for TBC algorithm)
# ---------------------------------------------------------------------------
def _cross(o, a, b):
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def _segments_cross(p1, p2, p3, p4):
    """
    Returns True if segment (p1→p2) properly crosses segment (p3→p4).
    Collinear / overlapping segments return False (same corridor = no penalty).
    """
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    # We intentionally exclude collinear and endpoint-touching cases.
    # Overlapping segments = same corridor = not a crossing.
    return False

def _build_per_path_segments(predecessor_paths):
    """
    Group segments by predecessor path index.
    Returns list of lists: [ [seg, seg, ...], [seg, seg, ...], ... ]
    Each inner list contains the segments of one predecessor path.
    """
    per_path = []
    if not predecessor_paths:
        return per_path
    for path in predecessor_paths:
        if not path or len(path) < 2:
            continue
        segs = []
        for i in range(len(path) - 1):
            segs.append((path[i], path[i+1]))
        per_path.append(segs)
    return per_path


# ---------------------------------------------------------------------------
# A* for TBC (Traffic-Based Crossing) planner
# ---------------------------------------------------------------------------
def astar_tbc(start_idx, goal_idx, graph, nodes, goal_pos,
              predecessor_paths=None):
    """
    A* over a visibility graph with CROSSING-based penalties.
    
    Unlike proximity-based penalty, TBC counts how many DISTINCT predecessor
    paths each edge physically crosses (intersects). Agents that follow
    the same corridor as a predecessor incur 0 crossing penalty.
    
    This naturally creates 2-3 corridors instead of 16 unique paths.
    """
    per_path_segs = _build_per_path_segments(predecessor_paths)
    path_segments = _build_path_segments(predecessor_paths)
    proximity_r = AGENT_DIAMETER * 0.75
    
    # Convergence zone: don't penalize crossings near the goal
    convergence_r_sq = (AGENT_DIAMETER * 3) ** 2
    gx, gy = goal_pos
    
    # Crossing penalty per distinct path crossed
    CROSSING_PENALTY = 200.0

    def get_neighbors(idx):
        neighbors = []
        for next_idx, weight in graph[idx]:
            nx, ny = nodes[next_idx]
            cx, cy = nodes[idx]
            
            # Skip crossing penalty near the goal convergence zone
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
                if path_crossed:
                    crossings += 1
            
            penalty = crossings * CROSSING_PENALTY * PRIORITY_PENALTY_MULTIPLIER
            noise = random.uniform(0, 5.0)
            neighbors.append((next_idx, weight + penalty + noise))
        return neighbors

    def heuristic(idx):
        return dist(nodes[idx], goal_pos)

    path_indices = _astar_core(start_idx, goal_idx, get_neighbors, heuristic)
    if path_indices is None:
        return None
    return [nodes[i] for i in path_indices]


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------
class Agent:
    def __init__(self, start, exit_manager, planner, predecessor_paths=None):
        self.pos = pygame.Vector2(start)
        self.exit_manager = exit_manager
        self.planner = planner
        self.predecessor_paths = predecessor_paths or []
        self.active = True
        self.waiting = False
        self.spot_reserved = False 
        self.target_grid_coord = None
        
        # Parking state
        self.dfs_current_cell = None
        self.dfs_settled = False
        self.grid_path = []
        self.grid_patience = 0
        self.grid_step_cooldown = 0     # Frames until next discrete step
        
        # Smooth parking blend: 0.0 = full A*, 1.0 = full funnel
        self.parking_blend = 0.0

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
        self.target_pos = self.exit_manager.center_pixel
        new_path = self.planner.find_path(self, self.target_pos, self.predecessor_paths)
        if new_path:
            self.path = new_path
            if len(new_path) > 1:
                self.current_wp_index = 1
            else:
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
            
            # Only hard-brake if they are literally about to collide (< 1.2 robot lengths).
            if distance < AGENT_DIAMETER * 1.2: 
                d_norm = d_vec.normalize()
                angle = heading.dot(d_norm)
                if angle > 0.8: 
                    # Smart Following: Don't wait if the agent ahead is moving fast enough
                    if other.velocity.length() > 0.5:
                        move_alignment = heading.dot(other.velocity.normalize())
                        if move_alignment > 0.7:
                            if distance > AGENT_DIAMETER * 1.05:
                                continue
                    self.waiting = True; return 

    def resolve_collision(self, agents, obstacles):
        if not self.active: return

        # Shape-specific physics for obstacles
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

        # Agent-Agent
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
            self.patience += 1
            if self.patience > self.patience_threshold:
                self.recalc_path()
                self.patience = 0
                self.patience_threshold = random.randint(30, 60)
            return

        # Always compute the A* waypoint velocity (even while blending into parking)
        astar_velocity = pygame.Vector2(0, 0)
        if self.current_wp_index < len(self.path):
            target = pygame.Vector2(self.path[self.current_wp_index])
            direction = target - self.pos
            if direction.length() < AGENT_RADIUS:
                self.pos = target
                self.current_wp_index += 1
            else:
                astar_velocity = direction.normalize() * AGENT_SPEED

        # Always compute the funnel pull velocity (even before fully in parking)
        funnel_velocity = pygame.Vector2(0, 0)
        if self.spot_reserved:
            funnel_target = self.exit_manager.center_pixel
            diff = funnel_target - self.pos
            dist_to_funnel = diff.length()
            if dist_to_funnel > 2.0:
                pull_strength = AGENT_SPEED * 0.8
                funnel_velocity = diff.normalize() * pull_strength

        blend = self.parking_blend  # 0.0 = pure A*, 1.0 = pure funnel

        if blend >= 1.0:
            # Fully in parking funnel — hand off entirely to exit_manager
            self.exit_manager.update_agent(self)
        elif blend > 0.0:
            # Blending: lerp between A* velocity and funnel pull
            blended_vel = astar_velocity * (1.0 - blend) + funnel_velocity * blend
            if blended_vel.length() > AGENT_SPEED:
                blended_vel.scale_to_length(AGENT_SPEED)
            self.velocity = blended_vel
            self.pos += self.velocity
        else:
            # Pure A* path following
            if not self.waiting:
                moved_dist = (self.pos - self.prev_pos).length()
                self.prev_pos = pygame.Vector2(self.pos)
                if self.push_state == 0 and self.path_valid and moved_dist < 0.5:
                    self.patience += 1
                    if self.patience > self.patience_threshold:
                        self.recalc_path()
                        self.patience = 0
                        self.patience_threshold = random.randint(30, 60)
                        self.push_state = 45
                else:
                    self.patience = 0
            self.velocity = astar_velocity
            self.pos += self.velocity

        # Always tick the entry check so parking_blend ramps up each frame
        self._check_parking_logic(end_rect)

    def get_color(self):
        if not self.path_valid: return (100, 100, 100)
        if not self.active: return PARKED_GREEN
        if self.waiting: return ORANGE
        return RED