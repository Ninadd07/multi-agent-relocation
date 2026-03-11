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
            
            # Skip crossing penalty if this edge is near the goal (convergence zone)
            mid_x, mid_y = (cx + nx) / 2, (cy + ny) / 2
            near_goal = ((mid_x - gx)**2 + (mid_y - gy)**2) < convergence_r_sq
            
            penalty = 0
            if not near_goal and per_path_segs:
                # 1) CROSSING PENALTY: count distinct predecessor paths this edge crosses
                paths_crossed = 0
                for path_segs in per_path_segs:
                    crossed_this_path = False
                    for seg in path_segs:
                        if _segments_cross((cx, cy), (nx, ny), seg[0], seg[1]):
                            crossed_this_path = True
                            break
                    if crossed_this_path:
                        paths_crossed += 1
                
                penalty += paths_crossed * CROSSING_PENALTY
            
            # 2) PROXIMITY PENALTY: penalize edges near predecessor paths (anti-queueing)
            for seg in path_segments:
                d1 = _point_to_segment_dist(nx, ny, seg[0][0], seg[0][1], seg[1][0], seg[1][1])
                d2 = _point_to_segment_dist(mid_x, mid_y, seg[0][0], seg[0][1], seg[1][0], seg[1][1])
                if min(d1, d2) < proximity_r:
                    penalty += AGENT_DIAMETER * 4.0 * PRIORITY_PENALTY_MULTIPLIER
            
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
# A* for Discrete Adaptive-Grid planner
# ---------------------------------------------------------------------------
def astar_discrete(start_idx, goal_idx, grid_cells, end_center,
                   agent_positions, robot_radius, predecessor_paths=None):
    """
    A* over an adaptive quadtree grid with dynamic agent penalties,
    symmetry-breaking noise, and predecessor path deprioritization.

    Parameters:
        start_idx / goal_idx : cell indices into `grid_cells`
        grid_cells           : [(x0, y0, w, h, is_free), ...]
        end_center           : (x, y) goal center for heuristic
        agent_positions      : [(x, y), ...] of other active agents
        robot_radius         : agent radius for penalty zones
        predecessor_paths    : list of coordinate-path lists from earlier agents

    Returns:
        List of cell indices (start → goal) or None.
    """
    path_segments = _build_path_segments(predecessor_paths)
    inner_r_sq = (robot_radius * 2.5) ** 2
    outer_r_sq = (robot_radius * 6.0) ** 2
    radius_6 = robot_radius * 6.0

    def get_neighbors(current_idx):
        current_cell = grid_cells[current_idx]
        cx = current_cell[0] + current_cell[2] / 2
        cy = current_cell[1] + current_cell[3] / 2
        cw, ch = current_cell[2], current_cell[3]
        neighbors = []

        for i, cell in enumerate(grid_cells):
            if i == current_idx or not cell[4]:
                continue

            nx = cell[0] + cell[2] / 2
            ny = cell[1] + cell[3] / 2
            nw, nh = cell[2], cell[3]

            # Quick bounding-box adjacency reject
            if abs(nx - cx) > (cw + nw) / 2 + 0.1:
                continue
            if abs(ny - cy) > (ch + nh) / 2 + 0.1:
                continue

            # Geometric adjacency check
            dx_val = abs(nx - cx)
            dy_val = abs(ny - cy)
            sum_w = (cw + nw) / 2
            sum_h = (ch + nh) / 2
            EPS = 0.1

            touch_x = (abs(dx_val - sum_w) < EPS) and (dy_val < sum_h - EPS)
            touch_y = (abs(dy_val - sum_h) < EPS) and (dx_val < sum_w - EPS)
            touch_diag = (abs(dx_val - sum_w) < EPS) and (abs(dy_val - sum_h) < EPS)

            if not (touch_x or touch_y or touch_diag):
                continue

            edge_dist = math.hypot(nx - cx, ny - cy)

            # --- Dynamic agent penalty ---
            penalty = 0
            for ax, ay in agent_positions:
                if abs(nx - ax) > radius_6 or abs(ny - ay) > radius_6:
                    continue
                d_sq = (nx - ax) ** 2 + (ny - ay) ** 2
                if d_sq < inner_r_sq:
                    penalty += 60
                elif d_sq < outer_r_sq:
                    penalty += 10

            # --- Predecessor path penalty ---
            for seg in path_segments:
                if _point_to_segment_dist(nx, ny, seg[0][0], seg[0][1], seg[1][0], seg[1][1]) < AGENT_DIAMETER * 1.5:
                    penalty += AGENT_DIAMETER
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

    return _astar_core(start_idx, goal_idx, get_neighbors, heuristic)


class Agent:
    def __init__(self, start, exit_manager, planner, predecessor_paths=None):
        self.pos = pygame.Vector2(start)
        self.exit_manager = exit_manager
        self.planner = planner
        self.predecessor_paths = predecessor_paths or []
        self.target_pos = pygame.Vector2(exit_manager.rect.center)
        
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
        new_path = self.planner.find_path(self, self.target_pos, self.predecessor_paths)
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
            # The 4x4 spawn grid has 1.5 spacing, so this ensures agents do not mistakenly 
            # brake for their stationary neighbors at T=0, allowing simultaneous departure!
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

        # Agent-Agent (CONTINUOUS PARKING PHYSICS ENABLED)
        for other in agents:
            if other is self: continue
            
            # Dampen forces if both are parked/parking to prevent violent bouncing, allowing dense packing
            if self.spot_reserved and other.spot_reserved:
                min_dist = AGENT_DIAMETER * 0.95 # Allow slight visible squishing in the crowd
                push_factor = 0.3 # Gentle nudging
            else:
                if self.spot_reserved: return # Do not let marching traffic push safely parked agents
                min_dist = AGENT_DIAMETER + 1 
                push_factor = 0.5 # Regular strong avoidance
            
            diff = self.pos - other.pos
            dist = diff.length()
            min_dist = AGENT_DIAMETER + 1 
            if dist < min_dist:
                if dist == 0: correction = pygame.Vector2(1, 0)
                else:
                    overlap = min_dist - dist
                    correction = diff.normalize() * (overlap * push_factor)
                self.pos += correction



    def _check_parking_logic(self, end_rect):
        """When agent enters the 20px threshold around end_rect, start continuous parking."""
        if not self.spot_reserved:
            # 20px radius = 40px inflation (20 on each side)
            threshold_rect = end_rect.inflate(40, 40)
            if threshold_rect.collidepoint(self.pos.x, self.pos.y):
                self.spot_reserved = True
                self.path = []  # Stop following A* path
                self.fluid_target_cell = None
                self.current_grid_cell = None


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
            # Physics-based continuous parking towards the absolute center of the box
            target = self.exit_manager.center_pixel
            diff = target - self.pos
            dist = diff.length()
            
            # If we are basically mathematically dead-center, stop forever
            if dist < 2.0:
                self.pos = pygame.Vector2(target)
                self.velocity = pygame.Vector2(0, 0)
                self.active = False
                self.exit_manager.park_agent(self)
                return
            
            # Drive straight towards the center point
            self.velocity = diff.normalize() * AGENT_SPEED
            self.pos += self.velocity
            
            # Have we stopped moving physically? (Because of physics crowding)
            # We measure this by checking if our actual position barely changed 
            # despite our velocity engine running.
            moved_dist = (self.pos - getattr(self, 'prev_parking_pos', pygame.Vector2(0,0))).length()
            self.prev_parking_pos = pygame.Vector2(self.pos)
            
            if moved_dist < 0.1:
                self.grid_patience += 1
                if self.grid_patience > 30:  # 0.5s of being blocked by the crowd = close enough!
                    self.velocity = pygame.Vector2(0, 0)
                    self.active = False
                    self.exit_manager.park_agent(self)
            else:
                self.grid_patience = 0
                
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

    def get_color(self):
        if not self.path_valid: return (100, 100, 100)
        if not self.active: return PARKED_GREEN
        if self.waiting: return ORANGE
        return RED
