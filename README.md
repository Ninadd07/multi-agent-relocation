# MARS — Multi-Agent Relocation Simulator

A real-time simulation of multiple agents navigating from a spawn zone to a parking zone while avoiding obstacles and each other. Three distinct pathfinding algorithms are implemented, all built on a shared generic A\* core.

---

## Project Structure

```
├── config.py                    # Simulation constants (dimensions, physics, colors)
├── fundamental.py               # Shared primitives: dist(), clamp(), astar()
├── parking.py                   # SmartExit parking planner + zone layout
├── gui.py                       # Pygame UI, rendering, and main simulation loop
├── algorithms/
│   ├── field.py                 # Agent class (movement, collision, parking logic)
│   ├── standard.py              # Visibility-graph A* (GlobalPlanner)
│   └── discrete_grid.py         # Adaptive quadtree-grid A* (Discretisation)
```

---

## The Fundamental A\* Engine

**File:** `fundamental.py`

All pathfinding in the project routes through a single, generic A\* function:

```python
def astar(start, goal, get_neighbors, heuristic):
```

| Parameter | Type | Description |
|---|---|---|
| `start` | hashable | Start node identifier |
| `goal` | hashable | Goal node identifier |
| `get_neighbors(node)` | callback | Returns `[(neighbor, edge_cost), ...]` |
| `heuristic(node)` | callback | Returns estimated cost from `node` to `goal` |

**Returns:** Ordered list of nodes from `start` → `goal`, or `None` if unreachable.

**How it works:**
1. Maintains an open set (min-heap) sorted by `f = g + h`, with a counter tiebreaker for equal priorities.
2. For each node popped, queries `get_neighbors()` for adjacent nodes and their edge costs.
3. Relaxes edges: if a cheaper path to a neighbor is found, updates `g_score` and `came_from`.
4. On reaching `goal`, reconstructs the path via the `came_from` chain.

Each algorithm plugs into this engine by providing its own `get_neighbors` and `heuristic` implementations, which encode the algorithm's unique graph structure, cost model, and search behavior.

---

## Algorithm 1: Standard Visibility-Graph A\*

**File:** `algorithms/standard.py` → `GlobalPlanner`

### Concept

Constructs a **visibility graph** over inflated obstacle corners, then runs A\* over it. This is a classic approach for 2D pathfinding in continuous space with polygonal obstacles.

### How It Links to `fundamental.astar()`

```python
# Graph is pre-built as adjacency list: graph[i] = [(neighbor_idx, distance), ...]
get_neighbors = lambda idx: graph[idx]
heuristic     = lambda idx: dist(nodes[idx], end_pos)

path_indices = astar(0, target_idx, get_neighbors, heuristic)
```

### Detailed Pipeline

```
 ┌──────────────────────────────────────────────────────────────┐
 │  1. OBSTACLE INFLATION                                      │
 │     Each obstacle rect is inflated by (AGENT_DIAMETER + 6)  │
 │     to create bloated_obstacles — ensures the agent's       │
 │     center never gets too close to a wall.                   │
 ├──────────────────────────────────────────────────────────────┤
 │  2. CORNER EXTRACTION                                       │
 │     For each bloated obstacle, extract 4 corner points      │
 │     (with a small margin). Discard any corner that falls    │
 │     inside another bloated obstacle.                         │
 ├──────────────────────────────────────────────────────────────┤
 │  3. VISIBILITY GRAPH CONSTRUCTION                           │
 │     nodes = [start, goal] + valid_corners                   │
 │     For every pair (i, j), check if the line segment        │
 │     between them is clear of all bloated obstacles           │
 │     (using pygame's clipline). If clear, add bidirectional  │
 │     edge with weight = Euclidean distance.                   │
 ├──────────────────────────────────────────────────────────────┤
 │  4. A* SEARCH (via fundamental.astar)                       │
 │     start = index 0, goal = index 1                         │
 │     get_neighbors: returns pre-computed graph[idx]           │
 │     heuristic: Euclidean distance to goal position           │
 │     Returns: list of node indices → mapped to coordinates   │
 └──────────────────────────────────────────────────────────────┘
```

### Characteristics
- **Optimal** in continuous 2D with convex obstacles
- **No dynamic obstacle avoidance** built into the graph — relies on the Agent's `local_safety_check()` to handle other agents
- **Fast** for small obstacle counts (graph size = O(corners²))

---

## Algorithm 2: Electric Field (Artificial Potential Field)

**File:** `algorithms/field.py` → `Agent.update_electric()`

### Concept

Combines A\* **global planning** (for waypoints) with **artificial potential fields** (for real-time steering). The Agent follows waypoints from the GlobalPlanner, but its frame-by-frame motion is governed by attractive and repulsive electric-field forces.

### How It Links to `fundamental.astar()`

The electric field algorithm does **not** call `astar()` directly for steering. Instead, it uses `GlobalPlanner.find_path()` (which calls `astar()`) to produce waypoints. The potential field then steers the agent toward each waypoint while dynamically avoiding obstacles and other agents.

```
GlobalPlanner.find_path()          ← uses fundamental.astar()
       │
       ▼
  waypoints [wp0, wp1, ..., wpN]
       │
       ▼
  Agent.update_electric()          ← potential field steering per frame
```

### Force Model

Each frame, three forces are computed and summed:

| Force | Formula | Purpose |
|---|---|---|
| **Attraction** | `direction_to_waypoint × K_ATTRACTION` | Pulls agent toward next waypoint |
| **Agent Repulsion** | `Σ (diff / d²) × K_AGENT × 1.5` | Pushes away from nearby agents (within 60px) |
| **Wall Repulsion** | `Σ (diff / d²) × K_WALL × 2.0` | Pushes away from obstacle surfaces (within 40px) |

```
total_force = attraction + agent_repulsion + wall_repulsion
velocity += clamp(total_force, MAX_FORCE)
velocity = clamp(velocity, AGENT_SPEED)
position += velocity
```

### Waypoint Advancement
The agent advances to the next waypoint when it gets within `AGENT_RADIUS × 1.5` of the current one. After movement, `resolve_collision()` enforces hard constraints (no overlap with obstacles or agents).

### Characteristics
- **Smooth, natural-looking** paths with gentle curves around obstacles
- **Real-time reactive** — automatically adjusts to moving agents
- **Can get stuck** in local minima (agent between two repulsive sources), mitigated by the global A\* waypoints

---

## Algorithm 3: Adaptive Quadtree-Grid A\*

**File:** `algorithms/discrete_grid.py` → `Discretisation`

### Concept

Discretizes the continuous world into an **adaptive quadtree** — fine cells near obstacles, coarse cells in open space — then runs A\* over the cell adjacency graph. This balances path quality with computational cost.

### How It Links to `fundamental.astar()`

```python
def get_neighbors(cell_idx):
    # For each cell, find geometrically adjacent cells
    # Cost = Euclidean distance + dynamic agent penalty + random noise
    return [(neighbor_idx, cost), ...]

def heuristic(cell_idx):
    return euclidean_distance(cell_center, goal_center)

path_indices = astar(start_idx, end_idx, get_neighbors, heuristic)
```

### Detailed Pipeline

```
 ┌──────────────────────────────────────────────────────────────┐
 │  1. QUADTREE CONSTRUCTION (build_occupancy_grid)            │
 │     Start with the full world as one cell.                  │
 │     Recursively subdivide into 4 children if:               │
 │       • Cell is near an obstacle AND size > min_grid (8px)  │
 │       • Cell is free AND size > base_grid (64px)            │
 │     Leaf cells are tagged as free or occupied.               │
 │                                                              │
 │     Result: Coarse cells (64px) in open areas,              │
 │             Fine cells (8px) near walls.                     │
 ├──────────────────────────────────────────────────────────────┤
 │  2. CELL ADJACENCY (get_neighbors callback)                 │
 │     For a given cell, iterate all other cells and check     │
 │     geometric adjacency (shared edge or corner):            │
 │       • touch_x: vertical edge shared (side neighbors)      │
 │       • touch_y: horizontal edge shared (top/bottom)        │
 │       • touch_diag: corner shared (diagonal neighbors)      │
 │     Skip occupied cells entirely.                            │
 ├──────────────────────────────────────────────────────────────┤
 │  3. DYNAMIC COST MODEL                                      │
 │     edge_cost = euclidean_dist                               │
 │               + agent_penalty (60 if <2.5r, 10 if <6r)      │
 │               + random_noise (0–5, breaks symmetry)          │
 │                                                              │
 │     Agent penalty makes the path curve around other agents.  │
 │     Random noise prevents all agents from choosing the       │
 │     exact same path simultaneously.                          │
 ├──────────────────────────────────────────────────────────────┤
 │  4. A* SEARCH (via fundamental.astar)                       │
 │     Returns cell indices → mapped to cell centers.           │
 ├──────────────────────────────────────────────────────────────┤
 │  5. PATH SMOOTHING (smooth_path)                            │
 │     Greedy line-of-sight optimization:                       │
 │     From current point, try to connect to the furthest      │
 │     visible point (skip intermediate waypoints).             │
 │     Uses segment safety checks with a 0.2px margin.         │
 └──────────────────────────────────────────────────────────────┘
```

### Characteristics
- **Adaptive resolution** — doesn't waste computation on open space
- **Built-in multi-agent avoidance** via the dynamic cost penalties
- **Symmetry breaking** via random noise prevents agent herding
- **Path smoothing** removes unnecessary zigzag through grid cells
- **Grid is cached** and only rebuilt when obstacles change

---

## Shared Agent Behavior

**File:** `algorithms/field.py` → `Agent`

Regardless of which algorithm computes the path, every agent shares the same behavior for:

### Local Safety Check (`local_safety_check`)
Before moving each frame (standard mode only), agents look ahead in their direction of travel. If another agent is directly ahead (within `VIEW_DISTANCE × 0.75`, alignment > 0.9), the agent **waits** — unless the agent ahead is moving in the same direction (smart following).

### Collision Resolution (`resolve_collision`)
Hard constraint enforcement after each movement step:
- **Wall collision**: pushes agent out of obstacle bounds and map edges
- **Agent collision**: separates overlapping agents by half the overlap distance

### Smart Exit Parking (`parking.py` → `SmartExit`)
When an agent enters the parking zone, it requests a grid slot from `SmartExit`, which assigns spots in a **deepest-first** order based on entry velocity. The agent then uses BFS (`find_next_grid_step`) to navigate to its assigned cell, treating other parked agents as dynamic obstacles.

### Deadlock Recovery
Agents track physical displacement. If stuck for a random threshold (30–60 frames), they re-plan their path and enter **push mode** (45 frames of forced movement ignoring safety checks) to break deadlocks.

---

## Dependency Graph

```
                    config.py
                       │
                 fundamental.py
                 (dist, clamp, astar)
                  /      |       \
                 /       |        \
  standard.py   field.py    parking.py
  (GlobalPlanner) (Agent)    (SmartExit)
       │            │            │
       └────────────┼────────────┘
                    │
                  gui.py
                    
  discrete_grid.py  ← imports fundamental.astar + config directly
  (Discretisation)
```

---

## Running

```bash
python3 gui.py
```

Use the sidebar to select an algorithm mode, place start/end zones, draw walls, and press Enter to begin the simulation.
