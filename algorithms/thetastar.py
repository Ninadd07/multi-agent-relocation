# algorithms/thetastar.py
# Theta* path planner adapted from temp/central.py grid-based approach.

import math
import heapq
import pygame
from config import MAP_WIDTH, MAP_HEIGHT, AGENT_RADIUS

class ThetaStarPlanner:
    def __init__(self, obstacles):
        self.grid_size = 10
        self.cols = int(MAP_WIDTH // self.grid_size) + 1
        self.rows = int(MAP_HEIGHT // self.grid_size) + 1
        self.grid = [[0 for _ in range(self.rows)] for _ in range(self.cols)]
        self.update_obstacles(obstacles)

    def update_obstacles(self, obstacles):
        self.grid = [[0 for _ in range(self.rows)] for _ in range(self.cols)]
        inflation = AGENT_RADIUS + 2
        for obs in obstacles:
            inf_obs = obs.inflate(inflation*2, inflation*2)
            left = max(0, int(inf_obs.left // self.grid_size))
            right = min(self.cols - 1, int(inf_obs.right // self.grid_size))
            top = max(0, int(inf_obs.top // self.grid_size))
            bottom = min(self.rows - 1, int(inf_obs.bottom // self.grid_size))
            for c in range(left, right + 1):
                for r in range(top, bottom + 1):
                    self.grid[c][r] = 1

    def get_neighbors(self, node):
        c, r = node
        neighbors = []
        for dc in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if dc == 0 and dr == 0: continue
                nc, nr = c + dc, r + dr
                if 0 <= nc < self.cols and 0 <= nr < self.rows:
                    if self.grid[nc][nr] == 0:
                        # Prevent diagonal corner cutting through walls
                        if dc != 0 and dr != 0:
                            if self.grid[c+dc][r] == 1 or self.grid[c][r+dr] == 1:
                                continue
                        neighbors.append((nc, nr))
        return neighbors

    def line_of_sight(self, start, end):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if self.grid[x][y] == 1: return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if self.grid[x][y] == 1: return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        
        if self.grid[x][y] == 1: return False
        return True

    def reconstruct_path(self, came_from, start, goal):
        if goal not in came_from: return []
        current = goal
        path = []
        while current != start:
            px = current[0] * self.grid_size + self.grid_size / 2
            py = current[1] * self.grid_size + self.grid_size / 2
            path.append((px, py))
            current = came_from[current]
        
        px = start[0] * self.grid_size + self.grid_size / 2
        py = start[1] * self.grid_size + self.grid_size / 2
        path.append((px, py))
        path.reverse()
        return path

    def find_path(self, agent, end_pos, predecessor_paths=None):
        start_c = int(agent.pos.x // self.grid_size)
        start_r = int(agent.pos.y // self.grid_size)
        goal_c = int(end_pos[0] // self.grid_size)
        goal_r = int(end_pos[1] // self.grid_size)
        
        start_c = max(0, min(self.cols-1, start_c))
        start_r = max(0, min(self.rows-1, start_r))
        goal_c = max(0, min(self.cols-1, goal_c))
        goal_r = max(0, min(self.rows-1, goal_r))

        start = (start_c, start_r)
        goal = (goal_c, goal_r)
        
        if self.grid[goal[0]][goal[1]] == 1: return None
        if self.grid[start[0]][start[1]] == 1: return None

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: start}
        cost_so_far = {start: 0}

        def dist(a, b):
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
                
            for next_node in self.get_neighbors(current):
                parent = came_from[current]
                
                if self.line_of_sight(parent, next_node):
                    new_cost = cost_so_far[parent] + dist(parent, next_node)
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        heapq.heappush(frontier, (new_cost + dist(next_node, goal), next_node))
                        came_from[next_node] = parent
                else:
                    new_cost = cost_so_far[current] + dist(current, next_node)
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        heapq.heappush(frontier, (new_cost + dist(next_node, goal), next_node))
                        came_from[next_node] = current
                        
        path = self.reconstruct_path(came_from, start, goal)
        if path:
            path[0] = (agent.pos.x, agent.pos.y)
            path[-1] = (end_pos[0], end_pos[1])
            
            # Additional heuristic: If direct line of sight to final goal, skip intermediate points
            # to make paths look smoother and more optimal for ThetaStar
            final_smooth = [path[0]]
            curr_idx = 0
            while curr_idx < len(path)-1:
                for future_idx in range(len(path)-1, curr_idx, -1):
                    p1_c = int(path[curr_idx][0] // self.grid_size)
                    p1_r = int(path[curr_idx][1] // self.grid_size)
                    p2_c = int(path[future_idx][0] // self.grid_size)
                    p2_r = int(path[future_idx][1] // self.grid_size)
                    if self.line_of_sight((p1_c, p1_r), (p2_c, p2_r)):
                        final_smooth.append(path[future_idx])
                        curr_idx = future_idx
                        break
                if curr_idx == len(path)-1: 
                    break
            
            return final_smooth if len(final_smooth) > 1 else path
        return path
