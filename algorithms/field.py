# algorithms/field.py
# Electric-field update logic specific to the Electric Field planner

import pygame
import math
from config import *
from fundamental import clamp

def update_electric(agent, agents, obstacles, end_rect):
    if not agent.active: return
    agent._check_parking_logic(end_rect)
    
    if agent.spot_reserved:
        agent.update(end_rect)
        return

    if not agent.path_valid: 
        agent.resolve_collision(agents, obstacles); return

    if agent.current_wp_index < len(agent.path):
        target = pygame.Vector2(agent.path[agent.current_wp_index])
        desired = target - agent.pos
        attraction = desired.normalize() * K_ATTRACTION if desired.length() > 0 else pygame.Vector2(0,0)
    else: attraction = pygame.Vector2(0,0)

    repulsion = pygame.Vector2(0, 0)
    for other in agents:
        if other is agent: continue
        diff = agent.pos - other.pos
        d = diff.length()
        if 0.1 < d < 60: repulsion += diff.normalize() * (K_AGENT / (d**2))

    wall_repulsion = pygame.Vector2(0, 0)
    for obs in obstacles:
        cx = clamp(agent.pos.x, obs.left, obs.right)
        cy = clamp(agent.pos.y, obs.top, obs.bottom)
        diff = agent.pos - pygame.Vector2(cx, cy)
        d = diff.length()
        if 0.1 < d < 40: wall_repulsion += diff.normalize() * (K_WALL / (d**2))

    total = attraction + repulsion * 1.5 + wall_repulsion * 2.0
    if total.length() > MAX_FORCE: total = total.normalize() * MAX_FORCE
    
    agent.velocity += total
    if agent.velocity.length() > AGENT_SPEED: agent.velocity = agent.velocity.normalize() * AGENT_SPEED
    agent.pos += agent.velocity
    
    if agent.current_wp_index < len(agent.path):
        if agent.pos.distance_to(agent.path[agent.current_wp_index]) < AGENT_RADIUS * 1.5:
            agent.current_wp_index += 1
    
    agent.resolve_collision(agents, obstacles)
