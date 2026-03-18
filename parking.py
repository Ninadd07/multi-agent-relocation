# parking.py
# Simple grid parking: each cell is a parking spot.
# Agents move strictly dot-to-dot (4 cardinal directions), snapping discretely.

import pygame
import math
from collections import deque
from config import *
from fundamental import clamp


def get_grid_positions(center, count):
    positions = []
    cols = int(math.ceil(math.sqrt(count)))
    spacing = AGENT_DIAMETER + 5
    
    start_x = center[0] - ((cols * spacing) / 2) + spacing/2
    start_y = center[1] - ((cols * spacing) / 2) + spacing/2
    
    for i in range(count):
        row = i // cols
        col = i % cols
        x = start_x + (col * spacing)
        y = start_y + (row * spacing)
        positions.append((x, y))
    
    zone_size = cols * spacing + 10
    zone_rect = pygame.Rect(center[0] - zone_size/2, center[1] - zone_size/2, zone_size, zone_size)
    return positions, zone_rect


class SmartExit:
    """
    Continuous parking zone manager.
    Agents target the geometric center of this rectangle.
    """
    
    def __init__(self, rect):
        self.rect = rect
        self.center_pixel = pygame.Vector2(rect.centerx, rect.centery)
        self.parked_count = 0
        
    def check_entry(self, agent):
        """
        Gradually blend the agent from path-following into the parking funnel.
        
        - Outer blend zone (inflate 120): agent.spot_reserved = True, blend starts at 0.0
        - blend ramps up slowly (0.015/tick) until inside the commit zone
        - Commit zone (inflate 20): blend ramps faster (0.04/tick)
        - At blend == 1.0: A* path is cleared; agent is fully in funnel mode
        
        This avoids the instant "teleport snap" of the old threshold check.
        """
        if not agent.spot_reserved:
            blend_rect  = self.rect.inflate(0, 0)   # or just:  self.rect
            if blend_rect.collidepoint(agent.pos.x, agent.pos.y):
                agent.spot_reserved = True
                agent.parking_blend = 0.0  # 0 = path-following, 1 = full funnel

        if agent.spot_reserved and agent.active != False:
            commit_rect = self.rect
            in_commit = commit_rect.collidepoint(agent.pos.x, agent.pos.y)

            # Ramp blend faster once deep inside
            blend_speed = 0.04 if in_commit else 0.015
            agent.parking_blend = min(1.0, getattr(agent, 'parking_blend', 0.0) + blend_speed)

            # Only fully release A* path once fully committed
            if agent.parking_blend >= 1.0:
                agent.path = []
                agent.fluid_target_cell = None
                agent.current_grid_cell = None

    def update_agent(self, agent):
        """
        Continuous physics loop sliding the agent toward the parking center.
        
        When parking_blend < 1.0, the funnel pull is scaled by blend so the agent
        smoothly transitions from its existing velocity into the gravity well.
        """
        target = self.center_pixel
        diff = target - agent.pos
        dist = diff.length()
        
        # If we are basically mathematically dead-center, stop forever
        if dist < 2.0:
            agent.pos = pygame.Vector2(target)
            agent.velocity = pygame.Vector2(0, 0)
            agent.active = False
            self.park_agent(agent)
            return
        
        # Scale the pull force by parking_blend so early in the funnel it's gentle
        blend = getattr(agent, 'parking_blend', 1.0)
        pull_strength = AGENT_SPEED * 0.8 * blend  # Ramps from 0 → 80% of max speed

        pull_force = diff.normalize() * pull_strength
        
        # Apply friction/dampening so they don't bounce endlessly off each other
        agent.velocity = agent.velocity * 0.5 + pull_force * 0.5
        if agent.velocity.length() > AGENT_SPEED:
            agent.velocity.scale_to_length(AGENT_SPEED)
            
        agent.pos += agent.velocity
        
        # STEEP PYRAMID WALLS: Prevent popping back out of the zone
        agent.pos.x = clamp(agent.pos.x, self.rect.left + AGENT_RADIUS, self.rect.right - AGENT_RADIUS)
        agent.pos.y = clamp(agent.pos.y, self.rect.top + AGENT_RADIUS, self.rect.bottom - AGENT_RADIUS)
        
        # Have we stopped moving physically? (Because of physics crowding)
        moved_dist = (agent.pos - getattr(agent, 'prev_parking_pos', pygame.Vector2(0,0))).length()
        agent.prev_parking_pos = pygame.Vector2(agent.pos)
        
        if moved_dist < 0.05:
            agent.grid_patience += 1
            if agent.grid_patience > 60:  
                agent.velocity = pygame.Vector2(0, 0)
                agent.active = False
                self.park_agent(agent)
        else:
            agent.grid_patience = 0

    def resolve_collision(self, agent, other):
        """Hard-shell pyramidal overlapping resolver."""
        diff = agent.pos - other.pos
        dist = diff.length()
        min_dist = AGENT_DIAMETER + 0.5 
        
        if dist < min_dist:
            if dist == 0: correction = pygame.Vector2(1, 0)
            else:
                overlap = min_dist - dist
                correction = diff.normalize() * (overlap * 0.55)
                agent.pos += correction
                agent.velocity *= 0.5
                
        agent.pos.x = clamp(agent.pos.x, self.rect.left + AGENT_RADIUS, self.rect.right - AGENT_RADIUS)
        agent.pos.y = clamp(agent.pos.y, self.rect.top + AGENT_RADIUS, self.rect.bottom - AGENT_RADIUS)

    def park_agent(self, agent):
        """Mark an agent as permanently parked."""
        self.parked_count += 1