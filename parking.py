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
        
    def park_agent(self, agent):
        """Mark an agent as permanently parked."""
        self.parked_count += 1
