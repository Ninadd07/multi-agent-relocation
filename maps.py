# maps.py
import pygame
import math
from config import *

def create_maps():
    maps = []
    
    obs1 = [pygame.Rect(300, 300, 200, 200)]
    maps.append({"obs": obs1, "start": (50, 400), "end": (750, 400), "name": "Center Block"})

    obs2 = [pygame.Rect(300, 0, 100, 350), pygame.Rect(300, 450, 100, 350)]
    maps.append({"obs": obs2, "start": (50, 400), "end": (750, 400), "name": "Corridor"})

    obs3 = [pygame.Rect(200, 0, 50, 500), pygame.Rect(550, 300, 50, 500)]
    maps.append({"obs": obs3, "start": (50, 100), "end": (750, 700), "name": "Zig Zag"})

    return maps

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