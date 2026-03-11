# config.py
import pygame

# --- DIMENSIONS ---
# Layout: [ Sidebar (200px) ] [ Left Map (650px) ] [ Right Map (650px) ]
PANEL_WIDTH = 200
MAP_WIDTH = 656
MAP_HEIGHT = 850

# Total Window Size
SCREEN_WIDTH = PANEL_WIDTH + (MAP_WIDTH * 2)
SCREEN_HEIGHT = MAP_HEIGHT
FPS = 60
NUM_AGENTS = 16

# --- COLORS ---
WHITE = (255, 255, 255)
OFF_WHITE = (240, 242, 245)
BLACK = (20, 20, 25)
GRAY_BG = (33, 37, 41)      # Sidebar Background
GRAY_HOVER = (52, 58, 64)
GRAY_ACTIVE = (73, 80, 87)

RED = (220, 53, 69)       
BLUE = (13, 110, 253)     
GREEN = (25, 135, 84)     
YELLOW = (255, 193, 7)    
ORANGE = (253, 126, 20)   
PARKED_GREEN = (100, 255, 100)

TEXT_WHITE = (248, 249, 250)
TEXT_GRAY = (173, 181, 189)

# --- AGENT SETTINGS ---
AGENT_RADIUS = 8
AGENT_DIAMETER = AGENT_RADIUS * 2
AGENT_SPEED = 2.0
VIEW_DISTANCE = 40      

# --- PHYSICS CONSTANTS ---
SPRING_K = 0.5 
DAMPING = 5.0  

# --- ALGORITHM CONSTANTS ---
# Multiplier for the penalty applied to path edges overlapping with higher-priority agents
PRIORITY_PENALTY_MULTIPLIER = 1.0
WIDE_CORNER_MAX_MARGIN = 30.0
K_ATTRACTION = 500.0
K_AGENT = 1000.0  
K_WALL = 1000.0   
MAX_FORCE = 0.2