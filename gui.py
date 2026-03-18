# gui.py
print("--- Initializing MARS (Clean UI) ---")

import pygame
import sys
import time
import json
import os
import math
from PIL import Image
from config import *
from parking import SmartExit, get_grid_positions
from fundamental import Agent
from algorithms.field import update_electric
from algorithms.standard import GlobalPlanner
from algorithms.discrete_grid import Discretisation
from algorithms.thetastar import ThetaStarPlanner
from algorithms.obstacles import RectObstacle, CircleObstacle, FreehandObstacle

class SimState:
    def __init__(self, map_offset_x):
        self.map_offset_x = map_offset_x
        self.agents = []
        self.planner = None
        self.exit_manager = None
        self.all_parked = False
        self.path_error = False
        self.start_ticks = 0
        self.completion_time_ms = None
        self.use_electric = False
        self.recalc_index = 0
        self.elapsed_time_ms = 0.0
    
    def reset(self):
        self.agents = []
        self.all_parked = False
        self.path_error = False
        self.start_ticks = 0
        self.completion_time_ms = None
        self.elapsed_time_ms = 0.0
        self.recalc_index = 0

# --- UI CLASSES ---
class Button:
    def __init__(self, x, y, w, h, text, action_code, color=GRAY_BG, hover_color=GRAY_HOVER, toggle=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action_code = action_code
        self.base_color = color
        self.hover_color = hover_color
        self.toggle = toggle
        self.active = False
        self.hovered = False

    def draw(self, surface, font):
        color = self.base_color
        if self.active: color = BLUE
        elif self.hovered: color = self.hover_color
        
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        if not self.active:
            pygame.draw.rect(surface, (80, 80, 80), self.rect, 1, border_radius=5)
        
        txt_surf = font.render(self.text, True, TEXT_WHITE)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)

    def check_hover(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def check_click(self, mouse_pos):
        if self.hovered:
            if self.toggle: self.active = not self.active
            return self.action_code
        return None

class Dropdown:
    def __init__(self, x, y, w, h, options, default_index=0):
        self.rect = pygame.Rect(x, y, w, h)
        self.options = options
        self.selected_index = default_index
        self.is_open = False
        self.hovered_index = -1
        self.option_rects = [pygame.Rect(x, y + (i+1)*h, w, h) for i in range(len(options))]

    def draw(self, surface, font):
        # Draw Header
        pygame.draw.rect(surface, BLUE, self.rect, border_radius=5)
        pygame.draw.rect(surface, (100, 150, 255), self.rect, 2, border_radius=5)
        
        text = self.options[self.selected_index]
        # Truncate text if too long
        if len(text) > 18: text = text[:15] + "..."
        
        txt_surf = font.render(text, True, TEXT_WHITE)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)
        
        # Draw Arrow
        cx = self.rect.right - 15
        cy = self.rect.centery
        if self.is_open:
            pygame.draw.polygon(surface, TEXT_WHITE, [(cx - 4, cy + 2), (cx + 4, cy + 2), (cx, cy - 3)])
        else:
            pygame.draw.polygon(surface, TEXT_WHITE, [(cx - 4, cy - 2), (cx + 4, cy - 2), (cx, cy + 3)])

        # Draw Options
        if self.is_open:
            for i, rect in enumerate(self.option_rects):
                color = GRAY_HOVER if i == self.hovered_index else GRAY_BG
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (100, 100, 100), rect, 1)
                
                opt_txt = font.render(self.options[i], True, TEXT_WHITE)
                opt_rect = opt_txt.get_rect(center=rect.center)
                surface.blit(opt_txt, opt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            if self.is_open:
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(mouse_pos):
                        self.selected_index = i
                        self.is_open = False
                        return True
                self.is_open = False
            
            if self.rect.collidepoint(mouse_pos):
                self.is_open = not self.is_open
                return True
                
        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            self.hovered_index = -1
            if self.is_open:
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(mouse_pos):
                        self.hovered_index = i
        return False

# --- FILE OPERATIONS ---
def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory)

def save_map(obstacles, filename="mars_map.json"):
    ensure_dir("maps")
    filepath = os.path.join("maps", filename)
    data = [obs.to_dict() for obs in obstacles]
    try:
        with open(filepath, 'w') as f: json.dump(data, f)
        print(f"[OK] Map saved to {filepath}")
    except Exception as e: print(f"[ERROR] Error saving map: {e}")

def load_map(filename="mars_map.json"):
    filepath = os.path.join("maps", filename)
    if not os.path.exists(filepath):
        print(f"[WARNING] Map not found: {filepath}")
        return []
    try:
        with open(filepath, 'r') as f: data = json.load(f)
        # Backwards compatibility for old rect maps
        loaded = []
        for item in data:
            if isinstance(item, list):
                loaded.append(RectObstacle(item[0], item[1], item[2], item[3]))
            else:
                loaded.append(Obstacle.from_dict(item))
        return [o for o in loaded if o is not None]
    except Exception as e:
        print(f"[ERROR] Error loading map: {e}")
        return []

def save_gif(frames, filename="mars_sim.gif"):
    if not frames: 
        print("[ERROR] Save Failed: No frames.")
        return
    ensure_dir("gifs")
    filepath = os.path.join("gifs", filename)
    print(f"[SAVE] Saving {len(frames)} frames to {filepath}...")
    pil_images = []
    
    for surface in frames:
        str_data = pygame.image.tostring(surface, 'RGB')
        img = Image.frombytes('RGB', surface.get_size(), str_data)
        pil_images.append(img)

    try:
        pil_images[0].save(filepath, save_all=True, append_images=pil_images[1:], optimize=True, duration=33, loop=0)
        print(f"[OK] GIF Saved!")
    except Exception as e: print(f"[ERROR] Error saving GIF: {e}")

def draw_electric_field(surface, agents, obstacles, end_rect, map_offset_x):
    step = 40 
    zero_zone = end_rect.inflate(300, 300)
    field_surf = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), pygame.SRCALPHA)
    
    for x in range(20, MAP_WIDTH, step):
        for y in range(20, MAP_HEIGHT, step):
            pos = pygame.Vector2(x, y)
            if zero_zone.collidepoint(x, y): continue 
            
            force = pygame.Vector2(0, 0)
            for ag in agents:
                diff = pos - ag.pos
                dist = diff.length()
                if 1 < dist < 80: force += diff.normalize() * (K_AGENT / (dist**2))
            for obs in obstacles:
                cx = max(obs.left, min(pos.x, obs.right))
                cy = max(obs.top, min(pos.y, obs.bottom))
                diff = pos - pygame.Vector2(cx, cy)
                dist = diff.length()
                if 1 < dist < 60: force += diff.normalize() * (K_WALL / (dist**2))
            
            f_len = force.length()
            if f_len > 0.5:
                draw_len = min(f_len * 5, 20)
                end_pos = pos + force.normalize() * draw_len
                intensity = min(255, int(f_len * 50))
                color = (intensity, 100, 255 - intensity, 150)
                pygame.draw.line(field_surf, color, pos, end_pos, 1)
                pygame.draw.circle(field_surf, color, (int(pos.x), int(pos.y)), 1)
    surface.blit(field_surf, (map_offset_x, 0))

def draw_exit_grid(surface, exit_manager, map_offset_x):
    if not exit_manager: return
    # The parking zone is now continuous. We only draw its geometric center point.
    center_pos = exit_manager.center_pixel
    draw_pos = (int(center_pos.x + map_offset_x), int(center_pos.y))
    pygame.draw.circle(surface, (255, 180, 50), draw_pos, 4)
    pygame.draw.circle(surface, (255, 255, 255), draw_pos, 2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("MARS: Sim Control")
    clock = pygame.time.Clock()
    
    # MacOS Retina displays can blow up unadjusted System fonts.
    # We use smaller absolute values for the font heights.
    font_ui = pygame.font.SysFont("Helvetica", 11)
    font_bold = pygame.font.SysFont("Helvetica", 11, bold=True)
    font_tiny = pygame.font.SysFont("Helvetica", 9, bold=True)
    font_timer = pygame.font.SysFont("Menlo", 28, bold=True)
    font_header = pygame.font.SysFont("Helvetica", 13, bold=True) 
    font_label = pygame.font.SysFont("Helvetica", 11, bold=True)
    error_font = pygame.font.SysFont("Helvetica", 20, bold=True)

    bx = 15; bw = 170; by = 155
    
    # --- UI LAYOUT ---
    buttons = [
        Button(bx, by, 80, 28, "Restart", "RESTART", color=RED),
        Button(bx+90, by, 80, 28, "Pause", "PAUSE_RESUME", color=ORANGE),
        
        Button(bx, by+35, 40, 25, "1x", "SPD_1"),
        Button(bx+45, by+35, 40, 25, "2x", "SPD_2"),
        Button(bx+90, by+35, 40, 25, "3x", "SPD_3"),
        Button(bx+135, by+35, 40, 25, "4x", "SPD_4"),
    ]
    
    algo_left_y = by + 100
    algo_right_y = algo_left_y + 60
    algo_options = ["Electric Field", "Standard Path", "Standard (Penalized)", "TBC", "Discrete Grid", "Theta*"]
    algo_dropdown_left = Dropdown(bx, algo_left_y, bw, 28, algo_options, default_index=1)
    algo_dropdown_right = Dropdown(bx, algo_right_y, bw, 28, algo_options, default_index=5)
    
    def _get_planner_mode(idx):
        """Map dropdown index to GlobalPlanner mode string."""
        return {1: 'standard', 2: 'penalized', 3: 'tbc'}.get(idx, 'standard')
    
    grid_y = algo_right_y + 40
    btn_show_grid = Button(bx, grid_y, bw, 28, "Show Grid", "TOGGLE_GRID", toggle=True)
    buttons.append(btn_show_grid)

    file_y = grid_y + 45
    buttons += [
        Button(bx, file_y, 80, 28, "Save Map", "SAVE_MAP", color=BLUE),
        Button(bx+90, file_y, 80, 28, "Load Map", "LOAD_MAP", color=BLUE),
        Button(bx, file_y+35, bw, 28, "Clear Walls", "CLEAR_WALLS", color=ORANGE),
    ]
    
    tools_y = file_y + 75
    btn_tool_rect = Button(bx, tools_y, 50, 28, "Rect", "TOOL_RECT", toggle=True, color=(100, 100, 100))
    btn_tool_circ = Button(bx+60, tools_y, 50, 28, "Circ", "TOOL_CIRC", toggle=True, color=(100, 100, 100))
    btn_tool_draw = Button(bx+120, tools_y, 50, 28, "Draw", "TOOL_DRAW", toggle=True, color=(100, 100, 100))
    btn_tool_rect.active = True
    buttons += [btn_tool_rect, btn_tool_circ, btn_tool_draw]
    
    rec_y = tools_y + 45
    buttons += [
        Button(bx, rec_y, 80, 30, "REC", "TOGGLE_REC", color=(200, 50, 50), toggle=True),
        Button(bx+90, rec_y, 80, 30, "Export", "SAVE_GIF", color=GREEN)
    ]
    
    buttons[2].active = True # 1x default

    # --- STATE ---
    state = "DRAWING"
    sim_speed = 1
    is_recording = False
    recorded_frames = []
    
    start_time = 0
    paused_time_accumulator = 0
    last_pause_start = 0
    
    timer_running = False
    input_blocked = False
    frame_counter = 0

    custom_obstacles = []
    current_drawing_rect = None
    current_drawing_circ = None  # (cx, cy, current_r)
    current_drawing_freehand = [] # [(x, y), ...]
    active_tool = "TOOL_RECT"
    
    sim_left = SimState(PANEL_WIDTH)
    sim_right = SimState(PANEL_WIDTH + MAP_WIDTH)
    
    start_center = (100, 100)
    end_center = (MAP_WIDTH - 150, MAP_HEIGHT - 150)
    spawn_positions, start_rect = get_grid_positions(start_center, NUM_AGENTS)
    parking_positions, end_rect = get_grid_positions(end_center, NUM_AGENTS)

    last_tick_time = pygame.time.get_ticks()
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        dt = current_time - last_tick_time
        last_tick_time = current_time
        
        mouse_pos = pygame.mouse.get_pos()
        map_mouse_pos = (mouse_pos[0] - PANEL_WIDTH, mouse_pos[1])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            # 1. Dropdown (High Priority)
            dropdown_consumed = False
            for sim, dropdown in [(sim_left, algo_dropdown_left), (sim_right, algo_dropdown_right)]:
                prev_algo = dropdown.selected_index
                if dropdown.handle_event(event):
                    dropdown_consumed = True
                    if dropdown.selected_index != prev_algo and state in ["RUNNING", "PAUSED"]:
                        if dropdown.selected_index == 4:
                            sim.planner = Discretisation(custom_obstacles)
                        else:
                            sim.planner = GlobalPlanner(custom_obstacles, mode=_get_planner_mode(dropdown.selected_index))
                        
                        if sim.agents:
                            for agent in sim.agents:
                                agent.planner = sim.planner
                                agent.recalc_path()

            # 2. Buttons
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not dropdown_consumed:
                if mouse_pos[0] < PANEL_WIDTH:
                    for btn in buttons:
                        action = btn.check_click(mouse_pos)
                        if action:
                            if action == "RESTART":
                                state = "DRAWING"; is_recording = False; recorded_frames = []
                                timer_running = False; frame_counter = 0
                                sim_left.reset(); sim_right.reset()
                                buttons[1].text = "Pause"; buttons[1].base_color = ORANGE
                                
                            elif action == "PAUSE_RESUME":
                                if state == "RUNNING":
                                    state = "PAUSED"; btn.text = "Resume"; btn.base_color = GREEN
                                    last_pause_start = pygame.time.get_ticks(); timer_running = False
                                elif state == "PAUSED":
                                    state = "RUNNING"; btn.text = "Pause"; btn.base_color = ORANGE
                                    paused_time_accumulator += (pygame.time.get_ticks() - last_pause_start)
                                    timer_running = True
                                    
                            elif action == "SPD_1": sim_speed = 1; buttons[2].active=True; buttons[3].active=False; buttons[4].active=False; buttons[5].active=False
                            elif action == "SPD_2": sim_speed = 2; buttons[2].active=False; buttons[3].active=True; buttons[4].active=False; buttons[5].active=False
                            elif action == "SPD_3": sim_speed = 3; buttons[2].active=False; buttons[3].active=False; buttons[4].active=True; buttons[5].active=False
                            elif action == "SPD_4": sim_speed = 4; buttons[2].active=False; buttons[3].active=False; buttons[4].active=False; buttons[5].active=True
                            elif action == "TOGGLE_REC": is_recording = btn.active
                            elif action == "SAVE_GIF": save_gif(recorded_frames)
                            elif action == "SAVE_MAP": save_map(custom_obstacles)
                            elif action == "LOAD_MAP":
                                loaded = load_map()
                                if loaded: 
                                    custom_obstacles = loaded
                                    if state in ["RUNNING", "PAUSED"]:
                                        for sim in (sim_left, sim_right):
                                            if sim.planner:
                                                sim.planner.update_obstacles(custom_obstacles)
                                                for agent in sim.agents: agent.recalc_path()
                            elif action == "CLEAR_WALLS":
                                custom_obstacles = []
                                if state in ["RUNNING", "PAUSED"]:
                                    for sim in (sim_left, sim_right):
                                        if sim.planner:
                                            sim.planner.update_obstacles([])
                                            for agent in sim.agents: agent.recalc_path()
                
                            elif action in ["TOOL_RECT", "TOOL_CIRC", "TOOL_DRAW"]:
                                active_tool = action
                                btn_tool_rect.active = (action == "TOOL_RECT")
                                btn_tool_circ.active = (action == "TOOL_CIRC")
                                btn_tool_draw.active = (action == "TOOL_DRAW")
                
                elif map_mouse_pos[0] >= 0:
                    if active_tool == "TOOL_RECT":
                        current_drawing_rect = [map_mouse_pos[0], map_mouse_pos[1], 0, 0]
                    elif active_tool == "TOOL_CIRC":
                        current_drawing_circ = [map_mouse_pos[0], map_mouse_pos[1], 0]
                    elif active_tool == "TOOL_DRAW":
                        current_drawing_freehand = [map_mouse_pos]

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if map_mouse_pos[0] >= 0:
                    to_remove = None
                    for obs in reversed(custom_obstacles):
                        if obs.collidepoint(map_mouse_pos): to_remove = obs; break
                    if to_remove: 
                        custom_obstacles.remove(to_remove)
                        if state in ["RUNNING", "PAUSED"]:
                            for sim in (sim_left, sim_right):
                                if sim.planner:
                                    sim.planner.update_obstacles(custom_obstacles)
                                    for agent in sim.agents: agent.recalc_path()

            if event.type == pygame.MOUSEMOTION:
                for btn in buttons: btn.check_hover(mouse_pos)
                if current_drawing_rect:
                    current_drawing_rect[2] = map_mouse_pos[0] - current_drawing_rect[0]
                    current_drawing_rect[3] = map_mouse_pos[1] - current_drawing_rect[1]
                if current_drawing_circ:
                    r = math.hypot(map_mouse_pos[0] - current_drawing_circ[0], map_mouse_pos[1] - current_drawing_circ[1])
                    current_drawing_circ[2] = r
                if current_drawing_freehand is not None and len(current_drawing_freehand) > 0 and pygame.mouse.get_pressed()[0]:
                    last_point = current_drawing_freehand[-1]
                    if math.hypot(last_point[0] - map_mouse_pos[0], last_point[1] - map_mouse_pos[1]) > 5:
                        current_drawing_freehand.append(map_mouse_pos)


            if event.type == pygame.MOUSEBUTTONUP:
                input_blocked = False
                
                def _check_valid_obs(obs):
                    valid = True
                    # Check agents
                    if sim_left.agents or sim_right.agents:
                        for sim in (sim_left, sim_right):
                            for agent in sim.agents:
                                a_rect = pygame.Rect(agent.pos.x - AGENT_RADIUS, agent.pos.y - AGENT_RADIUS, AGENT_DIAMETER, AGENT_DIAMETER)
                                # Adjust obstacle by translating to sim panel
                                moved_obs = obs.move(-sim.map_offset_x + PANEL_WIDTH, 0)
                                if moved_obs.colliderect(a_rect): 
                                    return False
                    # Check starts/ends
                    if obs.colliderect(start_rect) or obs.colliderect(end_rect):
                        return False
                    return True
                
                new_obs = None
                if current_drawing_rect:
                    r = pygame.Rect(current_drawing_rect)
                    r.normalize()
                    if r.width > 5 and r.height > 5:
                        new_obs = RectObstacle(r.x, r.y, r.width, r.height)
                elif current_drawing_circ:
                    if current_drawing_circ[2] > 5:
                        new_obs = CircleObstacle(current_drawing_circ[0], current_drawing_circ[1], current_drawing_circ[2])
                elif current_drawing_freehand is not None and len(current_drawing_freehand) > 1:
                    new_obs = FreehandObstacle(list(current_drawing_freehand), thickness=10)

                current_drawing_rect = None
                current_drawing_circ = None
                current_drawing_freehand = []

                if new_obs:
                    if not _check_valid_obs(new_obs):
                        input_blocked = True
                    else:
                        custom_obstacles.append(new_obs)
                        if state in ["RUNNING", "PAUSED"]:
                            for sim in (sim_left, sim_right):
                                if sim.planner:
                                    sim.planner.update_obstacles(custom_obstacles)
                                    for agent in sim.agents: agent.recalc_path()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
                if event.key == pygame.K_RETURN and state == "DRAWING":
                    print("Initializing Dual Simulation...")
                    for sim, dropdown in [(sim_left, algo_dropdown_left), (sim_right, algo_dropdown_right)]:
                        if dropdown.selected_index == 4:
                            sim.planner = Discretisation(custom_obstacles)
                        elif dropdown.selected_index == 5:
                            sim.planner = ThetaStarPlanner(custom_obstacles)
                        else:
                            sim.planner = GlobalPlanner(custom_obstacles, mode=_get_planner_mode(dropdown.selected_index))
                        sim.exit_manager = SmartExit(end_rect)
                        sim.agents = []
                        predecessor_paths = []
                        
                        # Sort spawn positions by distance to goal (closest = highest priority)
                        indexed_spawns = list(enumerate(spawn_positions))
                        indexed_spawns.sort(key=lambda x: math.hypot(x[1][0] - end_center[0], x[1][1] - end_center[1]))
                        
                        for priority, (orig_idx, pos) in enumerate(indexed_spawns):
                            agent = Agent(pos, sim.exit_manager, sim.planner,
                                          predecessor_paths=list(predecessor_paths))
                            agent.index = priority + 1  # 1 = closest to goal = highest priority
                            sim.agents.append(agent)
                            if agent.path:
                                predecessor_paths.append(list(agent.path))
                                
                    state = "RUNNING"
                    start_time = pygame.time.get_ticks()
                    paused_time_accumulator = 0
                    timer_running = True
                    last_tick_time = start_time
                    dt = 0
                    for sim in (sim_left, sim_right):
                        sim.start_ticks = start_time
                        sim.elapsed_time_ms = 0.0
                    buttons[1].text = "Pause"; buttons[1].base_color = ORANGE
                
                if event.key == pygame.K_r: 
                    state = "DRAWING"; timer_running = False
                    sim_left.reset(); sim_right.reset()
                    buttons[1].text = "Pause"; buttons[1].base_color = ORANGE
                
                if event.key == pygame.K_p: 
                    if state == "RUNNING":
                        state = "PAUSED"; buttons[1].text = "Resume"; buttons[1].base_color = GREEN
                        last_pause_start = pygame.time.get_ticks(); timer_running = False
                    elif state == "PAUSED":
                        state = "RUNNING"; buttons[1].text = "Pause"; buttons[1].base_color = ORANGE
                        paused_time_accumulator += (pygame.time.get_ticks() - last_pause_start)
                        timer_running = True

        if state == "RUNNING":
            for sim in (sim_left, sim_right):
                if not sim.all_parked:
                    sim.elapsed_time_ms += dt * sim_speed

            for sim, dropdown in [(sim_left, algo_dropdown_left), (sim_right, algo_dropdown_right)]:
                sim.use_electric = (dropdown.selected_index == 0)
                if isinstance(sim.planner, Discretisation): sim.planner.set_agents(sim.agents)

            for _ in range(sim_speed):
                for sim in (sim_left, sim_right):
                    if sim.all_parked: continue # Map finished

                    if not sim.use_electric:
                        for agent in sim.agents: agent.local_safety_check(sim.agents)

                    sim.all_parked = True
                    sim.path_error = False
                    for agent in sim.agents:
                        if sim.use_electric: 
                            update_electric(agent, sim.agents, custom_obstacles, end_rect)
                        else: 
                            agent.update(end_rect)
                        
                        if agent.active: sim.all_parked = False
                        if not agent.path_valid: sim.path_error = True
                    
                    for _ in range(4):
                        for agent in sim.agents:
                            if not agent.dfs_settled:  # Allow agents navigating the grid to push past each other
                                agent.resolve_collision(sim.agents, custom_obstacles)

                    # Timer stops when ALL agents have entered the grid
                    all_entered = all(agent.spot_reserved for agent in sim.agents) if sim.agents else False
                    if all_entered and sim.completion_time_ms is None:
                        sim.completion_time_ms = sim.elapsed_time_ms
                        for agent in sim.agents:
                            agent.active = False
                            agent.velocity = pygame.math.Vector2(0, 0)

            all_entered_both = (
                (all(a.spot_reserved for a in sim_left.agents) if sim_left.agents else False) and
                (all(a.spot_reserved for a in sim_right.agents) if sim_right.agents else False)
            )
            if all_entered_both:
                timer_running = False

        # --- DRAWING ---
        screen.fill(OFF_WHITE)
        pygame.draw.rect(screen, GRAY_BG, (0, 0, PANEL_WIDTH, SCREEN_HEIGHT))
        pygame.draw.line(screen, (60,60,60), (PANEL_WIDTH, 0), (PANEL_WIDTH, SCREEN_HEIGHT), 2)
        
        # --- SIDEBAR CONTENT ---
        status_txt = state
        if not timer_running and state == "RUNNING": status_txt = "FINISHED"
        screen.blit(font_bold.render(f"STATUS: {status_txt}", True, TEXT_GRAY), (bx, 30))
        
        info_y = 65
        if state == "DRAWING":
            screen.blit(font_header.render("INSTRUCTIONS", True, TEXT_WHITE), (bx, info_y))
            screen.blit(font_ui.render("L-Click: Draw Wall", True, TEXT_GRAY), (bx, info_y + 20))
            screen.blit(font_ui.render("R-Click: Erase Wall", True, TEXT_GRAY), (bx, info_y + 35))
            screen.blit(font_bold.render("ENTER: Start Sim", True, GREEN), (bx, info_y + 60))
        else:
            screen.blit(font_header.render("LIVE EDITING", True, TEXT_WHITE), (bx, info_y))
            screen.blit(font_ui.render("Draw/Erase walls to", True, TEXT_GRAY), (bx, info_y + 20))
            screen.blit(font_ui.render("force re-routing!", True, TEXT_GRAY), (bx, info_y + 35))

        # 2. Controls
        for btn in buttons: btn.draw(screen, font_ui)
        
        # 3. Algorithm Labels (Dropdowns drawn later)
        screen.blit(font_label.render("LEFT MAP", True, TEXT_WHITE), (bx, algo_left_y - 15))
        screen.blit(font_label.render("RIGHT MAP", True, TEXT_WHITE), (bx, algo_right_y - 15))

        # --- MAP AREA ---
        for sim, dropdown in [(sim_left, algo_dropdown_left), (sim_right, algo_dropdown_right)]:
            map_clip = pygame.Rect(sim.map_offset_x, 0, MAP_WIDTH, MAP_HEIGHT)
            screen.set_clip(map_clip)
            
            # Map separator
            if sim == sim_right:
                pygame.draw.line(screen, (80,80,80), (sim.map_offset_x, 0), (sim.map_offset_x, SCREEN_HEIGHT), 2)
            
            s_rect = start_rect.move(sim.map_offset_x, 0)
            e_rect = end_rect.move(sim.map_offset_x, 0)
            pygame.draw.rect(screen, GREEN, s_rect, 2)
            pygame.draw.rect(screen, GREEN, e_rect, 2)
            
            lbl_s = font_label.render("START", True, GREEN)
            lbl_e = font_label.render("END", True, GREEN)
            screen.blit(lbl_s, (s_rect.x, s_rect.y - 18))
            screen.blit(lbl_e, (e_rect.x, e_rect.y - 18))

            # Draw Map Timer
            elapsed = sim.elapsed_time_ms
            if sim.completion_time_ms is not None:
                elapsed = sim.completion_time_ms
            
            time_str = f"{elapsed/1000:.2f}s"
            timer_color = GREEN if sim.all_parked else TEXT_WHITE
            timer_surf = font_timer.render(time_str, True, timer_color)
            
            # Subtle background pill
            tx = sim.map_offset_x + (MAP_WIDTH // 2) - (timer_surf.get_width() // 2)
            ty = 20
            bg_rect = timer_surf.get_rect(topleft=(tx, ty)).inflate(20, 10)
            pygame.draw.rect(screen, (30, 30, 30), bg_rect, border_radius=5)
            pygame.draw.rect(screen, (80, 80, 80), bg_rect, 1, border_radius=5)
            screen.blit(timer_surf, (tx, ty))

            # Draw exit grid first, regardless of running state
            fake_exit_manager = SmartExit(end_rect) if sim.exit_manager is None else sim.exit_manager
            draw_exit_grid(screen, fake_exit_manager, sim.map_offset_x)

            for obs in custom_obstacles:
                obs.draw(screen, BLUE, sim.map_offset_x, 0)
                
            # Draw Discretisation Grid
            if btn_show_grid.active:
                viz_planner = sim.planner
                if (viz_planner is None or not isinstance(viz_planner, Discretisation)) and dropdown.selected_index == 4:
                    viz_planner = Discretisation(custom_obstacles)

                if isinstance(viz_planner, Discretisation):
                    grid_cells = viz_planner.build_occupancy_grid(AGENT_RADIUS)
                    grid_surf = pygame.Surface((MAP_WIDTH, MAP_HEIGHT), pygame.SRCALPHA)
                    for (x, y, w, h, is_free) in grid_cells:
                        r = pygame.Rect(x, y, w, h)
                        if is_free:
                            pygame.draw.rect(grid_surf, (0, 255, 0, 50), r, 1)
                        else:
                            pygame.draw.rect(grid_surf, (255, 0, 0, 100), r)
                            pygame.draw.rect(grid_surf, (255, 0, 0), r, 1)
                    screen.blit(grid_surf, (sim.map_offset_x, 0))
            
            if current_drawing_rect:
                preview = [current_drawing_rect[0] + sim.map_offset_x, current_drawing_rect[1], current_drawing_rect[2], current_drawing_rect[3]]
                temp_rect = pygame.Rect(preview)
                temp_rect.normalize()
                color = RED if input_blocked else (100, 100, 255)
                pygame.draw.rect(screen, color, temp_rect, 2)
            if current_drawing_circ:
                cx = current_drawing_circ[0] + sim.map_offset_x
                cy = current_drawing_circ[1]
                r = current_drawing_circ[2]
                color = RED if input_blocked else (100, 100, 255)
                pygame.draw.circle(screen, color, (int(cx), int(cy)), int(r), 2)
            if current_drawing_freehand is not None and len(current_drawing_freehand) > 1:
                pts = [(p[0] + sim.map_offset_x, p[1]) for p in current_drawing_freehand]
                color = RED if input_blocked else (100, 100, 255)
                pygame.draw.lines(screen, color, False, pts, 4)

            if state in ["RUNNING", "PAUSED"]:
                if sim.use_electric: 
                    draw_electric_field(screen, sim.agents, custom_obstacles, end_rect, sim.map_offset_x)
                
                for agent in sim.agents:
                    draw_pos = (int(agent.pos.x + sim.map_offset_x), int(agent.pos.y))
                    if len(agent.path) > 1 and agent.active:
                        future_waypoints = agent.path[agent.current_wp_index:]
                        offset_start = (agent.pos.x + sim.map_offset_x, agent.pos.y)
                        display_points = [offset_start] + [(p[0] + sim.map_offset_x, p[1]) for p in future_waypoints]
                        if len(display_points) > 1:
                            pygame.draw.lines(screen, YELLOW, False, display_points, 1)
                    pygame.draw.circle(screen, agent.get_color(), draw_pos, AGENT_RADIUS)
                    pygame.draw.circle(screen, BLACK, draw_pos, AGENT_RADIUS, 1)
                    # Draw agent number
                    if hasattr(agent, 'index'):
                        num_surf = font_tiny.render(str(agent.index), True, WHITE)
                        screen.blit(num_surf, (draw_pos[0] - num_surf.get_width()//2, draw_pos[1] - num_surf.get_height()//2))
                
                if sim.path_error:
                    box_rect = pygame.Rect(sim.map_offset_x + MAP_WIDTH//2 - 200, MAP_HEIGHT//2 - 50, 400, 100)
                    pygame.draw.rect(screen, (50, 50, 50), box_rect)
                    pygame.draw.rect(screen, RED, box_rect, 3)
                    txt1 = error_font.render("NO PATH FOUND!", True, RED)
                    screen.blit(txt1, (box_rect.centerx - txt1.get_width()//2, box_rect.centery - 15))

        screen.set_clip(None)
        
        # Draw lowest vertical elements first so upper ones render ON TOP.
        algo_dropdown_right.draw(screen, font_ui)
        algo_dropdown_left.draw(screen, font_ui)

        if is_recording and state == "RUNNING":
            pygame.draw.circle(screen, RED, (SCREEN_WIDTH - 30, 30), 8)
            if frame_counter % 2 == 0: recorded_frames.append(screen.copy())
            frame_counter += 1

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()