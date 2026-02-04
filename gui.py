# gui.py
print("--- Initializing MARS (Clean UI) ---")

import pygame
import sys
import time
import json
import os
from PIL import Image
from config import *
from maps import get_grid_positions
from algorithms import Agent, GlobalPlanner, SmartExit

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
        
        arrow = "▼" if not self.is_open else "▲"
        text = self.options[self.selected_index]
        # Truncate text if too long
        if len(text) > 18: text = text[:15] + "..."
        
        txt_surf = font.render(f"{text} {arrow}", True, TEXT_WHITE)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)

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
    data = [[r.x, r.y, r.width, r.height] for r in obstacles]
    try:
        with open(filepath, 'w') as f: json.dump(data, f)
        print(f"✅ Map saved to {filepath}")
    except Exception as e: print(f"❌ Error saving map: {e}")

def load_map(filename="mars_map.json"):
    filepath = os.path.join("maps", filename)
    if not os.path.exists(filepath):
        print(f"⚠️ Map not found: {filepath}")
        return []
    try:
        with open(filepath, 'r') as f: data = json.load(f)
        return [pygame.Rect(item[0], item[1], item[2], item[3]) for item in data]
    except Exception as e:
        print(f"❌ Error loading map: {e}")
        return []

def save_gif(frames, filename="mars_sim.gif"):
    if not frames: 
        print("❌ Save Failed: No frames.")
        return
    ensure_dir("gifs")
    filepath = os.path.join("gifs", filename)
    print(f"💾 Saving {len(frames)} frames to {filepath}...")
    pil_images = []
    
    for surface in frames:
        str_data = pygame.image.tostring(surface, 'RGB')
        img = Image.frombytes('RGB', surface.get_size(), str_data)
        pil_images.append(img)

    try:
        pil_images[0].save(filepath, save_all=True, append_images=pil_images[1:], optimize=True, duration=33, loop=0)
        print(f"✅ GIF Saved!")
    except Exception as e: print(f"❌ Error saving GIF: {e}")

def draw_electric_field(surface, agents, obstacles, end_rect):
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
    surface.blit(field_surf, (PANEL_WIDTH, 0))

def draw_exit_grid(surface, exit_manager):
    if not exit_manager: return
    for r in range(exit_manager.rows):
        for c in range(exit_manager.cols):
            pos = exit_manager.get_pixel_center(r, c)
            draw_pos = (int(pos.x + PANEL_WIDTH), int(pos.y))
            
            is_available = (r, c) in exit_manager.parking_queue
            if is_available:
                pygame.draw.circle(surface, (120, 120, 120), draw_pos, 2)
            else:
                pygame.draw.circle(surface, (255, 50, 50), draw_pos, 4)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("MARS: Sim Control")
    clock = pygame.time.Clock()
    
    font_ui = pygame.font.SysFont("Segoe UI", 16)
    font_bold = pygame.font.SysFont("Segoe UI", 16, bold=True)
    font_timer = pygame.font.SysFont("Consolas", 32, bold=True)
    font_header = pygame.font.SysFont("Segoe UI", 20, bold=True) # Slightly smaller header
    font_label = pygame.font.SysFont("Arial", 14, bold=True)
    error_font = pygame.font.SysFont("Arial", 30, bold=True)

    bx = 20; bw = 160; by = 260
    
    # --- UI LAYOUT ---
    # Group 1: Time Controls (Tighter spacing)
    buttons = [
        Button(bx, by, 75, 35, "Restart", "RESTART", color=RED),
        Button(bx+85, by, 75, 35, "Pause", "PAUSE_RESUME", color=ORANGE),
        
        # Speed Controls immediately below
        Button(bx, by+45, 35, 30, "1x", "SPD_1"),
        Button(bx+40, by+45, 35, 30, "2x", "SPD_2"),
        Button(bx+80, by+45, 35, 30, "3x", "SPD_3"),
        Button(bx+120, by+45, 35, 30, "4x", "SPD_4"),
    ]
    
    # Group 2: Algorithms (Dropdown handled separately)
    # y = by + 130
    
    # Group 3: File Ops (Push down)
    file_y = by + 220
    buttons += [
        Button(bx, file_y, 75, 30, "Save Map", "SAVE_MAP", color=BLUE),
        Button(bx+85, file_y, 75, 30, "Load Map", "LOAD_MAP", color=BLUE),
        Button(bx, file_y+40, bw, 30, "Clear Walls", "CLEAR_WALLS", color=ORANGE),
    ]
    
    # Group 4: Recording (Bottom)
    rec_y = file_y + 100
    buttons += [
        Button(bx, rec_y, 75, 35, "REC", "TOGGLE_REC", color=(200, 50, 50), toggle=True),
        Button(bx+85, rec_y, 75, 35, "Export", "SAVE_GIF", color=GREEN)
    ]
    
    buttons[2].active = True # 1x default
    
    # Dropdown for Algorithm Selection (Positioned in the middle gap)
    algo_options = ["Electric Field", "Standard Path"]
    algo_dropdown = Dropdown(bx, by + 130, bw, 35, algo_options, default_index=0)
    
    # --- STATE ---
    state = "DRAWING"
    sim_speed = 1
    is_recording = False
    recorded_frames = []
    
    start_time = 0
    final_time = 0
    paused_time_accumulator = 0
    last_pause_start = 0
    
    timer_running = False
    input_blocked = False
    frame_counter = 0

    custom_obstacles = []
    current_drawing_rect = None
    agents = []
    planner = None
    exit_manager = None
    
    start_center = (100, 100)
    end_center = (700, 600)
    spawn_positions, start_rect = get_grid_positions(start_center, NUM_AGENTS)
    parking_positions, end_rect = get_grid_positions(end_center, NUM_AGENTS)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        map_mouse_pos = (mouse_pos[0] - PANEL_WIDTH, mouse_pos[1])
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            # 1. Dropdown (High Priority)
            if algo_dropdown.handle_event(event):
                continue 

            # 2. Buttons
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if mouse_pos[0] < PANEL_WIDTH:
                    for btn in buttons:
                        action = btn.check_click(mouse_pos)
                        if action:
                            if action == "RESTART":
                                state = "DRAWING"; agents = []; is_recording = False; recorded_frames = []
                                timer_running = False; final_time = 0; frame_counter = 0
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
                                    if state in ["RUNNING", "PAUSED"] and planner:
                                        planner.update_obstacles(custom_obstacles)
                                        for agent in agents: agent.recalc_path()
                            elif action == "CLEAR_WALLS":
                                custom_obstacles = []
                                if state in ["RUNNING", "PAUSED"] and planner:
                                    planner.update_obstacles([])
                                    for agent in agents: agent.recalc_path()
                
                elif map_mouse_pos[0] >= 0:
                    current_drawing_rect = [map_mouse_pos[0], map_mouse_pos[1], 0, 0]

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if map_mouse_pos[0] >= 0:
                    to_remove = None
                    for obs in reversed(custom_obstacles):
                        if obs.collidepoint(map_mouse_pos): to_remove = obs; break
                    if to_remove: 
                        custom_obstacles.remove(to_remove)
                        if state in ["RUNNING", "PAUSED"] and planner:
                            planner.update_obstacles(custom_obstacles)
                            for agent in agents: agent.recalc_path()

            if event.type == pygame.MOUSEMOTION:
                for btn in buttons: btn.check_hover(mouse_pos)
                if current_drawing_rect:
                    current_drawing_rect[2] = map_mouse_pos[0] - current_drawing_rect[0]
                    current_drawing_rect[3] = map_mouse_pos[1] - current_drawing_rect[1]

            if event.type == pygame.MOUSEBUTTONUP:
                input_blocked = False
                if current_drawing_rect:
                    r = pygame.Rect(current_drawing_rect)
                    r.normalize()
                    if r.width > 5 and r.height > 5:
                        valid_placement = True
                        if agents:
                            for agent in agents:
                                a_rect = pygame.Rect(agent.pos.x - AGENT_RADIUS, agent.pos.y - AGENT_RADIUS, AGENT_DIAMETER, AGENT_DIAMETER)
                                if r.colliderect(a_rect): valid_placement = False; input_blocked = True; break
                        if valid_placement:
                            if r.colliderect(start_rect) or r.colliderect(end_rect):
                                valid_placement = False; input_blocked = True
                        if valid_placement:
                            custom_obstacles.append(r)
                            if state in ["RUNNING", "PAUSED"] and planner:
                                planner.update_obstacles(custom_obstacles)
                                for agent in agents: agent.recalc_path()
                    current_drawing_rect = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and state == "DRAWING":
                    print("Initializing...")
                    planner = GlobalPlanner(custom_obstacles)
                    exit_manager = SmartExit(end_rect)
                    agents = []
                    for i in range(NUM_AGENTS):
                        agents.append(Agent(spawn_positions[i], exit_manager, planner))
                    state = "RUNNING"
                    start_time = pygame.time.get_ticks()
                    paused_time_accumulator = 0
                    timer_running = True
                    buttons[1].text = "Pause"; buttons[1].base_color = ORANGE
                
                if event.key == pygame.K_r: 
                    state = "DRAWING"; agents = []; timer_running = False
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
            if exit_manager: exit_manager.register_agents(agents)
            use_electric = (algo_dropdown.selected_index == 0)

            for _ in range(sim_speed):
                if not use_electric:
                    for agent in agents: agent.local_safety_check(agents)
                
                all_parked = True
                path_error = False
                for agent in agents:
                    if use_electric: 
                        agent.update_electric(agents, custom_obstacles, end_rect)
                    else: 
                        agent.update(end_rect)
                    
                    if agent.active: all_parked = False
                    if not agent.path_valid: path_error = True
                
                if not use_electric:
                    for _ in range(4):
                        for agent in agents: agent.resolve_collision(agents, custom_obstacles)

                if all_parked: timer_running = False

        # --- DRAWING ---
        screen.fill(OFF_WHITE)
        pygame.draw.rect(screen, GRAY_BG, (0, 0, PANEL_WIDTH, SCREEN_HEIGHT))
        pygame.draw.line(screen, (60,60,60), (PANEL_WIDTH, 0), (PANEL_WIDTH, SCREEN_HEIGHT), 2)
        
        # --- SIDEBAR CONTENT ---
        # 1. Title Area
        if state == "RUNNING":
            if timer_running:
                elapsed = pygame.time.get_ticks() - start_time - paused_time_accumulator
                final_time = elapsed
            elif state == "PAUSED":
                pass
        elif state == "DRAWING":
            final_time = 0
            
        time_str = f"{final_time/1000:.2f}s"
        timer_surf = font_timer.render(time_str, True, GREEN if not timer_running and state=="RUNNING" else TEXT_WHITE)
        screen.blit(timer_surf, (20, 30))
        
        status_txt = state
        if not timer_running and state == "RUNNING": status_txt = "FINISHED"
        screen.blit(font_bold.render(f"STATUS: {status_txt}", True, TEXT_GRAY), (20, 70))
        
        if state == "DRAWING":
            screen.blit(font_header.render("INSTRUCTIONS", True, TEXT_WHITE), (20, 120))
            screen.blit(font_ui.render("L-Click: Draw Wall", True, TEXT_GRAY), (20, 155))
            screen.blit(font_ui.render("R-Click: Erase Wall", True, TEXT_GRAY), (20, 180))
            screen.blit(font_ui.render("ENTER: Start Sim", True, GREEN), (20, 215))
        else:
            screen.blit(font_header.render("LIVE EDITING", True, TEXT_WHITE), (20, 120))
            screen.blit(font_ui.render("Draw/Erase walls to", True, TEXT_GRAY), (20, 155))
            screen.blit(font_ui.render("force re-routing!", True, TEXT_GRAY), (20, 180))

        # 2. Controls
        for btn in buttons: btn.draw(screen, font_ui)
        
        # 3. Algorithm Label & Dropdown
        screen.blit(font_header.render("Algorithm", True, TEXT_WHITE), (20, 355)) # Above dropdown
        algo_dropdown.draw(screen, font_ui)

        # --- MAP AREA ---
        map_clip = pygame.Rect(PANEL_WIDTH, 0, MAP_WIDTH, MAP_HEIGHT)
        screen.set_clip(map_clip)
        
        s_rect = start_rect.move(PANEL_WIDTH, 0)
        e_rect = end_rect.move(PANEL_WIDTH, 0)
        pygame.draw.rect(screen, GREEN, s_rect, 2)
        pygame.draw.rect(screen, GREEN, e_rect, 2)
        
        lbl_s = font_label.render("START", True, GREEN)
        lbl_e = font_label.render("END", True, GREEN)
        screen.blit(lbl_s, (s_rect.x, s_rect.y - 18))
        screen.blit(lbl_e, (e_rect.x, e_rect.y - 18))

        if state in ["RUNNING", "PAUSED"] and exit_manager:
            draw_exit_grid(screen, exit_manager)

        for obs in custom_obstacles:
            r = obs.move(PANEL_WIDTH, 0)
            pygame.draw.rect(screen, BLUE, r)
        
        if current_drawing_rect:
            preview = [current_drawing_rect[0] + PANEL_WIDTH, current_drawing_rect[1], current_drawing_rect[2], current_drawing_rect[3]]
            temp_rect = pygame.Rect(preview)
            temp_rect.normalize()
            color = RED if input_blocked else (100, 100, 255)
            pygame.draw.rect(screen, color, temp_rect, 2)

        if state in ["RUNNING", "PAUSED"]:
            if use_electric: # Visuals for electric mode
                draw_electric_field(screen, agents, custom_obstacles, end_rect)
            
            for agent in agents:
                draw_pos = (int(agent.pos.x + PANEL_WIDTH), int(agent.pos.y))
                if len(agent.path) > 1 and agent.active:
                    future_waypoints = agent.path[agent.current_wp_index:]
                    offset_start = (agent.pos.x + PANEL_WIDTH, agent.pos.y)
                    display_points = [offset_start] + [(p[0] + PANEL_WIDTH, p[1]) for p in future_waypoints]
                    if len(display_points) > 1:
                        pygame.draw.lines(screen, YELLOW, False, display_points, 1)
                pygame.draw.circle(screen, agent.get_color(), draw_pos, AGENT_RADIUS)
                pygame.draw.circle(screen, BLACK, draw_pos, AGENT_RADIUS, 1)
            
            if path_error:
                box_rect = pygame.Rect(PANEL_WIDTH + MAP_WIDTH//2 - 200, MAP_HEIGHT//2 - 50, 400, 100)
                pygame.draw.rect(screen, (50, 50, 50), box_rect)
                pygame.draw.rect(screen, RED, box_rect, 3)
                txt1 = error_font.render("NO PATH FOUND!", True, RED)
                screen.blit(txt1, (box_rect.centerx - txt1.get_width()//2, box_rect.centery - 15))

        screen.set_clip(None)
        if is_recording and state == "RUNNING":
            pygame.draw.circle(screen, RED, (SCREEN_WIDTH - 30, 30), 8)
            if frame_counter % 2 == 0: recorded_frames.append(screen.copy())
            frame_counter += 1

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()