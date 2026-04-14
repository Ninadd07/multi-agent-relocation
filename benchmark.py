
import os
import json
import math
import random
import statistics
import pygame

from config import *
from parking import SmartExit, get_grid_positions
from fundamental import Agent, CentralManager
from algorithms.field import update_electric
from algorithms.standard import GlobalPlanner
from algorithms.standard_queue import QueuePlanner
from algorithms.discrete_grid import Discretisation
from algorithms.thetastar import ThetaStarPlanner
from algorithms.obstacles import Obstacle, RectObstacle


# ----------------------------
# SETTINGS
# ----------------------------
MAP_FILES = [f"mars_map_{i}.json" for i in range(1, 11)]
RUNS_PER_MAP = 5

# Design-space / base canvas size
BASE_WIDTH = 1720
BASE_HEIGHT = 980

LEFT_PANEL_WIDTH = 500
MAP_VIEW_WIDTH = 1160
MAP_VIEW_HEIGHT = 860

BG = (244, 247, 252)
WHITE = (255, 255, 255)
PANEL_BG = (24, 28, 34)
CARD_BG = (36, 42, 50)
CARD_BG_ALT = (44, 50, 58)
PANEL_BORDER = (70, 78, 90)

TEXT = (245, 247, 250)
SUBTEXT = (180, 188, 198)
MUTED = (130, 140, 150)

ACCENT = (66, 133, 244)
ACCENT_HOVER = (88, 150, 255)
GREEN_OK = (70, 190, 100)
RED_BAD = (230, 90, 90)
YELLOW_WARN = (245, 190, 80)

ALGOS = [
    ("Electric", "electric"),
    ("Standard", "standard"),
    ("Penalized", "penalized"),
    ("Discrete", "discrete"),
    ("Theta", "theta"),
    ("Queue", "queue"),
]


# ----------------------------
# HELPERS
# ----------------------------
def load_map(filename):
    filepath = os.path.join("maps", filename)
    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        loaded = []
        for item in data:
            if isinstance(item, list):
                loaded.append(RectObstacle(item[0], item[1], item[2], item[3]))
            else:
                loaded.append(Obstacle.from_dict(item))

        return [o for o in loaded if o is not None]
    except Exception as e:
        print(f"[ERROR] Failed to load {filename}: {e}")
        return []


def make_planner(algo_name, obstacles):
    if algo_name == "electric":
        return GlobalPlanner(obstacles, mode="standard")
    elif algo_name == "standard":
        return GlobalPlanner(obstacles, mode="standard")
    elif algo_name == "penalized":
        return GlobalPlanner(obstacles, mode="penalized")
    elif algo_name == "tbc":
        return GlobalPlanner(obstacles, mode="tbc")
    elif algo_name == "discrete":
        return Discretisation(obstacles)
    elif algo_name == "theta":
        return ThetaStarPlanner(obstacles)
    elif algo_name == "queue":
        return QueuePlanner(obstacles)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def percentile(sorted_data, p):
    if not sorted_data:
        return None
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def should_periodic_replan(algo_name):
    # Optimized policy:
    # skip frequent replanning for electric / penalized / queue
    return algo_name in {"standard", "tbc", "discrete", "theta"}


def fmt_time(v):
    if v is None:
        return "-"
    return f"{v:.2f}s"


def draw_card(surface, rect, fill=CARD_BG, border=PANEL_BORDER, radius=10):
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, 1, border_radius=radius)


def get_initial_window_size():
    info = pygame.display.Info()
    width = min(BASE_WIDTH, max(1100, info.current_w - 80))
    height = min(BASE_HEIGHT, max(700, info.current_h - 100))
    return width, height


def scale_mouse_pos(raw_mouse_pos, base_width, base_height, window_width, window_height):
    scale_x = base_width / max(window_width, 1)
    scale_y = base_height / max(window_height, 1)
    return (
        int(raw_mouse_pos[0] * scale_x),
        int(raw_mouse_pos[1] * scale_y),
    )


# ----------------------------
# UI
# ----------------------------
class Button:
    def __init__(self, rect, text, value=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.value = value

    def draw(self, surface, font, active=False, hovered=False):
        if active:
            fill = ACCENT
        elif hovered:
            fill = ACCENT_HOVER
        else:
            fill = CARD_BG_ALT

        pygame.draw.rect(surface, fill, self.rect, border_radius=8)
        pygame.draw.rect(surface, PANEL_BORDER, self.rect, 1, border_radius=8)

        txt = font.render(self.text, True, TEXT)
        txt_rect = txt.get_rect(center=self.rect.center)
        surface.blit(txt, txt_rect)

    def hit(self, pos):
        return self.rect.collidepoint(pos)


# ----------------------------
# BENCHMARK STATE
# ----------------------------
class BenchmarkState:
    def __init__(self):
        self.selected_algo = "penalized"
        self.running = False
        self.paused = False

        self.map_index = 0
        self.run_index = 0

        self.map_results = {name: [] for name in MAP_FILES}
        self.fail_counts = {name: 0 for name in MAP_FILES}

        self.obstacles = []
        self.planner = None
        self.exit_manager = None
        self.central_manager = None
        self.agents = []

        self.elapsed_time_ms = 0.0
        self.completion_time_ms = None
        self.use_electric = False
        self.replan_interval_ms = 2000
        self.replan_accumulator_ms = 0.0

        self.start_center = (100, 100)
        self.end_center = (MAP_WIDTH - 150, MAP_HEIGHT - 150)
        self.spawn_positions, self.start_rect = get_grid_positions(self.start_center, NUM_AGENTS)
        self.parking_positions, self.end_rect = get_grid_positions(self.end_center, NUM_AGENTS)

    def reset_all_results(self):
        self.map_results = {name: [] for name in MAP_FILES}
        self.fail_counts = {name: 0 for name in MAP_FILES}
        self.map_index = 0
        self.run_index = 0
        self.running = False
        self.paused = False
        self.clear_current_sim()

    def clear_current_sim(self):
        self.obstacles = []
        self.planner = None
        self.exit_manager = None
        self.central_manager = None
        self.agents = []
        self.elapsed_time_ms = 0.0
        self.completion_time_ms = None
        self.use_electric = False
        self.replan_accumulator_ms = 0.0

    def current_map_name(self):
        if 0 <= self.map_index < len(MAP_FILES):
            return MAP_FILES[self.map_index]
        return None

    def all_times(self):
        out = []
        for m in MAP_FILES:
            out.extend(self.map_results[m])
        return out

    def prepare_current_run(self):
        if self.map_index >= len(MAP_FILES):
            self.running = False
            return

        map_name = self.current_map_name()
        self.obstacles = load_map(map_name)
        self.planner = make_planner(self.selected_algo, self.obstacles)
        self.exit_manager = SmartExit(self.end_rect)
        self.central_manager = CentralManager(self.planner, self.exit_manager)

        self.agents = [Agent(pos, self.exit_manager) for pos in self.spawn_positions]

        if isinstance(self.planner, Discretisation):
            self.planner.set_agents(self.agents)

        self.central_manager.plan_all_paths(self.agents)

        if isinstance(self.planner, QueuePlanner):
            self.planner.reset_queue(self.agents)

        self.elapsed_time_ms = 0.0
        self.completion_time_ms = None
        self.replan_accumulator_ms = 0.0
        self.use_electric = (self.selected_algo == "electric")

    def start(self):
        self.map_index = 0
        self.run_index = 0
        self.map_results = {name: [] for name in MAP_FILES}
        self.fail_counts = {name: 0 for name in MAP_FILES}
        self.running = True
        self.paused = False
        self.prepare_current_run()

    def switch_algo(self, algo_name):
        self.selected_algo = algo_name
        self.start()

    def finalize_current_run(self, success=True):
        map_name = self.current_map_name()
        if map_name is None:
            self.running = False
            return

        if success and self.completion_time_ms is not None:
            self.map_results[map_name].append(self.completion_time_ms / 1000.0)
        elif success:
            self.map_results[map_name].append(self.elapsed_time_ms / 1000.0)
        else:
            self.fail_counts[map_name] += 1

        self.run_index += 1

        if self.run_index >= RUNS_PER_MAP:
            self.run_index = 0
            self.map_index += 1

        if self.map_index >= len(MAP_FILES):
            self.running = False
            self.clear_current_sim()
        else:
            self.prepare_current_run()

    def update_step(self, dt):
        if not self.running or self.paused or not self.agents:
            return

        self.elapsed_time_ms += dt

        if isinstance(self.planner, Discretisation):
            self.planner.set_agents(self.agents)

        if should_periodic_replan(self.selected_algo):
            self.replan_accumulator_ms += dt
            while self.replan_accumulator_ms >= self.replan_interval_ms:
                self.replan_accumulator_ms -= self.replan_interval_ms
                if isinstance(self.planner, Discretisation):
                    self.planner.set_agents(self.agents)
                self.central_manager.plan_all_paths(self.agents)

        if not isinstance(self.planner, QueuePlanner):
            needs_replan = False
            for agent in self.agents:
                if agent.is_stuck:
                    needs_replan = True
                    agent.is_stuck = False
                    agent.pos.x += random.uniform(-20, 20)
                    agent.pos.y += random.uniform(-20, 20)

            if needs_replan:
                if isinstance(self.planner, Discretisation):
                    self.planner.set_agents(self.agents)
                self.central_manager.plan_all_paths(self.agents)

        if not self.use_electric:
            if isinstance(self.planner, QueuePlanner):
                self.planner.tick(self.agents)
            else:
                for agent in self.agents:
                    agent.local_safety_check(self.agents)

        all_parked = True
        for agent in self.agents:
            if self.use_electric:
                update_electric(agent, self.agents, self.obstacles, self.end_rect)
            else:
                agent.update(self.end_rect)

            if agent.active:
                all_parked = False

        for _ in range(4):
            for agent in self.agents:
                if not agent.dfs_settled:
                    agent.resolve_collision(self.agents, self.obstacles)
            if self.exit_manager:
                self.exit_manager.resolve_collisions_inside(self.agents)

        if isinstance(self.planner, QueuePlanner):
            self.planner.enforce_queue(self.agents)

        if self.exit_manager:
            for agent in self.agents:
                if self.exit_manager.rect.collidepoint(agent.pos):
                    self.exit_manager.check_entry(agent)
                if agent.spot_reserved:
                    self.exit_manager.update_agent(agent)

        all_entered = all(agent.spot_reserved for agent in self.agents) if self.agents else False
        if all_entered:
            self.completion_time_ms = self.elapsed_time_ms
            for agent in self.agents:
                agent.active = False
                agent.velocity = pygame.math.Vector2(0, 0)
            self.finalize_current_run(success=True)
            return

        if all_parked:
            self.completion_time_ms = self.elapsed_time_ms
            self.finalize_current_run(success=True)
            return

        if self.elapsed_time_ms > 180000:
            self.finalize_current_run(success=False)


# ----------------------------
# DRAWING
# ----------------------------
def draw_map_preview(surface, state, rect, font_label, font_tiny, font_subtitle):
    title_rect = pygame.Rect(rect.x, rect.y - 46, rect.width, 36)
    draw_card(surface, title_rect, fill=CARD_BG)
    surface.blit(font_subtitle.render("Current Map Preview", True, TEXT), (rect.x + 14, rect.y - 40))

    pygame.draw.rect(surface, WHITE, rect, border_radius=10)
    pygame.draw.rect(surface, PANEL_BORDER, rect, 2, border_radius=10)

    map_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))
    map_surface.fill(OFF_WHITE)

    pygame.draw.rect(map_surface, GREEN, state.start_rect, 2)
    pygame.draw.rect(map_surface, GREEN, state.end_rect, 2)

    lbl_s = font_label.render("START", True, GREEN)
    lbl_e = font_label.render("END", True, GREEN)
    map_surface.blit(lbl_s, (state.start_rect.x, max(0, state.start_rect.y - 18)))
    map_surface.blit(lbl_e, (state.end_rect.x, max(0, state.end_rect.y - 18)))

    for obs in state.obstacles:
        obs.draw(map_surface, BLUE, 0, 0)

    for agent in state.agents:
        draw_pos = (int(agent.pos.x), int(agent.pos.y))

        if len(agent.path) > 1 and agent.active:
            future_waypoints = agent.path[agent.current_wp_index:]
            display_points = [(agent.pos.x, agent.pos.y)] + [(p[0], p[1]) for p in future_waypoints]
            if len(display_points) > 1:
                pygame.draw.lines(map_surface, YELLOW, False, display_points, 1)

        pygame.draw.circle(map_surface, agent.get_color(), draw_pos, AGENT_RADIUS)
        pygame.draw.circle(map_surface, BLACK, draw_pos, AGENT_RADIUS, 1)

        if hasattr(agent, "index"):
            num_surf = font_tiny.render(str(agent.index), True, WHITE)
            map_surface.blit(
                num_surf,
                (draw_pos[0] - num_surf.get_width() // 2, draw_pos[1] - num_surf.get_height() // 2),
            )

    scaled = pygame.transform.smoothscale(map_surface, (rect.width, rect.height))
    surface.blit(scaled, rect.topleft)


def draw_stats_panel(surface, state, left_rect, fonts, algo_buttons, mouse_pos):
    font_title, font_subtitle, font_text, font_small, font_tiny = fonts

    pygame.draw.rect(surface, PANEL_BG, left_rect)
    pygame.draw.rect(surface, PANEL_BORDER, left_rect, 2)

    x = left_rect.x + 20
    y = left_rect.y + 18

    surface.blit(font_title.render("Benchmark Dashboard", True, TEXT), (x, y))
    y += 46

    card_w = left_rect.width - 40
    card_h = 88
    summary_rect = pygame.Rect(x, y, card_w, card_h)
    draw_card(surface, summary_rect)

    current_map = state.current_map_name() if state.running else "-"
    run_num = state.run_index + 1 if state.running else 0
    status = "PAUSED" if state.paused else ("RUNNING" if state.running else "IDLE")

    surface.blit(font_text.render(f"Algorithm: {state.selected_algo}", True, TEXT), (x + 16, y + 14))
    surface.blit(font_text.render(f"Current map: {current_map}", True, SUBTEXT), (x + 16, y + 40))
    surface.blit(font_text.render(f"Run: {run_num}/{RUNS_PER_MAP}    Status: {status}", True, SUBTEXT), (x + 16, y + 64))
    y += card_h + 18

    surface.blit(font_subtitle.render("Algorithm", True, TEXT), (x, y))
    y += 34

    for b in algo_buttons:
        hovered = b.hit(mouse_pos)
        active = (b.value == state.selected_algo)
        b.draw(surface, font_text, active=active, hovered=hovered)

    y = 330

    surface.blit(font_subtitle.render("Map Results", True, TEXT), (x, y))
    y += 34

    results_rect = pygame.Rect(x, y, card_w, 500)
    draw_card(surface, results_rect, fill=CARD_BG_ALT)

    header_y = y + 14
    surface.blit(font_small.render("Map", True, SUBTEXT), (x + 16, header_y))
    surface.blit(font_small.render("Runs", True, SUBTEXT), (x + 155, header_y))
    surface.blit(font_small.render("Avg", True, SUBTEXT), (x + 225, header_y))
    surface.blit(font_small.render("Min", True, SUBTEXT), (x + 290, header_y))
    surface.blit(font_small.render("Max", True, SUBTEXT), (x + 355, header_y))
    surface.blit(font_small.render("Status", True, SUBTEXT), (x + 415, header_y))

    row_y = header_y + 28

    for idx, map_name in enumerate(MAP_FILES):
        row_rect = pygame.Rect(x + 10, row_y - 4, card_w - 20, 30)
        draw_card(surface, row_rect, fill=CARD_BG if idx % 2 == 0 else CARD_BG_ALT, border=CARD_BG_ALT, radius=8)

        times = state.map_results[map_name]
        avg = statistics.mean(times) if times else None
        mn = min(times) if times else None
        mx = max(times) if times else None
        runs_done = len(times)

        if idx < state.map_index:
            status_text = "DONE"
            status_color = GREEN_OK
        elif idx == state.map_index and state.running:
            status_text = "RUNNING"
            status_color = YELLOW_WARN
        else:
            status_text = "WAIT"
            status_color = MUTED

        surface.blit(font_tiny.render(map_name.replace(".json", ""), True, TEXT), (x + 18, row_y + 4))
        surface.blit(font_tiny.render(f"{runs_done}/{RUNS_PER_MAP}", True, TEXT), (x + 155, row_y + 4))
        surface.blit(font_tiny.render(fmt_time(avg), True, TEXT), (x + 225, row_y + 4))
        surface.blit(font_tiny.render(fmt_time(mn), True, TEXT), (x + 290, row_y + 4))
        surface.blit(font_tiny.render(fmt_time(mx), True, TEXT), (x + 355, row_y + 4))
        surface.blit(font_tiny.render(status_text, True, status_color), (x + 415, row_y + 4))

        row_y += 36

    overall_y = y + 520
    surface.blit(font_subtitle.render("Overall", True, TEXT), (x, overall_y))
    overall_y += 34

    overall_rect = pygame.Rect(x, overall_y, card_w, 150)
    draw_card(surface, overall_rect)

    all_times = state.all_times()
    if all_times:
        s = sorted(all_times)
        avg = statistics.mean(all_times)
        p1 = percentile(s, 1)
        p99 = percentile(s, 99)
        mn = min(all_times)
        mx = max(all_times)
        std = statistics.stdev(all_times) if len(all_times) > 1 else 0.0

        lines = [
            f"Runs completed: {len(all_times)}",
            f"Average: {avg:.2f}s",
            f"1% Low / High: {p1:.2f}s / {p99:.2f}s",
            f"Min / Max: {mn:.2f}s / {mx:.2f}s",
            f"Std Dev: {std:.2f}s",
        ]
    else:
        lines = ["No completed runs yet."]

    oy = overall_y + 16
    for line in lines:
        surface.blit(font_text.render(line, True, SUBTEXT), (x + 16, oy))
        oy += 24


def main():
    pygame.init()

    base_width = BASE_WIDTH
    base_height = BASE_HEIGHT

    window_width, window_height = get_initial_window_size()
    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    pygame.display.set_caption("MARS Benchmark Viewer")
    clock = pygame.time.Clock()

    base_surface = pygame.Surface((base_width, base_height))

    font_title = pygame.font.SysFont("Segoe UI", 30, bold=True)
    font_subtitle = pygame.font.SysFont("Segoe UI", 20, bold=True)
    font_text = pygame.font.SysFont("Segoe UI", 17)
    font_small = pygame.font.SysFont("Consolas", 15)
    font_tiny = pygame.font.SysFont("Segoe UI", 13)
    font_label = pygame.font.SysFont("Segoe UI", 12, bold=True)

    left_panel = pygame.Rect(0, 0, LEFT_PANEL_WIDTH, BASE_HEIGHT)
    map_rect = pygame.Rect(LEFT_PANEL_WIDTH + 24, 70, MAP_VIEW_WIDTH, MAP_VIEW_HEIGHT)

    algo_buttons = []
    bx = 20
    by = 205
    bw = 118
    bh = 34
    gap_x = 10
    gap_y = 10
    per_row = 3

    for i, (label, value) in enumerate(ALGOS):
        row = i // per_row
        col = i % per_row
        rect = (bx + col * (bw + gap_x), by + row * (bh + gap_y), bw, bh)
        algo_buttons.append(Button(rect, label, value))

    start_btn = Button((20, 295, 118, 34), "Start", "start")
    pause_btn = Button((148, 295, 118, 34), "Pause", "pause")
    reset_btn = Button((276, 295, 118, 34), "Reset", "reset")

    state = BenchmarkState()

    last_tick = pygame.time.get_ticks()
    running = True

    while running:
        now = pygame.time.get_ticks()
        dt = now - last_tick
        last_tick = now

        raw_mouse_pos = pygame.mouse.get_pos()
        mouse_pos = scale_mouse_pos(
            raw_mouse_pos,
            base_width,
            base_height,
            window_width,
            window_height,
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                window_width, window_height = max(900, event.w), max(600, event.h)
                screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                event_mouse_pos = mouse_pos
                if hasattr(event, "pos"):
                    event_mouse_pos = scale_mouse_pos(
                        event.pos,
                        base_width,
                        base_height,
                        window_width,
                        window_height,
                    )

                for b in algo_buttons:
                    if b.hit(event_mouse_pos):
                        state.switch_algo(b.value)

                if start_btn.hit(event_mouse_pos):
                    state.start()

                if pause_btn.hit(event_mouse_pos):
                    if state.running:
                        state.paused = not state.paused

                if reset_btn.hit(event_mouse_pos):
                    state.reset_all_results()

        state.update_step(dt)

        base_surface.fill(BG)

        draw_stats_panel(
            base_surface,
            state,
            left_panel,
            (font_title, font_subtitle, font_text, font_small, font_tiny),
            algo_buttons,
            mouse_pos,
        )

        start_btn.draw(base_surface, font_text, hovered=start_btn.hit(mouse_pos))
        pause_btn.draw(base_surface, font_text, hovered=pause_btn.hit(mouse_pos))
        reset_btn.draw(base_surface, font_text, hovered=reset_btn.hit(mouse_pos))

        draw_map_preview(base_surface, state, map_rect, font_label, font_tiny, font_subtitle)

        status_y = map_rect.bottom + 16
        if state.running:
            status = "PAUSED" if state.paused else "RUNNING"
        else:
            status = "IDLE"

        base_surface.blit(font_text.render(f"Status: {status}", True, BLACK), (map_rect.x, status_y))
        base_surface.blit(font_text.render(f"Algorithm: {state.selected_algo}", True, BLACK), (map_rect.x + 180, status_y))
        base_surface.blit(font_text.render(f"Map: {state.current_map_name() or '-'}", True, BLACK), (map_rect.x + 430, status_y))
        base_surface.blit(font_text.render(f"Elapsed: {state.elapsed_time_ms / 1000.0:.2f}s", True, BLACK), (map_rect.x + 700, status_y))

        scaled_surface = pygame.transform.smoothscale(base_surface, (window_width, window_height))
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()