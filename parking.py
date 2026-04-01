# parking.py
# Exit zone modelled as a gravitational depression / hole.
# Once an agent crosses the rim it is fully handed to the well physics.
# Agents are hard spheres – no overlap permitted.

import math
import pygame
from config import *
from fundamental import clamp


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def get_grid_positions(center, count):
    positions = []
    cols = int(math.ceil(math.sqrt(count)))
    spacing = AGENT_DIAMETER + 5

    start_x = center[0] - ((cols * spacing) / 2) + spacing / 2
    start_y = center[1] - ((cols * spacing) / 2) + spacing / 2

    for i in range(count):
        row = i // cols
        col = i % cols
        x = start_x + (col * spacing)
        y = start_y + (row * spacing)
        positions.append((x, y))

    zone_size = cols * spacing + 10
    zone_rect = pygame.Rect(
        center[0] - zone_size / 2,
        center[1] - zone_size / 2,
        zone_size,
        zone_size,
    )
    return positions, zone_rect


# ---------------------------------------------------------------------------
# SmartExit – gravitational depression
# ---------------------------------------------------------------------------
class SmartExit:
    """
    The exit zone behaves like a hole/depression in the floor.

    Physics model
    -------------
    * An agent is "free" while outside the rim (A* drives it).
    * The moment its centre crosses the rim rectangle the agent is
      *captured* – A* is discarded and well-physics takes over completely.
    * Inside the well a radial gravity force accelerates the agent toward
      the pit centre.  The force grows with depth (distance already fallen)
      so agents don't stall near the rim.
    * Agents are treated as hard discs (radius = AGENT_RADIUS).  A
      multi-pass impulse solver keeps them from overlapping both with each
      other and with the rim walls while they are inside.
    * When an agent is within 2 px of the centre *and* effectively still,
      it is marked inactive (parked).
    """

    # Tuning knobs -----------------------------------------------------------
    GRAVITY_BASE   = AGENT_SPEED * 0.55   # pull at the rim
    GRAVITY_SCALE  = 2.8                  # multiplier at dead-centre (depth effect)
    DAMPING        = 0.72                 # velocity damping each tick (< 1 = friction)
    SETTLE_DIST    = 2.5                  # px – snap-to-centre & park
    SETTLE_SPEED   = 0.25                 # px/tick – speed below which we consider settled
    PATIENCE_TICKS = 90                   # ticks stuck before forced-park
    SOLVER_PASSES  = 6                    # collision solver iterations per frame
    # ------------------------------------------------------------------------

    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.center_pixel = pygame.Vector2(rect.centerx, rect.centery)
        self.parked_count = 0

        # Largest circle that fits inside the rect (used for depth factor)
        self._max_depth = min(rect.width, rect.height) / 2.0

    # ------------------------------------------------------------------
    # Public API called by Agent / gui
    # ------------------------------------------------------------------

    def check_entry(self, agent):
        """
        Capture the agent the instant it physically enters the rim.
        Once captured the agent belongs entirely to the well.
        """
        if agent.spot_reserved:
            return  # already captured

        if self.rect.collidepoint(agent.pos.x, agent.pos.y):
            agent.spot_reserved   = True
            agent.parking_blend   = 1.0   # signal to Agent.update: well owns this
            agent.path            = []
            agent.fluid_target_cell = None if hasattr(agent, 'fluid_target_cell') else None
            agent.current_grid_cell = None if hasattr(agent, 'current_grid_cell') else None
            agent.grid_patience   = 0
            # freeze A* velocity contribution
            agent.velocity        = agent.velocity * 0.4

    def update_agent(self, agent):
        """
        One physics tick for a captured agent.
        Gravity + damping; no A* involvement.
        """
        if not agent.active:
            return

        to_centre = self.center_pixel - agent.pos
        dist      = to_centre.length()

        # ---- Settled? -------------------------------------------------------
        if dist < self.SETTLE_DIST and agent.velocity.length() < self.SETTLE_SPEED:
            agent.pos      = pygame.Vector2(self.center_pixel)
            agent.velocity = pygame.Vector2(0, 0)
            agent.active   = False
            self.park_agent(agent)
            return

        # ---- Gravity --------------------------------------------------------
        # Depth factor: agents closer to centre feel stronger pull (steep well)
        depth_factor = 1.0 + (self.GRAVITY_SCALE - 1.0) * (
            1.0 - clamp(dist / self._max_depth, 0.0, 1.0)
        )
        gravity_strength = self.GRAVITY_BASE * depth_factor

        if dist > 0:
            gravity = to_centre.normalize() * gravity_strength
        else:
            gravity = pygame.Vector2(0, 0)

        # Apply gravity then damp
        agent.velocity += gravity
        agent.velocity *= self.DAMPING

        # Cap speed
        spd = agent.velocity.length()
        if spd > AGENT_SPEED * 1.4:
            agent.velocity.scale_to_length(AGENT_SPEED * 1.4)

        agent.pos += agent.velocity

        # ---- Rim clamping ---------------------------------------------------
        # Captured agents must not escape through the rim
        self._clamp_to_rim(agent)

        # ---- Patience (deadlock guard) --------------------------------------
        moved = (agent.pos - getattr(agent, '_prev_park_pos',
                                     pygame.Vector2(agent.pos))).length()
        agent._prev_park_pos = pygame.Vector2(agent.pos)

        if moved < 0.08:
            agent.grid_patience = getattr(agent, 'grid_patience', 0) + 1
            if agent.grid_patience > self.PATIENCE_TICKS:
                agent.velocity = pygame.Vector2(0, 0)
                agent.active   = False
                self.park_agent(agent)
        else:
            agent.grid_patience = 0

    def resolve_collisions_inside(self, agents):
        """
        Hard-disc impulse solver for all captured agents.
        Call this once per frame *after* all update_agent() calls.
        Multiple passes converge toward non-overlapping state.
        """
        captured = [a for a in agents if a.spot_reserved]
        if len(captured) < 2:
            return

        for _ in range(self.SOLVER_PASSES):
            for i in range(len(captured)):
                for j in range(i + 1, len(captured)):
                    a, b = captured[i], captured[j]
                    self._push_apart(a, b)

            # Re-clamp after each solver pass so agents never leak out
            for a in captured:
                if a.active:
                    self._clamp_to_rim(a)

    # ------------------------------------------------------------------
    # Legacy shim – called by Agent.resolve_collision for spot_reserved pairs
    # ------------------------------------------------------------------
    def resolve_collision(self, agent, other):
        self._push_apart(agent, other)
        if agent.active:
            self._clamp_to_rim(agent)
        if other.active:
            self._clamp_to_rim(other)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _push_apart(self, a, b):
        """Minimal impulse to separate two overlapping hard discs."""
        diff = a.pos - b.pos
        dist = diff.length()
        min_dist = AGENT_DIAMETER  # 2 * AGENT_RADIUS, no extra gap

        if dist >= min_dist:
            return

        if dist < 1e-4:
            # Exactly coincident – nudge randomly
            angle = math.atan2(a.pos.y - self.center_pixel.y,
                               a.pos.x - self.center_pixel.x)
            diff = pygame.Vector2(math.cos(angle + 0.1),
                                  math.sin(angle + 0.1))
            dist = 1e-4

        overlap   = min_dist - dist
        push      = diff.normalize() * (overlap * 0.5)   # split equally

        # Only move active agents; inactive (parked) agents are immovable
        if a.active and b.active:
            a.pos += push
            b.pos -= push
        elif a.active:
            a.pos += push * 2
        elif b.active:
            b.pos -= push * 2

        # Damp velocities on contact
        if a.active:
            a.velocity *= 0.6
        if b.active:
            b.velocity *= 0.6

    def _clamp_to_rim(self, agent):
        """Keep the agent's disc fully inside the rim rectangle."""
        agent.pos.x = clamp(
            agent.pos.x,
            self.rect.left  + AGENT_RADIUS,
            self.rect.right  - AGENT_RADIUS,
        )
        agent.pos.y = clamp(
            agent.pos.y,
            self.rect.top   + AGENT_RADIUS,
            self.rect.bottom - AGENT_RADIUS,
        )

    def park_agent(self, agent):
        self.parked_count += 1