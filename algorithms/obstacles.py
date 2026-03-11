import pygame
import math
from fundamental import _point_to_segment_dist, clamp, _segments_cross

class Obstacle:
    def __init__(self):
        pass

    def inflate(self, dx, dy):
        raise NotImplementedError

    def collidepoint(self, x, y):
        raise NotImplementedError

    def colliderect(self, rect):
        raise NotImplementedError

    def move(self, dx, dy):
        raise NotImplementedError

    def draw(self, surface, color, offset_x=0, offset_y=0):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @staticmethod
    def from_dict(data):
        if data['type'] == 'rect':
            return RectObstacle(data['x'], data['y'], data['w'], data['h'])
        elif data['type'] == 'circle':
            return CircleObstacle(data['cx'], data['cy'], data['r'])
        elif data['type'] == 'freehand':
            return FreehandObstacle(data['points'], data['thickness'])
        return None

class RectObstacle(Obstacle):
    def __init__(self, x, y, w, h):
        super().__init__()
        self.rect = pygame.Rect(x, y, w, h)
        self.rect.normalize()

    @property
    def left(self): return self.rect.left
    @property
    def right(self): return self.rect.right
    @property
    def top(self): return self.rect.top
    @property
    def bottom(self): return self.rect.bottom
    @property
    def width(self): return self.rect.width
    @property
    def height(self): return self.rect.height
    @property
    def x(self): return self.rect.x
    @property
    def y(self): return self.rect.y

    def inflate(self, dx, dy):
        inf = self.rect.inflate(dx, dy)
        return RectObstacle(inf.x, inf.y, inf.width, inf.height)

    def collidepoint(self, p, y=None):
        if y is not None:
             return self.rect.collidepoint(p, y)
        return self.rect.collidepoint(p)

    def colliderect(self, rect):
        if isinstance(rect, RectObstacle):
            return self.rect.colliderect(rect.rect)
        elif isinstance(rect, pygame.Rect):
            return self.rect.colliderect(rect)
        else:
            return rect.colliderect(self.rect)  # Let the other shape handle it if it knows how

    def clipline(self, p1, p2):
        return self.rect.clipline(p1, p2)

    def move(self, dx, dy):
        new_rect = self.rect.move(dx, dy)
        return RectObstacle(new_rect.x, new_rect.y, new_rect.width, new_rect.height)

    def draw(self, surface, color, offset_x=0, offset_y=0):
        draw_rect = self.rect.move(offset_x, offset_y)
        pygame.draw.rect(surface, color, draw_rect)

    def to_dict(self):
        return {
            'type': 'rect',
            'x': self.rect.x,
            'y': self.rect.y,
            'w': self.rect.width,
            'h': self.rect.height
        }

class CircleObstacle(Obstacle):
    def __init__(self, cx, cy, r):
        super().__init__()
        self.cx = cx
        self.cy = cy
        self.r = abs(r)

    @property
    def left(self): return self.cx - self.r
    @property
    def right(self): return self.cx + self.r
    @property
    def top(self): return self.cy - self.r
    @property
    def bottom(self): return self.cy + self.r
    @property
    def width(self): return self.r * 2
    @property
    def height(self): return self.r * 2
    @property
    def x(self): return self.cx - self.r
    @property
    def y(self): return self.cy - self.r

    def inflate(self, dx, dy):
        # We roughly approximate inflation by taking the max of dx and dy
        # divided by 2 (since inflate usually adds dx to width, meaning dx/2 to radius)
        inflation_radius = max(dx, dy) / 2
        return CircleObstacle(self.cx, self.cy, self.r + inflation_radius)

    def collidepoint(self, p, y_opt=None):
        px = p[0] if y_opt is None else p
        py = p[1] if y_opt is None else y_opt
        return math.hypot(px - self.cx, py - self.cy) <= self.r

    def colliderect(self, rect):
        # AABB vs Circle collision
        if isinstance(rect, RectObstacle): rect = rect.rect
        closest_x = clamp(self.cx, rect.left, rect.right)
        closest_y = clamp(self.cy, rect.top, rect.bottom)
        return math.hypot(self.cx - closest_x, self.cy - closest_y) <= self.r

    def clipline(self, p1, p2):
        return _point_to_segment_dist(self.cx, self.cy, p1[0], p1[1], p2[0], p2[1]) <= self.r

    def move(self, dx, dy):
        return CircleObstacle(self.cx + dx, self.cy + dy, self.r)

    def draw(self, surface, color, offset_x=0, offset_y=0):
        pygame.draw.circle(surface, color, (int(self.cx + offset_x), int(self.cy + offset_y)), int(self.r))

    def to_dict(self):
        return {
            'type': 'circle',
            'cx': self.cx,
            'cy': self.cy,
            'r': self.r
        }

class FreehandObstacle(Obstacle):
    def __init__(self, points, thickness=5):
        super().__init__()
        self.points = points  # List of (x, y) tuples
        self.thickness = thickness
        self._calc_bounds()

    def _calc_bounds(self):
        if not self.points:
            self._left = self._right = self._top = self._bottom = 0
            return
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        padding = self.thickness / 2
        self._left = min(xs) - padding
        self._right = max(xs) + padding
        self._top = min(ys) - padding
        self._bottom = max(ys) + padding

    @property
    def left(self): return self._left
    @property
    def right(self): return self._right
    @property
    def top(self): return self._top
    @property
    def bottom(self): return self._bottom
    @property
    def width(self): return self._right - self._left
    @property
    def height(self): return self._bottom - self._top
    @property
    def x(self): return self._left
    @property
    def y(self): return self._top

    def inflate(self, dx, dy):
        # Inflating a freehand is mostly increasing its mathematical thickness effectively
        # dx and dy are total width/height adds, so we add max(dx, dy) to thickness
        return FreehandObstacle(self.points, self.thickness + max(dx, dy))

    def collidepoint(self, p, y_opt=None):
        px = p[0] if y_opt is None else p
        py = p[1] if y_opt is None else y_opt
        
        # Quick AABB cull
        if px < self.left or px > self.right or py < self.top or py > self.bottom: return False
        
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i+1]
            dist = _point_to_segment_dist(px, py, p1[0], p1[1], p2[0], p2[1])
            if dist <= self.thickness / 2:
                return True
        return False

    def colliderect(self, rect):
        if isinstance(rect, RectObstacle): rect = rect.rect
        # AABB vs Freehand lines (thick lines)
        # 1. Quick AABB cull
        if rect.right < self.left or rect.left > self.right or rect.bottom < self.top or rect.top > self.bottom:
            return False
            
        # 2. Check if any line segment intersects the rect
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i+1]
            # Simple check: clip line against AABB
            # (or simplified: check if endpoints are in rect, or if distance from rect center to segment < radius)
            # More accurate: check point-segment distance from rect corners/center, etc.
            # Easiest solid collision check:
            if rect.collidepoint(p1) or rect.collidepoint(p2): return True
            
            # Distance from rect center to segment
            cx, cy = rect.center
            approx_radius = max(rect.width, rect.height) / 2
            if _point_to_segment_dist(cx, cy, p1[0], p1[1], p2[0], p2[1]) <= (approx_radius + self.thickness/2):
                return True
                
        return False

    def clipline(self, p1, p2):
        min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
        min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
        if max_x < self.left or min_x > self.right or max_y < self.top or min_y > self.bottom:
            return False
            
        for i in range(len(self.points) - 1):
            fp1 = self.points[i]
            fp2 = self.points[i+1]
            if _segments_cross(p1, p2, fp1, fp2): return True
            if _point_to_segment_dist(p1[0], p1[1], fp1[0], fp1[1], fp2[0], fp2[1]) <= self.thickness / 2: return True
            if _point_to_segment_dist(p2[0], p2[1], fp1[0], fp1[1], fp2[0], fp2[1]) <= self.thickness / 2: return True
        return False

    def move(self, dx, dy):
        new_points = [(p[0] + dx, p[1] + dy) for p in self.points]
        return FreehandObstacle(new_points, self.thickness)

    def draw(self, surface, color, offset_x=0, offset_y=0):
        if len(self.points) > 1:
            draw_points = [(p[0] + offset_x, p[1] + offset_y) for p in self.points]
            pygame.draw.lines(surface, color, False, draw_points, int(self.thickness))
            for p in draw_points:
                pygame.draw.circle(surface, color, (int(p[0]), int(p[1])), int(self.thickness/2))

    def to_dict(self):
        return {
            'type': 'freehand',
            'points': self.points,
            'thickness': self.thickness
        }
