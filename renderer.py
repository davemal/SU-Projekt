import pygame
import random
import math
import config as c

class GameRenderer:
    def __init__(self):
        pygame.init()
        self.display = pygame.display.set_mode((c.WINDOW_W, c.WINDOW_H))
        pygame.display.set_caption('SU Projekt - Behavior Cloning - Malík David')
        self.clock = pygame.time.Clock()
        
        fonts = ['Consolas', 'Courier New', 'Arial Black', 'Arial']
        self.font_score = None
        self.font_ui = None
        
        for f in fonts:
            try:
                self.font_score = pygame.font.SysFont(f, 32, bold=True)
                self.font_ui = pygame.font.SysFont(f, 14, bold=True)
                break
            except:
                continue

        self.y_offset = 70
        
        self.particles = [] # [x, y, vx, vy, size, max_size, start_col, end_col, decay_rate]
        self.shockwaves = [] 
        self.last_tail_pos = None 

        self.shake_timer = 0.0
        self.total_time = 0.0
        self.dt = 0.016

    def trigger_effect(self, grid_x, grid_y):
        """Efekt snězení jídla."""
        if not c.ENABLE_EFFECTS: return 
        
        center_x = grid_x * c.BLOCK_SIZE + c.BLOCK_SIZE // 2
        center_y = grid_y * c.BLOCK_SIZE + c.BLOCK_SIZE // 2 + self.y_offset
        self.shake_timer = c.SHAKE_DURATION

        # Kruhy (rázová vlna)
        for i in range(c.SHOCKWAVE_COUNT):
            start_radius = -(i * c.SHOCKWAVE_GAP)
            self.shockwaves.append([center_x, center_y, start_radius])

        # Částice (exploze)
        for _ in range(c.PARTICLE_COUNT):
            angle = random.uniform(0, 6.28)
            speed = random.uniform(c.PARTICLE_SPEED * 0.5, c.PARTICLE_SPEED)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            size = random.uniform(c.PARTICLE_SIZE_MIN, c.PARTICLE_SIZE_MAX)
            
            # x, y, vx, vy, size, max_size, color_start, color_end, decay
            self.particles.append([
                center_x, center_y, vx, vy, size, size, 
                c.COL_PARTICLE_START, c.COL_PARTICLE_END, c.PARTICLE_DECAY
            ])

    def _trigger_tail_dust(self, grid_x, grid_y):
        """
        Efekt rozplynutí na konci ocasu.
        """
        if not c.ENABLE_EFFECTS: return
        
        center_x = grid_x * c.BLOCK_SIZE + c.BLOCK_SIZE // 2
        center_y = grid_y * c.BLOCK_SIZE + c.BLOCK_SIZE // 2 + self.y_offset
        
        spawn_radius = (c.BLOCK_SIZE // 2) - c.SNAKE_PADDING
        
        for _ in range(c.TAIL_FX_COUNT):
            offset_x = random.uniform(-spawn_radius, spawn_radius)
            offset_y = random.uniform(-spawn_radius, spawn_radius)
            
            spawn_x = center_x + offset_x
            spawn_y = center_y + offset_y
            
            vx = random.uniform(-10, 10) 
            vy = random.uniform(-c.TAIL_FX_SPEED, -c.TAIL_FX_SPEED * 0.5)
            
            size = c.TAIL_FX_SIZE
            
            self.particles.append([
                spawn_x, spawn_y, vx, vy, size, size, 
                c.COL_TAIL_START, c.COL_TAIL_END, c.TAIL_FX_DECAY
            ])

    def _update_effects(self, dt):
        # Aktualizace částic
        for p in self.particles[::-1]:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            
            # Tření
            speed = math.sqrt(p[2]**2 + p[3]**2)
            if speed > 0:
                new_speed = max(0, speed - (c.PARTICLE_FRICTION * speed * dt))
                factor = new_speed / speed
                p[2] *= factor
                p[3] *= factor
            
            # Zmenšování
            p[4] -= p[8] * dt
            
            if p[4] <= 0:
                self.particles.remove(p)

        # Aktualizace vln
        for s in self.shockwaves[::-1]:
            s[2] += c.SHOCKWAVE_SPEED * dt
            if s[2] > c.SHOCKWAVE_MAX_SIZE: self.shockwaves.remove(s)

        if self.shake_timer > 0: self.shake_timer -= dt

    def _draw_effects(self, shake_x, shake_y):
        # Vykreslení vln
        for s in self.shockwaves:
            r = s[2]
            if r > 0:
                ratio = max(0.0, min(1.0, r / c.SHOCKWAVE_MAX_SIZE))
                color = self._get_gradient_color(c.COL_SHOCKWAVE_START, c.COL_SHOCKWAVE_END, ratio)
                rect = (int(s[0] - r + shake_x), int(s[1] - r + shake_y), int(r*2), int(r*2))
                pygame.draw.rect(self.display, color, rect, c.SHOCKWAVE_WIDTH)

        # Vykreslení částic
        for p in self.particles:
            life_ratio = max(0.0, min(1.0, p[4] / p[5]))
            color = self._get_gradient_color(p[7], p[6], life_ratio)
            pygame.draw.rect(self.display, color, (p[0] + shake_x, p[1] + shake_y, p[4], p[4]))

    def _get_gradient_color(self, start_col, end_col, ratio):
        r = start_col[0] + (end_col[0] - start_col[0]) * ratio
        g = start_col[1] + (end_col[1] - start_col[1]) * ratio
        b = start_col[2] + (end_col[2] - start_col[2]) * ratio
        return (int(r), int(g), int(b))

    def _draw_text_3d(self, text, font, x, y, color=c.COL_TEXT):
        shadow = font.render(text, True, c.COL_TEXT_SHADOW)
        self.display.blit(shadow, (x + 2, y + 2))
        surf = font.render(text, True, color)
        self.display.blit(surf, (x, y))

    def _draw_segmented_bar(self, x, y, value, max_val, scale=1.0):
        total_blocks = 10
        block_size = int(14 * scale) 
        gap = int(3 * scale)
        ratio = min(1.0, value / (max_val + 1e-6))
        filled_blocks = int(ratio * total_blocks)
        for i in range(total_blocks):
            bx = x + i * (block_size + gap)
            block_pos_ratio = i / max(1, total_blocks - 1)
            grad_col = self._get_gradient_color(c.COL_BAR_DIM, c.COL_BAR_BRIGHT, block_pos_ratio)
            color = grad_col if i < filled_blocks else c.COL_BAR_EMPTY
            pygame.draw.rect(self.display, color, (bx, y, block_size, block_size))

    def _draw_ui(self, score, trap_prob, hunger, panic):
        pygame.draw.rect(self.display, c.COL_UI_BG, (0, 0, c.WINDOW_W, self.y_offset))
        pygame.draw.line(self.display, (40, 40, 40), (0, self.y_offset-1), (c.WINDOW_W, self.y_offset-1), 2)
        
        self._draw_text_3d("SCORE", self.font_ui, 20, 10, c.COL_TEXT_DIM)
        self._draw_text_3d(str(score), self.font_score, 20, 28, c.COL_TEXT)

        center_x = c.WINDOW_W // 2
        danger_x = center_x - 90 
        self._draw_text_3d("DANGER", self.font_ui, danger_x, 15, c.COL_TEXT_DIM)
        self._draw_segmented_bar(danger_x, 35, trap_prob, 1.0, scale=1.0)
        
        hunger_x = center_x + 90
        lbl_col = c.COL_TEXT if panic else c.COL_TEXT_DIM
        self._draw_text_3d("HUNGER", self.font_ui, hunger_x, 15, lbl_col)
        self._draw_segmented_bar(hunger_x, 35, hunger, c.MAX_STEPS_NO_FOOD, scale=1.0)

    def draw(self, snake, food, score, trap_prob=0.0, hunger=0, panic=False):
        self.display.fill(c.COL_BG)
        self.total_time += self.dt
        
        shake_x = 0
        shake_y = 0
        if self.shake_timer > 0:
            shake_x = random.randint(-c.SHAKE_INTENSITY, c.SHAKE_INTENSITY)
            shake_y = random.randint(-c.SHAKE_INTENSITY, c.SHAKE_INTENSITY)
        
        for x in range(0, c.WINDOW_W, c.BLOCK_SIZE):
            pygame.draw.line(self.display, c.COL_GRID, (x + shake_x, self.y_offset + shake_y), (x + shake_x, c.WINDOW_H + shake_y))
        for y in range(self.y_offset, c.WINDOW_H, c.BLOCK_SIZE):
            pygame.draw.line(self.display, c.COL_GRID, (0 + shake_x, y + shake_y), (c.WINDOW_W + shake_x, y + shake_y))

        # Efekt ocasu
        if c.ENABLE_EFFECTS and len(snake) > 0:
            current_tail = snake[-1]
            if self.last_tail_pos is not None and self.last_tail_pos != current_tail:
                self._trigger_tail_dust(self.last_tail_pos[0], self.last_tail_pos[1])
            self.last_tail_pos = list(current_tail)

        # Jídlo
        fx, fy = food
        cell_center_x = fx * c.BLOCK_SIZE + (c.BLOCK_SIZE // 2) + shake_x
        cell_center_y = fy * c.BLOCK_SIZE + (c.BLOCK_SIZE // 2) + self.y_offset + shake_y
        pulse = math.sin(self.total_time * c.FOOD_PULSE_SPEED) * c.FOOD_PULSE_AMP
        current_pad = max(0, min(c.FOOD_PADDING - pulse, c.BLOCK_SIZE / 2 - 1))
        current_size = c.BLOCK_SIZE - (2 * current_pad)
        food_rect = pygame.Rect(0, 0, round(current_size), round(current_size))
        food_rect.center = (round(cell_center_x), round(cell_center_y))
        pygame.draw.rect(self.display, c.COL_FOOD, food_rect)

        # Had
        snake_len = len(snake)
        snake_pad = c.SNAKE_PADDING
        snake_size = c.BLOCK_SIZE - (2 * snake_pad)
        for i, pt in enumerate(snake):
            ratio = i / max(1, snake_len - 1)
            color = self._get_gradient_color(c.COL_SNAKE_HEAD, c.COL_SNAKE_TAIL, ratio)
            px = pt[0] * c.BLOCK_SIZE + snake_pad + shake_x
            py = pt[1] * c.BLOCK_SIZE + snake_pad + self.y_offset + shake_y
            pygame.draw.rect(self.display, color, (px, py, snake_size, snake_size))

        self._update_effects(self.dt)
        self._draw_effects(shake_x, shake_y)
        self._draw_ui(score, trap_prob, hunger, panic)
        pygame.display.flip()

    def tick(self):
        ms = self.clock.tick(c.FPS)
        self.dt = ms / 1000.0
        
    def quit(self):
        pygame.quit()