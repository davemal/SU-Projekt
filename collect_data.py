import pygame
import random
import pickle
import os
import numpy as np
from collections import deque

import config as c
import utils
from renderer import GameRenderer

# Konfigurace
DATA_FILE = "snake_data.pkl"
ABSOLUTE_ACTIONS = {
    (0, -1): 0, # UP
    (0, 1): 1,  # DOWN
    (-1, 0): 2, # LEFT
    (1, 0): 3   # RIGHT
}

class DataCollector:
    def __init__(self):
        self.renderer = GameRenderer()
        pygame.display.set_caption('Snake Data Collector (Arrows to Move, S=Save, D=Discard)')
        
        self.reset()
        self.input_buffer = [] 

    def reset(self):
        # Reset hry do počátečního stavu
        self.direction = random.choice(utils.DIR_VECS)
        self.head = [c.GRID_W // 2, c.GRID_H // 2]
        
        tx, ty = -self.direction[0], -self.direction[1]
        self.snake = [
            self.head,
            [self.head[0] + tx, self.head[1] + ty],
            [self.head[0] + 2*tx, self.head[1] + 2*ty]
        ]
        
        self.score = 0
        self.place_food()
        self.current_game_memory = []
        self.input_buffer = [] 

    def place_food(self):
        # Umístění jídla na prázdné místo
        while True:
            self.food = [random.randint(0, c.GRID_W-1), random.randint(0, c.GRID_H-1)]
            if self.food not in self.snake: break

    def save_to_disk(self):
        # Uložení hry do souboru
        all_games = []
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "rb") as f:
                    all_games = pickle.load(f)
            except EOFError:
                all_games = []
        
        game_record = {
            "steps": self.current_game_memory,
            "final_score": self.score
        }
        all_games.append(game_record)
        
        with open(DATA_FILE, "wb") as f:
            pickle.dump(all_games, f)
            
        print(f"✅ Uloženo! Celkem her v datasetu: {len(all_games)}")

    def handle_input(self):
        # Zpracování kláves a bufferování pohybů
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if len(self.input_buffer) < 3:
                    if event.key == pygame.K_UP: self.input_buffer.append((0, -1))
                    elif event.key == pygame.K_DOWN: self.input_buffer.append((0, 1))
                    elif event.key == pygame.K_LEFT: self.input_buffer.append((-1, 0))
                    elif event.key == pygame.K_RIGHT: self.input_buffer.append((1, 0))
        return True

    def process_move(self):
        # Aplikace změny směru z bufferu
        if self.input_buffer:
            next_move = self.input_buffer[0]
            curr_x, curr_y = self.direction
            next_x, next_y = next_move
            if not (curr_x == -next_x and curr_y == -next_y):
                self.direction = next_move
            self.input_buffer.pop(0)

        # Uložení stavu
        state = utils.get_game_state_dict(self.snake, self.food, self.direction)
        state['full_body'] = list(self.snake) 
        state['score'] = self.score

        action_idx = ABSOLUTE_ACTIONS.get(self.direction, 0)
        self.current_game_memory.append((state, action_idx))

        # Nejdřív vypočítáme novou hlavu
        nx, ny = self.head[0] + self.direction[0], self.head[1] + self.direction[1]
        self.head = [nx, ny]

        # Kontrola kolize
        if nx < 0 or nx >= c.GRID_W or ny < 0 or ny >= c.GRID_H or self.head in self.snake:
            return True # Konec hry

        # Posun hada
        self.snake.insert(0, self.head)

        # Kontrola jídla
        if self.head == self.food:
            self.score += 1
            self.renderer.trigger_effect(self.head[0], self.head[1])
            self.place_food()
        else:
            self.snake.pop()
            
        return False

    def show_game_over_screen(self):
        """Obrazovka po konci hry (Uložit/Zahodit)."""
        self.renderer.draw(self.snake, self.food, self.score)
        
        # Overlay
        overlay = pygame.Surface((c.WINDOW_W, c.WINDOW_H + 70))
        overlay.set_alpha(200) 
        overlay.fill((0, 0, 0))
        self.renderer.display.blit(overlay, (0, 0))
        
        font = self.renderer.font_score 
        
        txt_score = font.render(f"GAME OVER: {self.score}", True, c.COL_FOOD)
        txt_save = font.render("[S] SAVE", True, c.COL_TEXT)
        txt_discard = font.render("[D] DISCARD", True, c.COL_TEXT_DIM)
        
        cx = c.WINDOW_W // 2
        cy = c.WINDOW_H // 2
        
        r1 = txt_score.get_rect(center=(cx, cy - 40))
        r2 = txt_save.get_rect(center=(cx, cy + 10))
        r3 = txt_discard.get_rect(center=(cx, cy + 50))
        
        self.renderer.display.blit(txt_score, r1)
        self.renderer.display.blit(txt_save, r2)
        self.renderer.display.blit(txt_discard, r3)
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        self.save_to_disk()
                        waiting = False
                    elif event.key == pygame.K_d:
                        print("🗑️ Zahozeno.")
                        waiting = False
        return True

    def run(self):
        running = True
        
        move_timer = 0.0
        move_interval = 1.0 / c.GAME_SPEED 

        while running:
            self.renderer.tick() 
            dt = self.renderer.dt 

            if not self.handle_input():
                break
            
            move_timer += dt
            game_over = False
            
            # Pohyb v intervalech
            if move_timer >= move_interval:
                move_timer -= move_interval
                game_over = self.process_move()

            self.renderer.draw(self.snake, self.food, self.score)
            
            if game_over:
                should_continue = self.show_game_over_screen()
                self.renderer.tick() 

                if not should_continue:
                    running = False
                else:
                    self.reset()
                    move_timer = 0
        
        self.renderer.quit()

if __name__ == "__main__":
    DataCollector().run()