# play.py
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import pygame

import config as c
from model import MultiTaskSequenceTransformer
from renderer import GameRenderer
import utils

# Inicializace a Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_FILE = "snake_transformer_best.pth"

class AIPlayer:
    def __init__(self):
        self.renderer = GameRenderer()
        
        # Načtení modelu
        self.model = MultiTaskSequenceTransformer(
            input_dim=c.INPUT_DIM, 
            model_dim=c.MODEL_DIM,
            num_heads=c.NUM_HEADS, 
            num_layers=c.NUM_LAYERS, 
            predict_steps=c.PREDICT_STEPS, 
            max_len=c.SEQ_LEN
        ).to(device)
        
        try:
            self.model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
            self.model.eval()
            print(f"Model {MODEL_FILE} načten.")
        except FileNotFoundError:
            print(f"Nelze načíst {MODEL_FILE}!")
            exit()

        self.reset()
        self.paused = False

    def reset(self):
        # Reset stavu
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
        
        # Historie (buffer)
        self.history = deque(maxlen=c.SEQ_LEN)
        
        # Předvyplnění historie
        state_dict = utils.get_game_state_dict(self.snake, self.food, self.direction)
        vec = utils.prepare_vector(state_dict)
        for _ in range(c.SEQ_LEN):
            self.history.append(vec)
            
        self.trap_prob = 0.0
        self.steps_since_food = 0
        self.panic_mode = False

    def place_food(self):
        while True:
            self.food = [random.randint(0, c.GRID_W-1), random.randint(0, c.GRID_H-1)]
            if self.food not in self.snake: break

    def step(self):
        # 1. Update stavu
        state_dict = utils.get_game_state_dict(self.snake, self.food, self.direction)
        vec = utils.prepare_vector(state_dict)
        self.history.append(vec)
        
        inp = torch.tensor(np.array(self.history), dtype=torch.float32).unsqueeze(0).to(device)
        
        # 2. Predikce AI
        with torch.no_grad():
            seq_actions, trap_val = self.model(inp)
            
            # První krok
            first_step_logits = seq_actions[0, 0, :] / c.TEMPERATURE
            probs = F.softmax(first_step_logits, dim=0).cpu().numpy()
            
            # Past?
            self.trap_prob = torch.sigmoid(trap_val[0]).item()

        # 3. Logika (Bezpečnost + Anti-Loop)
        final_action = 0 
        
        if self.steps_since_food > c.MAX_STEPS_NO_FOOD:
            self.panic_mode = True
            # Panika: Zkusit náhodný bezpečný tah
            acts = [0, 1, 2]
            random.shuffle(acts)
            for act in acts:
                if utils.is_safe(self.head, utils.get_new_direction_relative(self.direction, act), self.snake):
                    final_action = act
                    self.steps_since_food -= 20 
                    break
        else:
            self.panic_mode = False
            # Normální režim
            safe_options = []
            for act in range(3):
                test_dir = utils.get_new_direction_relative(self.direction, act)
                if utils.is_safe(self.head, test_dir, self.snake):
                    safe_options.append((act, probs[act]))
            
            if safe_options:
                acts, p = zip(*safe_options)
                p = np.array(p)
                p = p / p.sum() 
                final_action = np.random.choice(acts, p=p)
            else:
                final_action = np.argmax(probs) 

        # 4. Pohyb
        self.direction = utils.get_new_direction_relative(self.direction, final_action)
        nx, ny = self.head[0] + self.direction[0], self.head[1] + self.direction[1]
        self.head = [nx, ny]
        
        # Kolize
        if nx < 0 or nx >= c.GRID_W or ny < 0 or ny >= c.GRID_H or self.head in self.snake:
            return True # Konec
        
        self.snake.insert(0, self.head)
        
        if self.head == self.food:
            self.score += 1
            self.renderer.trigger_effect(self.head[0], self.head[1])
            self.place_food()
            self.steps_since_food = 0
        else:
            self.snake.pop()
            self.steps_since_food += 1
            
        return False

    def run(self):
        running = True
        move_timer = 0.0
        move_interval = 1.0 / c.GAME_SPEED 

        while running:
            # Vstup
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: self.paused = not self.paused
                    if event.key == pygame.K_r: self.reset()

            # Render
            self.renderer.tick()
            dt = self.renderer.dt

            if not self.paused:
                move_timer += dt
                if move_timer >= move_interval:
                    move_timer = 0 
                    
                    game_over = self.step()
                    if game_over:
                        print(f"Game Over! Skóre: {self.score}")
                        self.reset()
            
            self.renderer.draw(
                self.snake, self.food, self.score, 
                self.trap_prob, self.steps_since_food, self.panic_mode
            )
        
        self.renderer.quit()

if __name__ == "__main__":
    AIPlayer().run()