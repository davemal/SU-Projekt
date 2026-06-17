# utils.py
import numpy as np
import torch
from config import GRID_W, GRID_H

DIR_VECS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT




def get_game_state_dict(snake, food, direction):
    """
    Převede stav hry na features.
    """
    head_x, head_y = snake[0]
    
    # Lokální mřížka 11x11 kolem hlavy (0=prázdno, -1=zeď, -0.5=tělo, 1=jídlo)
    grid = np.zeros((11, 11))
    off = 5
    for i in range(11):
        for j in range(11):
            cx, cy = head_x - off + j, head_y - off + i
            val = 0.0
            if cx < 0 or cx >= GRID_W or cy < 0 or cy >= GRID_H:
                val = -1.0
            elif [cx, cy] in snake:
                val = -0.5
            elif [cx, cy] == food:
                val = 1.0
            grid[i][j] = val

    # Vektor k jídlu (food_fwd, food_right)
    fx, fy = food[0] - head_x, food[1] - head_y
    dist = max(abs(fx), abs(fy), 1)
    dx, dy = direction
    food_fwd = (fx * dx + fy * dy) / dist
    food_right = (fx * (-dy) + fy * dx) / dist

    # Vzdálenosti ke stěnám (Top, Bottom, Left, Right)
    wd = [
        head_y / GRID_H,               
        (GRID_H - 1 - head_y) / GRID_H,
        head_x / GRID_W,               
        (GRID_W - 1 - head_x) / GRID_W 
    ]

    # One-Hot směr (UP, DOWN, LEFT, RIGHT)
    dir_onehot = [0, 0, 0, 0]
    try:
        dir_onehot[DIR_VECS.index(tuple(direction))] = 1
    except ValueError:
        pass 

    return {
        "grid": grid,
        "food_vector": (food_fwd, food_right),
        "wall_dists": wd,
        "direction": dir_onehot,
        "snake_ratio": len(snake) / (GRID_W * GRID_H)
    }

def prepare_vector(state_dict):
    """
    Zploští features do vektoru a provede rotaci podle směru hada.
    """
    grid = state_dict['grid']
    wall_dists = state_dict['wall_dists']
    dir_idx = np.argmax(state_dict['direction'])

    # Rotace o 90, 180 nebo 270 stupňů
    k_rot = 0
    if dir_idx == 1: k_rot = 2    # DOWN -> 180
    elif dir_idx == 2: k_rot = 3  # LEFT -> 270
    elif dir_idx == 3: k_rot = 1  # RIGHT -> 90

    # Otočení mřížky
    rotated_grid = np.rot90(grid, k=k_rot).flatten()

    # Otočení vzdáleností od zdí
    wd = wall_dists
    r_wd = wd[:]
    if dir_idx == 0: r_wd = [wd[0], wd[1], wd[2], wd[3]] # UP
    elif dir_idx == 1: r_wd = [wd[1], wd[0], wd[3], wd[2]] # DOWN
    elif dir_idx == 2: r_wd = [wd[2], wd[3], wd[1], wd[0]] # LEFT
    elif dir_idx == 3: r_wd = [wd[3], wd[2], wd[0], wd[1]] # RIGHT

    return np.concatenate([
        rotated_grid,
        state_dict['food_vector'],
        r_wd,
        [state_dict['snake_ratio']]
    ]).astype(np.float32)

def get_new_direction_relative(curr_dir, rel_action):
    # 0=Straight, 1=Left, 2=Right
    x, y = curr_dir
    if rel_action == 0: return (x, y)
    elif rel_action == 1: return (y, -x)
    elif rel_action == 2: return (-y, x)
    return curr_dir

def is_safe(head, direction, snake):
    nx, ny = head[0] + direction[0], head[1] + direction[1]
    if nx < 0 or nx >= GRID_W or ny < 0 or ny >= GRID_H: return False
    if [nx, ny] in snake: return False
    return True