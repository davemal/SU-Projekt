import pickle
import numpy as np
import os
from collections import deque
import config as c
import utils

INPUT_FILE = "snake_data.pkl"
OUTPUT_FILE = "snake_data_processed.pkl"

# Mapování rotace ID o 90 stupňů
ROT_90_MAP = {0: 2, 2: 1, 1: 3, 3: 0} 

def get_relative_action(curr_dir, next_dir):
    """
    0=Rovně, 1=Vlevo, 2=Vpravo
    """
    if curr_dir == next_dir: return 0
    
    dx, dy = curr_dir
    left_vec = (dy, -dx)
    right_vec = (-dy, dx)
    
    if next_dir == left_vec: return 1
    if next_dir == right_vec: return 2
    return 0 

def is_trapped(head, body_set):
    """
    Flood Fill detekce pasti (nedostatek prostoru).
    """
    start_node = tuple(head)
    if start_node in body_set: return True
    
    queue = deque([start_node])
    visited = {start_node}
    count = 0
    required_space = len(body_set) * 1.5 
    
    limit = c.GRID_W * c.GRID_H
    if required_space > limit: required_space = limit
    
    while queue:
        cx, cy = queue.popleft()
        count += 1
        if count >= required_space: 
            return False 
            
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < c.GRID_W and 0 <= ny < c.GRID_H:
                neighbor = (nx, ny)
                if neighbor not in visited and neighbor not in body_set:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
    return True 

def process():
    if not os.path.exists(INPUT_FILE):
        print(f"Chybí soubor {INPUT_FILE}.")
        return

    print(f"Načítám {INPUT_FILE}...")
    with open(INPUT_FILE, "rb") as f:
        raw_games = pickle.load(f)

    processed_games = []
    total_steps = 0
    traps_found = 0

    print(f"Zpracovávám {len(raw_games)} her...")

    for game in raw_games:
        steps = game['steps']
        final_score = game['final_score']
        
        # Ignorovat krátké hry
        if len(steps) < 10: continue

        # Rekonstrukce a čištění dat
        clean_steps = []
        
        for i in range(len(steps) - 1):
            curr_state, _ = steps[i]
            next_state, _ = steps[i+1]
            
            # Směry
            curr_dir_idx = np.argmax(curr_state['direction'])
            curr_dir = utils.DIR_VECS[curr_dir_idx]
            
            next_dir_idx = np.argmax(next_state['direction'])
            next_dir = utils.DIR_VECS[next_dir_idx]
            
            # Akce (relativní)
            rel_action = get_relative_action(curr_dir, next_dir)
            
            # Detekce pasti
            full_body = [tuple(p) for p in curr_state['full_body']]
            head = full_body[0]
            body_set = set(full_body[1:]) 
            
            trap = False
            if len(full_body) > 3:
                trap = is_trapped(head, body_set)
                if trap: traps_found += 1
            
            # Odstranění pomocných dat
            state_to_save = {k: v for k, v in curr_state.items() if k != 'full_body'}
            
            clean_steps.append((state_to_save, rel_action, trap))

        # Augmentace (4 rotace * 2 flipy = 8x data)
        for k in range(4): 
            for do_flip in [False, True]:
                aug_steps = []
                
                for step in clean_steps:
                    state, action, is_trap = step
                    
                    # Rotace Gridu
                    grid = np.rot90(state['grid'], k)
                    if do_flip: grid = np.fliplr(grid)
                    
                    # Rotace Vzdáleností
                    wd = state['wall_dists']
                    r_wd = [0.0]*4
                    indices = [0, 1, 2, 3] # UP, DOWN, LEFT, RIGHT
                    for _ in range(k):
                        indices = [ROT_90_MAP[x] for x in indices]
                    
                    for original_i, rotated_i in enumerate(indices):
                        r_wd[rotated_i] = wd[original_i]
                        
                    if do_flip: 
                        r_wd[2], r_wd[3] = r_wd[3], r_wd[2] 

                    # Rotace Vektoru jídla
                    fv = state['food_vector']
                    new_fv = (fv[0], -fv[1] if do_flip else fv[1])
                    
                    # Rotace Akce
                    new_action = action
                    if do_flip:
                        if action == 1: new_action = 2
                        elif action == 2: new_action = 1
                        
                    # Rotace Směr (OneHot)
                    curr_idx = np.argmax(state['direction'])
                    for _ in range(k): curr_idx = ROT_90_MAP[curr_idx]
                    if do_flip:
                        if curr_idx == 2: curr_idx = 3
                        elif curr_idx == 3: curr_idx = 2
                    
                    new_dir = [0]*4
                    new_dir[curr_idx] = 1
                    
                    new_state = {
                        "grid": grid,
                        "food_vector": new_fv,
                        "wall_dists": r_wd,
                        "direction": new_dir,
                        "snake_ratio": state['snake_ratio']
                    }
                    
                    aug_steps.append((new_state, new_action, is_trap))
                
                processed_games.append({
                    "steps": aug_steps,
                    "final_score": final_score
                })
                total_steps += len(aug_steps)

    print(f"✅ Hotovo! Zpracováno {len(processed_games)} her (včetně augmentací).")
    print(f"📈 Celkem kroků: {total_steps}")
    print(f"⚠️ Nalezeno pastí: {traps_found * 8} (vč. augmentace)")
    
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(processed_games, f)
    print(f"💾 Uloženo do {OUTPUT_FILE}")

if __name__ == "__main__":
    process()