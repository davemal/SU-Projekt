
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import os

import config as c
from model import MultiTaskSequenceTransformer
import utils

# Konfigurace
DATA_FILE = "snake_data_processed.pkl"
MODEL_FILE = "snake_transformer.pth"
BATCH_SIZE = 256
EPOCHS = 50
LR = 0.0003

# Váha trestu za past (aby se model vyhýbal zavření)
TRAP_LOSS_WEIGHT = 2.0 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SnakeDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, "rb") as f:
            games = pickle.load(f)
            
        self.samples = []
        print("Dataset setup...")
        
        for game in games:
            steps = game['steps']
            if len(steps) < c.SEQ_LEN + c.PREDICT_STEPS:
                continue
                
            # Předpočítání
            vectors = [utils.prepare_vector(s[0]) for s in steps]
            actions = [s[1] for s in steps]
            traps = [1.0 if s[2] else 0.0 for s in steps]
            
            # Sliding Window
            # Vstup: Historie [0:SEQ_LEN]
            # Výstup: Akce [SEQ_LEN:PREDICT_STEPS]
            for i in range(len(steps) - c.SEQ_LEN - c.PREDICT_STEPS):
                seq_in = vectors[i : i + c.SEQ_LEN]
                
                target_idx = i + c.SEQ_LEN
                seq_out = actions[target_idx : target_idx + c.PREDICT_STEPS]
                
                trap_val = traps[target_idx - 1] 
                
                self.samples.append((
                    np.array(seq_in, dtype=np.float32),
                    np.array(seq_out, dtype=np.int64),
                    np.array([trap_val], dtype=np.float32)
                ))
                
        print(f"Vzorků: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train():
    if not os.path.exists(DATA_FILE):
        print(f"❌ Chybí {DATA_FILE}. Spusť process_data.py!")
        return

    # Data
    dataset = SnakeDataset(DATA_FILE)
    if len(dataset) == 0:
        print("Prázdný dataset.")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = MultiTaskSequenceTransformer(
        input_dim=c.INPUT_DIM,
        model_dim=c.MODEL_DIM,
        num_heads=c.NUM_HEADS,
        num_layers=c.NUM_LAYERS,
        predict_steps=c.PREDICT_STEPS,
        max_len=c.SEQ_LEN
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    crit_action = nn.CrossEntropyLoss()
    crit_trap = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))

    best_loss = float('inf')

    print(f"🚀 Start tréninku na {device}...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for x, y_action, y_trap in train_loader:
            x, y_action, y_trap = x.to(device), y_action.to(device), y_trap.to(device)
            
            optimizer.zero_grad()
            pred_actions, pred_trap = model(x)
            
            # Loss: Akce + Past
            loss_a = crit_action(pred_actions.view(-1, 3), y_action.view(-1))
            loss_t = crit_trap(pred_trap, y_trap)
            
            loss = loss_a + (TRAP_LOSS_WEIGHT * loss_t)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        
        # Validace
        model.eval()
        val_loss = 0
        correct_actions = 0
        total_actions = 0
        
        with torch.no_grad():
            for x, y_action, y_trap in val_loader:
                x, y_action, y_trap = x.to(device), y_action.to(device), y_trap.to(device)
                pred_actions, pred_trap = model(x)
                
                loss_a = crit_action(pred_actions.view(-1, 3), y_action.view(-1))
                loss_t = crit_trap(pred_trap, y_trap)
                val_loss += (loss_a + TRAP_LOSS_WEIGHT * loss_t).item()
                
                # Accuracy (první krok)
                p = torch.argmax(pred_actions[:, 0, :], dim=1)
                t = y_action[:, 0]
                correct_actions += (p == t).sum().item()
                total_actions += t.size(0)

        avg_val_loss = val_loss / len(val_loader)
        acc = (correct_actions / total_actions) * 100
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Action Acc: {acc:.1f}%")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_FILE)
            print("  💾 Model uložen.")

    print("Hotovo.")

if __name__ == "__main__":
    train()