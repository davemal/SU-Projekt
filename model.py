# model.py
import torch
import torch.nn as nn

class MultiTaskSequenceTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, predict_steps, max_len):
        super(MultiTaskSequenceTransformer, self).__init__()
        self.predict_steps = predict_steps
        
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, model_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True, dropout=0.1, dim_feedforward=256
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_action = nn.Linear(model_dim, predict_steps * 3)
        
        self.trap_head = nn.Sequential(
            nn.Linear(model_dim, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        last_hidden = x[:, -1, :] 
        
        actions = self.fc_action(last_hidden)
        actions = actions.view(-1, self.predict_steps, 3)
        trap = self.trap_head(last_hidden)
        return actions, trap