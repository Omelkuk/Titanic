import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import optuna
from optuna.trial import TrialState
import copy
import torch.optim as optim


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TitanicNN(nn.Module):
    def __init__(self, input_dim, cfg_nn, trial=None):
        super(TitanicNN, self).__init__()
        if trial:
            n_layers = trial.suggest_int("n_layers", *cfg_nn.tuning.n_layers)
            hidden_dim = trial.suggest_int("hidden_dim", *cfg_nn.tuning.hidden_dim)
            dropout_val = trial.suggest_float("dropout", *cfg_nn.tuning.dropout)
        else:
            n_layers = cfg_nn.best_params.n_layers
            hidden_dim = cfg_nn.best_params.hidden_dim
            dropout_val = cfg_nn.best_params.dropout

        layers = []
        curr_dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_val))
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def create_data_loader(X, y=None, batch_size=32):
    X_tensor = torch.tensor(X.values.astype("float32"))
    if y is not None:
        y_tensor = torch.tensor(y.values.astype("float32")).reshape(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset = TensorDataset(X_tensor)  
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def train_nn(model, train_loader, epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    stop_counter = 0
    
    for epoch in range(epochs):
       
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        
        scheduler.step()
        
       
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                val_loss += criterion(model(vx), vy).item()
        
        val_loss /= len(val_loader)
        
       
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            stop_counter = 0
        else:
            stop_counter += 1
            
        if epoch % 20 == 0 or stop_counter == 0:
            print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if stop_counter >= patience:
            print(f" Early stopping на эпохе {epoch}. Возвращаем лучшие веса.")
            break
            
   
    model.load_state_dict(best_weights)
    return model

def predict_probs(model, X_df):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_df.values.astype("float32"))
        return torch.sigmoid(model(X_t)).numpy()