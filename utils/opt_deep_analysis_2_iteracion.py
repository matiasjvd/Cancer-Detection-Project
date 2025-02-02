import torch
torch.cuda.memory_cached()
import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import optuna
import sys
sys.path.append("C:/Users/Matias/Desktop/Tesis/Tesis-Codes/utils")
import creacion_df_torch as qol
dataset_dir = r"C:/Users/Matias/Desktop/Tesis/Dataset_Neoplasias"
from sklearn.metrics import f1_score

class Net(nn.Module):
    def __init__(self, num_filters_list, kernel_size, dropout_rate, num_classes, num_fc_layers=1, input_size=256):
        super(Net, self).__init__()
        layers = []
        in_channels = 3
        for num_filters in num_filters_list:
            layers.append(nn.Conv2d(in_channels, num_filters, kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = num_filters

        self.features = nn.Sequential(*layers)
        
        output_size = input_size
        for _ in range(len(num_filters_list)):
            output_size = (output_size - kernel_size + 2 * 1) / 1 + 1
            output_size = output_size // 2

        output_features = int(output_size * output_size * num_filters_list[-1])

        fc_layers = []
        fc_layers.append(nn.Flatten())
        fc_layers.append(nn.Dropout(dropout_rate))
        fc_layers.append(nn.Linear(output_features, 512))
        
        for _ in range(num_fc_layers - 1):
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            fc_layers.append(nn.Linear(512, 512))
        
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(dropout_rate))
        fc_layers.append(nn.Linear(512, num_classes))
        
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def objective(trial, cancer='all'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = "C:/Users/Matias/Desktop/Tesis/Dataset_Neoplasias"
    df = qol.crear_dataframe_multiclase(dataset_dir, filtro_cancer=cancer)
    num_classes = df['etiqueta'].nunique()
    
    # Hyperparameters search space
    num_layers = trial.suggest_int('num_layers', 1, 6)  # Hasta 6 capas
    num_fc_layers = trial.suggest_int('num_fc_layers', 1, 3)
    num_filters_list = [trial.suggest_categorical(f'num_filters_layer{i}', [64, 128, 256]) for i in range(num_layers)]
    kernel_size = trial.suggest_int('kernel_size', 2, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [20, 30, 60])
    steps_per_epoch = trial.suggest_int('steps_per_epoch', 100, 200)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])

    # Prepare data loaders
    train_loader, val_loader, test_loader = qol.prepare_data_loaders(df, batch_size, m_type='bin')

    # Initialize the model, optimizer, and loss function
    model = Net(num_filters_list, kernel_size, dropout_rate, num_classes, num_fc_layers).to(device)
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(10):
        steps = 0
        for images, labels in train_loader:
            if steps >= steps_per_epoch:
                break
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            # Free GPU memory
            del images, labels, output
            torch.cuda.empty_cache()

    # Validation loop
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            # Free GPU memory
            del images, labels, outputs
            torch.cuda.empty_cache()

    f1 = f1_score(all_labels, all_predictions, average='weighted')
    return f1