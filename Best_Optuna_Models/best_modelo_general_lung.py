import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("C:/Users/Matias/Desktop/Tesis/Tesis-Codes/utils")
dataset_dir = r"C:/Users/Matias/Desktop/Tesis/Dataset_Neoplasias"

# Definición del modelo ajustado con los parámetros del Trial 34
class Net(nn.Module):
    def __init__(self, num_filters0, num_filters1, num_filters2, num_filters3, num_filters4, kernel_size, dropout_rate, num_classes, input_size=256):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_filters0, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters0, num_filters1, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters1, num_filters2, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters2, num_filters3, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters3, num_filters4, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        output_size = input_size
        for _ in range(5):  # Dado que tenemos cinco bloques de Conv + MaxPool
            output_size = (output_size - kernel_size + 2 * 1) / 1 + 1
            output_size = output_size // 2  # MaxPooling divide el tamaño por 2

        output_features = int(output_size * output_size * num_filters4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_features, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(device, train_loader, val_loader):
    num_classes = 2  # Número de clases de salida

    # Inicializar el modelo con los parámetros del Trial 34
    model = Net(
        num_filters0=64,  # Primer filtro del Trial 34
        num_filters1=32,  # Segundo filtro del Trial 34
        num_filters2=64,  # Tercer filtro del Trial 34
        num_filters3=128, # Cuarto filtro del Trial 34
        num_filters4=128, # Quinto filtro del Trial 34
        kernel_size=3,    # Tamaño del kernel
        dropout_rate=0.3048,  # Tasa de dropout del Trial 34
        num_classes=num_classes
    ).to(device)

    # Optimización y criterio de pérdida
    optimizer = optim.Adam(model.parameters(), lr=0.0003732)  # Learning rate del Trial 34
    criterion = nn.NLLLoss()

    # Listas para almacenar pérdidas y precisión
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # Entrenamiento del modelo
    for epoch in range(60):  # 60 épocas como en el Trial 34
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        steps = 0

        for images, labels in train_loader:
            if steps >= 200:  # steps_per_epoch (valor del Trial 34)
                break
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Cálculo de precisión durante el entrenamiento
            _, predicted_train = torch.max(output.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        epoch_train_loss = running_loss / steps
        epoch_train_accuracy = correct_train / total_train
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_accuracy)

        # Evaluación en conjunto de validación
        model.eval()
        val_running_loss = 0.0
        val_steps = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_output = model(val_images)
                val_loss = criterion(val_output, val_labels)
                val_running_loss += val_loss.item()
                val_steps += 1

                # Cálculo de precisión durante la validación
                _, predicted_val = torch.max(val_output.data, 1)
                total_val += val_labels.size(0)
                correct_val += (predicted_val == val_labels).sum().item()

        val_epoch_loss = val_running_loss / val_steps
        val_epoch_accuracy = correct_val / total_val
        val_loss_history.append(val_epoch_loss)
        val_acc_history.append(val_epoch_accuracy)

        # Mostrar progreso en cada época
        print(f'Epoch {epoch+1}: - accuracy: {epoch_train_accuracy:.4f} - loss: {epoch_train_loss:.4f} - val_accuracy: {val_epoch_accuracy:.4f} - val_loss: {val_epoch_loss:.4f}')

    # Guardar la historia de la pérdida en un archivo Excel
    df_loss = pd.DataFrame({
        "Training Loss": train_loss_history,
        "Validation Loss": val_loss_history,
        "Training Accuracy": train_acc_history,
        "Validation Accuracy": val_acc_history
    })
    df_loss.to_excel("C:/Users/Matias/Desktop/Tesis/BBDD_loss_pulmon.xlsx", index=False)

    # Evaluación final en conjunto de validación
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    print(f'Final Validation Accuracy: {val_accuracy:.4f}')
    return model
