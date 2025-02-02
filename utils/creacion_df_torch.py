import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch.nn as nn

# Crear un DataFrame a partir de las imágenes en el directorio
def crear_dataframe_binario(dataset_dir, filtro_cancer = 'all'):
    data = {'ruta': [], 'etiqueta': [], 'tipo_cancer': []}
    for neoplasia in ['ACA', 'SCC', 'benigno']:
        etiqueta = 'maligno' if neoplasia in ['ACA', 'SCC'] else 'benigno'
        neoplasia_dir = os.path.join(dataset_dir, neoplasia)
        for filename in os.listdir(neoplasia_dir):
            ruta_completa = os.path.join(neoplasia_dir, filename)

            #Definimos el tipo de cancer por si quisieramos entrenarlo con algun tipo particular
            parts = filename.split('_')
            # Determinar el tipo de cáncer y la etiqueta basado en el nombre del archivo
            if len(parts) > 2 and parts[0] == 'oral':
                tipo_cancer = parts[0]  # 'oral'
            elif len(parts) == 3:
                tipo_cancer, neoplasia_foto, _ = parts  # Formato antiguo
            else:
                print(f"Archivo {filename} ignorado por tener un formato incorrecto.")
                continue

            data['ruta'].append(ruta_completa)
            data['etiqueta'].append(etiqueta)
            data['tipo_cancer'].append(tipo_cancer)
            
    data = pd.DataFrame(data)
    if filtro_cancer != 'all':
        data = data[data['tipo_cancer']==filtro_cancer]

    return data



# Clase para cargar imágenes y etiquetas desde DataFrame
class ImageDatasetBin(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['ruta']
        label = 1 if self.df.iloc[index]['etiqueta'] == 'maligno' else 0
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Función para crear un DataFrame desde el directorio del dataset
def crear_dataframe_multiclase(dataset_dir, filtro_cancer = 'all'):
    data = {'ruta': [], 'etiqueta': [], 'tipo_cancer': []}
    for categoria in os.listdir(dataset_dir):  # Asume que cada subcarpeta es una categoría
        categoria_dir = os.path.join(dataset_dir, categoria)
        for filename in os.listdir(categoria_dir):
            parts = filename.split('_')
            
            # Determinar el tipo de cáncer y la etiqueta basado en el nombre del archivo
            if len(parts) > 2 and parts[0] == 'oral':
                tipo_cancer = parts[0]  # 'oral'
                etiqueta = parts[1]     # 'benigno' o 'maligno'
            elif len(parts) == 3:
                tipo_cancer, etiqueta, _ = parts  # Formato antiguo
            else:
                print(f"Archivo {filename} ignorado por tener un formato incorrecto.")
                continue

            ruta_completa = os.path.join(categoria_dir, filename)
            data['ruta'].append(ruta_completa)
            data['etiqueta'].append(etiqueta)
            data['tipo_cancer'].append(tipo_cancer)
    if filtro_cancer != 'all':
        data = data[data['tipo_cancer'] == filtro_cancer]
        
    return pd.DataFrame(data)


# Clase del dataset para cargar imágenes y etiquetas desde el DataFrame
class ImageDatasetMul(Dataset):
    def __init__(self, dataframe, transform=None, class_to_idx=None):
        self.df = dataframe
        self.transform = transform
        self.class_to_idx = class_to_idx or {v: k for k, v in enumerate(sorted(dataframe['etiqueta'].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df.iloc[index]['ruta']
        label = self.class_to_idx[self.df.iloc[index]['etiqueta']]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    


#_______________________________________________ACA SE DEFINEN LAS FUNCIONES QUE SIRVEN PARA AMBOS DF_______________________________________________

def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def prepare_data_loaders(df, batch_size=20, m_type = 'bin'):
    # Crear una columna combinada para estratificar
    df['Combinaciones'] = df['etiqueta'].astype(str) + "_" + df['tipo_cancer'].astype(str)


    train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=46, stratify=df['Combinaciones'])
    val_df, test_df = train_test_split(test_val_df, test_size=0.33, random_state=46, stratify=test_val_df['Combinaciones'])
    print('test_df:')
    print(test_df)

    if m_type == 'bin':
        train_ds = ImageDatasetBin(train_df, transform=get_transforms())
        val_ds = ImageDatasetBin(val_df, transform=get_transforms())
        test_ds = ImageDatasetBin(test_df, transform=get_transforms())

    else:
        train_ds = ImageDatasetMul(train_df, transform=get_transforms())
        val_ds = ImageDatasetMul(val_df, transform=get_transforms())
        test_ds = ImageDatasetMul(test_df, transform=get_transforms())


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



def prepare_data_loaders_tipos(df,test_df ,batch_size=20, m_type = 'bin'):
    # Crear una columna combinada para estratificar
    df['Combinaciones'] = df['etiqueta'].astype(str) + "_" + df['tipo_cancer'].astype(str)


    train_df, val_df = train_test_split(df, test_size=0.3, random_state=46, stratify=df['Combinaciones'])
    print('test_df:')
    print(test_df)

    if m_type == 'bin':
        train_ds = ImageDatasetBin(train_df, transform=get_transforms())
        val_ds = ImageDatasetBin(val_df, transform=get_transforms())
        test_ds = ImageDatasetBin(test_df, transform=get_transforms())

    else:
        train_ds = ImageDatasetMul(train_df, transform=get_transforms())
        val_ds = ImageDatasetMul(val_df, transform=get_transforms())
        test_ds = ImageDatasetMul(test_df, transform=get_transforms())


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader




def evaluate_model(device, model, test_loader):
    # Evaluación en conjunto de prueba
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')
    return test_accuracy

def evaluate_model_with_metrics(device, model, val_loader, test_loader, criterion=nn.NLLLoss()):
    model.eval()  # Cambiar el modelo a modo evaluación
    
    def evaluate(loader):
        y_true = []
        y_pred = []
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # Predicciones
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                # Cálculo de la pérdida
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

        # Convertir listas a arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calcular métricas
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='binary')
        precision = precision_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)

        # Calcular pérdida promedio
        avg_loss = total_loss / len(loader.dataset)

        return accuracy, recall, precision, f1, cm, avg_loss

    # Evaluar en conjunto de validación y conjunto de prueba
    val_metrics = evaluate(val_loader)
    test_metrics = evaluate(test_loader)

    # Imprimir métricas
    print(f"Validation Metrics:")
    print(f"  Accuracy: {val_metrics[0]:.4f}")
    print(f"  Recall: {val_metrics[1]:.4f}")
    print(f"  Precision: {val_metrics[2]:.4f}")
    print(f"  F1 Score: {val_metrics[3]:.4f}")
    print(f"  Confusion Matrix:\n{val_metrics[4]}")
    print(f"  Loss (avg): {val_metrics[5]:.4f}\n")

    print(f"Test Metrics:")
    print(f"  Accuracy: {test_metrics[0]:.4f}")
    print(f"  Recall: {test_metrics[1]:.4f}")
    print(f"  Precision: {test_metrics[2]:.4f}")
    print(f"  F1 Score: {test_metrics[3]:.4f}")
    print(f"  Confusion Matrix:\n{test_metrics[4]}")
    print(f"  Loss (avg): {test_metrics[5]:.4f}")

    return val_metrics, test_metrics

'''def evaluate_model_with_metrics(device, model, val_loader, test_loader, criterion=nn.NLLLoss()):
    model.eval()  # Cambiar el modelo a modo evaluación
    
    def evaluate(loader):
        y_true = []
        y_pred = []
        total_loss = 0.0
        total_samples = 0
        loss_list = []  # Lista para almacenar las pérdidas por batch
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                # Cálculo de la pérdida
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)  # Multiplicar por el tamaño del batch
                total_samples += labels.size(0)
                
                # Guardar la pérdida del batch
                loss_list.append(loss.item())
                
                # Predicciones
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        # Calcular métricas
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='binary')
        precision = precision_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)
        
        # Calcular la pérdida promedio
        avg_loss = total_loss / total_samples
        
        return accuracy, recall, precision, f1, cm, avg_loss, loss_list  # Retornar los 7 valores: las métricas y las pérdidas
    
    # Evaluar en conjunto de validación y conjunto de prueba
    val_metrics = evaluate(val_loader)
    test_metrics = evaluate(test_loader)
    
    # Imprimir métricas y pérdidas por batch
    print(f'Validation Accuracy: {val_metrics[0]:.4f}')
    print(f'Validation Loss (avg): {val_metrics[5]:.4f}')
    print(f'Validation Loss (list): {val_metrics[6]}')  # Lista de pérdidas por batch
    
    print(f'Test Accuracy: {test_metrics[0]:.4f}')
    print(f'Test Loss (avg): {test_metrics[5]:.4f}')
    print(f'Test Loss (list): {test_metrics[6]}')  # Lista de pérdidas por batch
    
    return val_metrics, test_metrics'''

#_______________________________________________ACA SE DEFINEN LAS FUNCIONES QUE MUESTRA EL TRABAJO DE LA RED NEURONAL_______________________________________________

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()
        print(f"GradCAM initialized with model: {self.model}, target layer: {self.target_layer}")

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
            print(f"Forward hook activated. Activations shape: {self.activations.shape}")

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            print(f"Backward hook activated. Gradients shape: {self.gradients.shape}")

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
        print("Hooks registered for forward and backward passes.")

    def generate(self, input_image, class_idx=None):
        self.model.eval()
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        input_image.requires_grad = True
        print(f"Input image shape: {input_image.shape}")

        # Forward pass
        output = self.model(input_image)
        print(f"Model output shape: {output.shape}")

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        print(f"Class index selected: {class_idx}")

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()
        print(f"Gradients shape: {gradients.shape}, Activations shape: {activations.shape}")

        # Grad-CAM calculation
        weights = np.mean(gradients, axis=(2, 3))[0]
        print(f"Weights calculated from gradients: {weights}")

        grad_cam = np.zeros(activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights):
            grad_cam += w * activations[0, i, :, :]
        
        print(f"Initial grad_cam shape: {grad_cam.shape}")

        grad_cam = np.maximum(grad_cam, 0)
        grad_cam = cv2.resize(grad_cam, (input_image.shape[2], input_image.shape[3]))
        print(f"Resized grad_cam shape: {grad_cam.shape}")

        if np.max(grad_cam) != 0:
            grad_cam = grad_cam - np.min(grad_cam)
            grad_cam = grad_cam / np.max(grad_cam)
        else:
            grad_cam = np.zeros_like(grad_cam)

        return grad_cam


def show_cam_on_image(img, mask):
    print(f"Original image shape: {img.shape}, CAM mask shape: {mask.shape}")
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    print(f"Generated CAM shape: {cam.shape}")
    return np.uint8(255 * cam)


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    print(f"Loaded image shape (before transform): {np.array(image).shape}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    print(f"Transformed image shape (after transform): {image.shape}")
    
    return image
