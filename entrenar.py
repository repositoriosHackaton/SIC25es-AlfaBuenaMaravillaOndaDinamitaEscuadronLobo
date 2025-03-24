import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import itertools


from clases import CLASES



class GestosDataset(Dataset):
    
    def __init__(self, ruta_base="datos/procesados"):
        super().__init__()
        self.ruta_base = ruta_base
        
        self.clases = CLASES
        
        self.archivos = []
        self.labels = []
        
  
        for idx_clase, clase in enumerate(self.clases):
            carpeta_clase = os.path.join(ruta_base, clase)
            npys = glob.glob(os.path.join(carpeta_clase, "*.npy"))
            for npy_path in npys:
                self.archivos.append(npy_path)
                self.labels.append(idx_clase)  
        
    def __len__(self):
        return len(self.archivos)
    
    def __getitem__(self, idx):
        npy_path = self.archivos[idx]
        label = self.labels[idx]
        data = np.load(npy_path, allow_pickle=True)
        
        
        secuencia = []
        for frame_dict in data:
            cara = frame_dict["face"]
            pose = frame_dict["pose"]
            mano_izq = np.array(frame_dict["left_hand"]).flatten()
            mano_der = np.array(frame_dict["right_hand"]).flatten()
            frame_vector = np.concatenate([cara, pose, mano_izq, mano_der], axis=0)
            secuencia.append(frame_vector)
        
        secuencia = np.array(secuencia, dtype=np.float32)
        return secuencia, label

class ModeloLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(ModeloLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def graficar_matriz_confusion(y_true, y_pred, clases, titulo="Matriz de Confusión"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(titulo)
    plt.colorbar()
    tick_marks = np.arange(len(clases))
    plt.xticks(tick_marks, clases, rotation=45)
    plt.yticks(tick_marks, clases)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Clase real')
    plt.xlabel('Predicción')

def entrenar_modelo(modelo, device, train_loader, optimizer, criterion):
    modelo.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    for secuencias, labels in train_loader:
        secuencias = secuencias.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = modelo(secuencias)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * secuencias.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    avg_loss = epoch_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc

def validar_modelo(modelo, device, val_loader, criterion):
    modelo.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for secuencias, labels in val_loader:
            secuencias = secuencias.to(device)
            labels = labels.to(device)
            outputs = modelo(secuencias)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item() * secuencias.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    avg_loss = epoch_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc, all_labels, all_preds

def main():
    
    lr = 1e-3
    batch_size = 128
    epochs = 30
    hidden_size = 256
    num_layers = 2
    dropout = 0.5
    porcentaje_val = 0.2  

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {dispositivo}")

    
    dataset = GestosDataset(ruta_base="datos/procesados")
    dataset.clases = CLASES
    num_clases = len(dataset.clases)
    print(f"Clases encontradas: {dataset.clases}")
    print(f"Número total de muestras: {len(dataset)}")

   
    from sklearn.model_selection import StratifiedShuffleSplit
    labels = np.array(dataset.labels)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=porcentaje_val, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  
    modelo = ModeloLSTM(
        input_size=132,
        hidden_size=hidden_size,
        output_size=num_clases,
        num_layers=num_layers,
        dropout=dropout
    ).to(dispositivo)

    
    optimizer = optim.Adam(modelo.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

  
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    historial_entrenamiento = {
        "loss_train": [],
        "acc_train": [],
        "loss_val": [],
        "acc_val": [],
        "lr": []
    }

    for epoch in range(1, epochs+1):
        current_lr = optimizer.param_groups[0]["lr"]
        loss_train, acc_train = entrenar_modelo(modelo, dispositivo, train_loader, optimizer, criterion)
        loss_val, acc_val, y_true, y_pred = validar_modelo(modelo, dispositivo, val_loader, criterion)
        historial_entrenamiento["loss_train"].append(loss_train)
        historial_entrenamiento["acc_train"].append(acc_train)
        historial_entrenamiento["loss_val"].append(loss_val)
        historial_entrenamiento["acc_val"].append(acc_val)
        historial_entrenamiento["lr"].append(current_lr)
        print(f"Epoch [{epoch}/{epochs}] LR: {current_lr:.6f} | Loss_Train: {loss_train:.4f} Acc_Train: {acc_train:.4f} | Loss_Val: {loss_val:.4f} Acc_Val: {acc_val:.4f}")
        scheduler.step()

    os.makedirs("modelos", exist_ok=True)
    ruta_modelo = "modelos/modelo_lstm.pth"
    torch.save(modelo.state_dict(), ruta_modelo)
    print(f"Modelo guardado en: {ruta_modelo}")

    
    plt.figure(figsize=(8, 5))
    plt.plot(historial_entrenamiento["loss_train"], label="Pérdida Entrenamiento")
    plt.plot(historial_entrenamiento["loss_val"], label="Pérdida Validación")
    plt.title("Pérdida: Entrenamiento vs Validación")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    
    plt.figure(figsize=(8, 5))
    plt.plot(historial_entrenamiento["acc_train"], label="Exactitud Entrenamiento")
    plt.plot(historial_entrenamiento["acc_val"], label="Exactitud Validación")
    plt.title("Exactitud: Entrenamiento vs Validación")
    plt.xlabel("Épocas")
    plt.ylabel("Exactitud")
    plt.legend()
    plt.grid(True)
    plt.show()

   
    plt.figure(figsize=(6, 6))
    graficar_matriz_confusion(y_true, y_pred, dataset.clases, titulo="Matriz de Confusión (Validación)")
    plt.show()

    
    plt.figure(figsize=(8, 5))
    plt.plot(historial_entrenamiento["lr"], label="Tasa de Aprendizaje (LR)")
    plt.title("Evolución de la Tasa de Aprendizaje")
    plt.xlabel("Épocas")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
