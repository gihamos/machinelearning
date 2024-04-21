import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
# Charger les données d'entraînement
X_train = pd.read_csv('kanji_train_data.csv', header=None).to_numpy()
y_train = pd.read_csv('kanji_train_target.csv', header=None).squeeze().to_numpy()

# Charger les données de test
X_test = pd.read_csv('kanji_test_data.csv', header=None).to_numpy()

# Calculer la moyenne et l'écart type seulement à partir des données d'entraînement
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

# Éviter la division par zéro en remplaçant l'écart type de 0 par 1 (pour les caractéristiques constantes)
std[std == 0] = 1

# Normaliser les données d'entraînement et de test
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Diviser l'ensemble d'entraînement en nouveaux sous-ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Redéfinition des données pour correspondre à la forme attendue par un CNN
# Supposons que chaque entrée peut être restructurée en une image 64x64
X_train_reshaped = X_train.reshape(-1, 1, 64, 64)
X_val_reshaped = X_val.reshape(-1, 1, 64, 64)
X_test_reshaped = X_test.reshape(-1, 1, 64, 64)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Conversion en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val_reshaped, dtype=torch.float).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float).to(device)


# Création de datasets et de dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Définition du modèle CNN
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 1000)
        self.fc2 = nn.Linear(1000, 20)
    
    def forward(self, x):
        out = self.pool(self.relu(self.conv1(x)))
        out = self.pool(self.relu(self.conv2(out)))
        out = out.view(-1, 64 * 16 * 16)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

model = ConvNet().to(device)
# Perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement du modèle
num_epochs = 100
loss_values = []  # Initialisation de la liste pour stocker les valeurs de perte
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    loss_values.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Évaluation du modèle sur l'ensemble de validation pour calculer la précision
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Précision du modèle sur l\'ensemble de validation: {accuracy}%')

# Générer des prédictions pour l'ensemble de test complet
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predictions_test = torch.max(outputs, 1)
predictions_test_np = predictions_test.cpu().numpy()

# Sauvegarder les prédictions de test dans un fichier CSV
np.savetxt("kanji_test_predictions.csv", predictions_test_np, delimiter=",", fmt='%d')


# Affichage des graphiques de perte
plt.figure(figsize=(10, 5))
plt.plot(loss_values, label='Perte d\'entraînement')
plt.title('Perte pendant l\'entraînement')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.legend()
plt.show()