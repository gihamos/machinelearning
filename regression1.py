import pandas as pd
import numpy as np
from tqdm import tqdm  # Importe la bibliothèque tqdm pour afficher une barre de progression

def softmax(z):
    # Applique la fonction softmax pour calculer les probabilités à partir des scores logits z.
    # La soustraction de np.max(z) améliore la stabilité numérique.
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def compute_loss_and_gradients(X, y, W, b):
    # Calcule la perte (loss) et les gradients pour la mise à jour des poids W et du biais b
    # à partir des données d'entrée X, des labels y, des poids W et du biais b.
    num_examples = X.shape[0]
    # Calcul des scores logits et application de softmax pour obtenir les probabilités
    z = np.dot(X, W) + b
    probs = softmax(z)
    # Calcul de la log-perte (cross-entropy loss)
    correct_logprobs = -np.log(probs[range(num_examples), y])
    loss = np.sum(correct_logprobs) / num_examples
    # Calcul des gradients par rapport à z, puis par rapport à W et b
    dz = probs
    dz[range(num_examples), y] -= 1
    dz /= num_examples
    dW = np.dot(X.T, dz)
    db = np.sum(dz, axis=0, keepdims=True)
    return loss, dW, db

def train(X, y, num_classes, learning_rate=0.1, num_iterations=2000):
    # Entraîne le modèle sur les données X avec les labels y.
    # Initialise les poids et biais, et effectue la mise à jour pendant un nombre déterminé d'itérations.
    num_features = X.shape[1]
    W = np.zeros((num_features, num_classes))
    b = np.zeros((1, num_classes))
    costs = []  # Pour enregistrer l'évolution de la perte
    for i in tqdm(range(num_iterations)):
        loss, dW, db = compute_loss_and_gradients(X, y, W, b)
        # Mise à jour des poids et du biais avec le taux d'apprentissage
        W -= learning_rate * dW
        b -= learning_rate * db
        # Enregistrer la perte tous les 10 itérations
        if i % 10 == 0:
            costs.append(loss)
            print(f"Iteration {i} / {num_iterations}: Loss {loss}")
    
    return W, b, costs

def predict(X, W, b):
    # Prédit les classes pour les données X en utilisant les poids W et le biais b.
    z = np.dot(X, W) + b
    probs = softmax(z)
    return np.argmax(probs, axis=1)  # Retourne l'indice de la classe avec la probabilité la plus élevée

def calculate_accuracy(y_true, y_pred):
    """Calcule la précision, le nombre de prédictions correctes divisé par le nombre total de prédictions."""
    accuracy = np.mean(y_true == y_pred)
    return accuracy

# Charger les données d'entraînement
X_train = pd.read_csv('kanji_train_data.csv', header=None).to_numpy()
y_train = pd.read_csv('kanji_train_target.csv', header=None).squeeze().to_numpy()

# Charger les données de test
X_test = pd.read_csv('kanji_test_data.csv', header=None).to_numpy()

# Calculer la moyenne et l'écart type seulement à partir des données d'entraînement pour normaliser les données
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
# Éviter la division par zéro pour les caractéristiques constantes
std[std == 0] = 1

# Normaliser les données d'entraînement et de test
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

from sklearn.model_selection import train_test_split

# Diviser l'ensemble d'entraînement en nouveaux sous-ensembles d'entraînement et de validation
X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Entraîner le modèle et faire des prédictions sur l'ensemble de validation
W, b, couts = train(X_train_new, y_train_new, num_classes=20, learning_rate=0.1, num_iterations=100)
predictions_val = predict(X_val, W, b)
accuracy_val = calculate_accuracy(y_val, predictions_val)
print(f'Validation Accuracy: {accuracy_val}')

# Affichage de la courbe de coût
import matplotlib.pyplot as plt
plt.plot(couts)
plt.xlabel('Iterations (x10)')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations')
plt.show()
