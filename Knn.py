# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Charger les données et les étiquettes à partir de fichiers CSV
data = np.loadtxt("kanji_train_data.csv", delimiter=",")
target = np.loadtxt("kanji_train_target.csv")

# Normalisation des données d'entrainement pour améliorer les performances de l'algorithme
#data=(data-data.mean())/data.std()
# Cette ligne est commentée, signifiant qu'elle n'est pas exécutée. La normalisation est une étape commune de prétraitement.

# Fonction pour diviser les données en ensembles d'entraînement et de test
def split_data(data, target, test_size=0.8, random_state=90):
    # Fixe la graine du générateur aléatoire pour la reproductibilité des résultats
    np.random.seed(random_state)
    # Mélange aléatoirement les indices des données pour assurer une répartition aléatoire
    indices = np.random.permutation(data.shape[0])
    # Calcule l'indice de séparation basé sur le pourcentage de taille de test
    split_point = int(data.shape[0] * (1 - test_size))
    # Retourne les ensembles divisés d'entraînement et de test
    return (data[indices[:split_point]], data[indices[split_point:]],
            target[indices[:split_point]], target[indices[split_point:]])

def distance(X_train, x_test):
    # Calcule et retourne la distance euclidienne entre chaque échantillon de X_train et x_test
    return np.sqrt(np.sum((X_train - x_test) ** 2, axis=1))

def neighbors(X_train, y_train, x_test, k):
    # Identifie les k plus proches voisins de x_test parmi les échantillons de X_train
    distances = distance(X_train, x_test)
    indices = np.argsort(distances)
    # Retourne les étiquettes des k plus proches voisins
    return y_train[indices[:k]]

def prediction(X_train, y_train, X_test, k):
    # Prédit la classe pour chaque échantillon dans X_test en se basant sur les k plus proches voisins
    predictions = np.empty(len(X_test), dtype=int)
    for i, x_test in tqdm(enumerate(X_test)):
        nearest_labels = neighbors(X_train, y_train, x_test, k)
        # Détermine la classe majoritaire parmi les voisins
        predictions[i] = np.bincount(nearest_labels.astype(int)).argmax()
    return predictions

def accuracy(y_true, y_pred):
    # Calcule et retourne la précision comme le pourcentage de prédictions correctes
    return np.mean(y_true == y_pred)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = split_data(data, target)

# Recherche du meilleur k en testant différentes valeurs
k_values = range(1, 19, 10)
accuracies = []
for k in tqdm(k_values):
    y_pred = prediction(X_train, y_train, X_test, k)
    accuracies.append(accuracy(y_test, y_pred))

# Visualisation de la précision en fonction de k
plt.plot(k_values, accuracies)
plt.xlabel('Nombre de voisins k')
plt.ylabel('Précision')
plt.title('Précision en fonction de k')
plt.show()

# Affichage du meilleur k et de sa précision
best_k = k_values[np.argmax(accuracies)]
best_accuracy = np.max(accuracies)
print(f'Meilleur k: {best_k} avec une précision de: {best_accuracy}')

# Application du modèle au jeu de données de test et sauvegarde des prédictions
test_data = np.loadtxt('kanji_test_data.csv', delimiter=',')
test_predictions = prediction(data, target, test_data, best_k)
print(test_predictions)
np.savetxt('sauv/kanji_test_predictions.csv', test_predictions, delimiter=',', fmt='%d')
