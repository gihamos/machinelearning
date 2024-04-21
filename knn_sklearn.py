# Importation des bibliothèques nécessaires pour le traitement des données, la réduction de dimension, et la visualisation.
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from tqdm import tqdm

# Charger les données Kanji et leurs étiquettes à partir de fichiers CSV.
data = np.loadtxt("kanji_train_data.csv", delimiter=",")  # Chargement des données d'entraînement
target = np.loadtxt("kanji_train_target.csv")  # Chargement des étiquettes

# Définir le nombre de Kanji à visualiser avec UMAP.
nb_kanji_plot = 1000

# Réduction de la dimensionnalité des données à 2 dimensions avec UMAP pour la visualisation.
reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean')  # Initialisation de l'objet UMAP
data_embedded = reducer.fit_transform(data[:nb_kanji_plot])  # Transformation des données

print(data_embedded.shape)
print(target.shape)

# Visualisation des données réduites, colorées par catégorie.
for i in tqdm(range(20)):  # Itération sur chaque catégorie, en assumant 20 catégories différentes
    plt.scatter(data_embedded[target[:nb_kanji_plot] == i, 0],
                data_embedded[target[:nb_kanji_plot] == i, 1])  # Tracé des points par catégorie

plt.show()  # Affichage du graphique

# Étape 1: Division des données en ensembles d'entraînement et de test.
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Étape 2: Application de l'algorithme k-NN avec k initial de 3.
knn = KNeighborsClassifier(n_neighbors=3)  # Initialisation du classificateur k-NN
knn.fit(X_train, y_train)  # Entraînement du modèle
predictions = knn.predict(X_test)  # Prédiction sur l'ensemble de test

# Calcul et affichage de la précision initiale.
initial_accuracy = accuracy_score(y_test, predictions)
print(f'Précision initiale avec k=3: {initial_accuracy}')

# Étape 3: Recherche du meilleur k en termes de précision.
k_values = range(1, 26)  # Définition de la plage de valeurs de k à tester
accuracies = []  # Initialisation d'une liste pour stocker les précisions

for k in tqdm(k_values):  # Itération sur chaque valeur de k
    knn = KNeighborsClassifier(n_neighbors=k)  # Nouvel objet k-NN pour chaque k
    knn.fit(X_train, y_train)  # Entraînement du modèle
    predictions = knn.predict(X_test)  # Prédiction sur l'ensemble de test
    accuracy = accuracy_score(y_test, predictions)  # Calcul de la précision
    accuracies.append(accuracy)  # Ajout de la précision à la liste

# Visualisation de la précision en fonction de k.
plt.plot(k_values, accuracies)
plt.xlabel('Nombre de voisins k')
plt.ylabel('Précision')
plt.title('Précision en fonction de k')
plt.show()

# Détermination du meilleur k et affichage de sa précision.
best_k = k_values[accuracies.index(max(accuracies))]
best_accuracy = max(accuracies)
print(f'Meilleur k: {best_k} avec une précision de: {best_accuracy}')

# Chargement des données de test pour la prédiction finale.
test_data = np.loadtxt('kanji_test_data.csv', delimiter=',')

# Création d'un nouveau classificateur k-NN avec le meilleur k trouvé.
knn2 = KNeighborsClassifier(n_neighbors=best_k)
knn2.fit(data, target)  # Entraînement du modèle sur l'ensemble des données

# Prédiction des catégories sur les nouvelles données de test.
test_predictions = knn2.predict(test_data)

print(test_predictions)

# Définition du chemin pour sauvegarder les prédictions et enregistrement dans un fichier CSV.
file_sauv = 'sauv/kanji_test_predictions.csv'
np.savetxt(file_sauv, test_predictions, delimiter=',', fmt='%d')

