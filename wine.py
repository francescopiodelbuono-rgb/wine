
# Importazione delle librerie
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay


# 1. CARICAMENTO DEL DATASET WINE

wine = load_wine(as_frame=True)

df = wine.frame

# Separazione delle feature (X) e del target (y)
X = wine.data
y = wine.target

# Analisi iniziale del dataset
print("Dimensioni del dataset:")
print(df.shape)

print("\nPrime 5 osservazioni:")
print(df.head())

print("\nStatistiche descrittive:")
print(df.describe())

print("\nTipologie delle colonne:")
print(df.dtypes)

# 2. STANDARDIZZAZIONE DELLE FEATURE

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Verifica della standardizzazione
print("\nMedia delle feature standardizzate:")
print(pd.DataFrame(X_scaled, columns=X.columns).mean())

print("\nDeviazione standard delle feature standardizzate:")
print(pd.DataFrame(X_scaled, columns=X.columns).std())



# 3. PCA - RIDUZIONE DELLA DIMENSIONALITÀ

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# Varianza spiegata dalle due componenti
print("\nVarianza spiegata dalle componenti principali:")
print(pca.explained_variance_ratio_)

# Varianza spiegata complessiva
print("\nVarianza spiegata complessiva:")
print(pca.explained_variance_ratio_.sum())

# Dimensioni del dataset dopo PCA
print("\nDimensioni del dataset dopo PCA:")
print(X_pca.shape)

# Prime 5 osservazioni dopo PCA
print("\nPrime 5 osservazioni dopo PCA:")
print(X_pca[:5])


# 4. ADDESTRAMENTO E VALUTAZIONE DEI MODELLI

# Inizializzazione dei modelli
knn = KNeighborsClassifier(n_neighbors=5)

svm = SVC(kernel='linear')

# Addestramento KNN
knn.fit(X_pca, y)

# Predizione KNN
y_pred_knn = knn.predict(X_pca)

# Accuracy KNN
accuracy_knn = accuracy_score(y, y_pred_knn)

# Addestramento SVM
svm.fit(X_pca, y)

# Predizione SVM
y_pred_svm = svm.predict(X_pca)

# Accuracy SVM
accuracy_svm = accuracy_score(y, y_pred_svm)

# Stampa dei risultati
print("\nAccuracy Score dei modelli:")

print(f"KNN: {accuracy_knn * 100:.2f}%")

print(f"SVM: {accuracy_svm * 100:.2f}%")

print("Accuracy Score dei modelli:")
print(f"KNN: {accuracy_knn:.2%}")
print(f"SVM: {accuracy_svm:.2%}")

# 4 Analisi e visualizzazione

# Creazione della figura con due subplot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Ciclo sui due modelli
for ax, (nome, modello) in zip(
    axes,
    [("KNN", knn), ("SVM", svm)]
):

    # Visualizzazione dei confini decisionali
    DecisionBoundaryDisplay.from_estimator(
        modello,
        X_pca,
        response_method="predict",
        ax=ax,
        cmap=plt.cm.coolwarm,
        alpha=0.3
    )

    # Visualizzazione dei punti colorati per classe
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap=plt.cm.coolwarm,
        edgecolors="black",
        s=40
    )

    ax.set_title(f"Confini decisionali - {nome}")
    ax.set_xlabel("Prima componente principale (PC1)")
    ax.set_ylabel("Seconda componente principale (PC2)")

# Legenda delle classi
fig.colorbar(
    scatter,
    ax=axes,
    label="Classe reale del vino",
    ticks=[0, 1, 2]
)

plt.suptitle("Confronto KNN e SVM sul dataset Wine dopo PCA")
plt.tight_layout()
plt.show()