import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

def load_data(nombres, ruta='../data/seismic-bumps.arff'):
    input_data, input_meta = arff.loadarff(ruta)
    df = pd.DataFrame(input_data)
    df.columns = nombres
    return df

# Nombres de columnas
nombres = [
    'seismic', 'seismoacoustic', 'shift', 'genergy', 'gpuls',
    'gdenergy', 'gdpuls', 'ghazard', 'nbumps', 'nbumps2',
    'nbumps3', 'nbumps4', 'nbumps5', 'nbumps6', 'nbumps7',
    'nbumps89', 'energy', 'maxenergy', 'clase'
]

# Cargar datos
df = load_data(nombres)

# Codificar variables categóricas
cat_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard', 'clase']
le = preprocessing.LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Separar características y variable objetivo
X = df.drop('clase', axis=1)
y = df['clase']

# Selección de características usando mutual_info_classif
importance = mutual_info_classif(X, y)
selected_features_idx = importance.argsort()[-5:]
selected_features = X.columns[selected_features_idx]
X_selected = X[selected_features]

# Estandarizar características seleccionadas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Encontrar número óptimo de clusters con método del codo
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), wcss)
plt.title('Método del Codo para número óptimo de clusters')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()

# Ajustar K-Means con número de clusters elegido (3)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Evaluar calidad de clusters con Silhouette Score
score = silhouette_score(X_scaled, y_kmeans)
print(f'Silhouette Score: {score:.3f}')

# Añadir clusters al DataFrame original
cluster = pd.DataFrame(y_kmeans + 1, columns=['cluster'])
df = pd.concat([df, cluster], axis=1)

# Calcular medias por cluster
kmeans_mean_cluster = df.groupby('cluster').mean().round(1)
print('Medias por cluster:')
print(kmeans_mean_cluster)

# Visualización con PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_kmeans, cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title='Clusters')
ax.add_artist(legend1)
ax.set_title('Visualización 3D de clusters con PCA')
plt.show()
