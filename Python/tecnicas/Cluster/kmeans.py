# Importando modulos

import matplotlib.pyplot as plt
import pandas as pd 
from scipy.io import arff
from sklearn import preprocessing
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Cargando la base de datos

def load_data(nombres):
    data = '../data/seismic-bumps.arff'
    input_data, input_meta = arff.loadarff(data)
    df = pd.DataFrame(input_data)
    df.columns = nombres
    return df

# Rescribiendo las etiquetas
    
nombres = ['seismic',  
'seismoacoustic', 
'shift', 
'genergy', 
'gpuls', 
'gdenergy', 
'gdpuls', 
'ghazard',
'nbumps',
'nbumps2',
'nbumps3',
'nbumps4',
'nbumps5',
'nbumps6',
'nbumps7',
'nbumps89',
'energy',
'maxenergy',
'clase']

# Cargando las nuevas equitas, e imprimiendolas
df = load_data(nombres)
df.describe()


def preprocess_features(df, cols):
    """transform categorical features"""
    le = preprocessing.LabelEncoder()
    for clmn in cols:
        df[clmn] = le.fit_transform(df[clmn])
    
    return df

cat_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard', 'clase']
df = preprocess_features(df, cat_cols)
for clmn in cat_cols:
    print(df[clmn].unique)
    
X = df.iloc[:, 0:17].values  
y = df.iloc[:, 18].values
    
# Estandarizando los datos a una distribucion normal
df_standardized = preprocessing.scale(df)
df_standardized = pd.DataFrame(df_standardized)

# Encontrando el numero de cluster apropiados
plt.figure(figsize=(10, 8))
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(df_standardized)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Metodo de Elbow')
plt.xlabel('Numero de cluster')
plt.ylabel('WCSS')
plt.show()

# Ajustando los datos mediante el algoritmo K-Means
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(df_standardized)

# Inicia el cluster en numero 1 en lugar de 0
y_kmeans1=y_kmeans
y_kmeans1=y_kmeans+1

# Nuevo dataframe llamado cluster
cluster = pd.DataFrame(y_kmeans1)

# Adicionando cluster a al conjunto de datos df
df['cluster'] = cluster

# Valor medio de los clusters
kmeans_mean_cluster = pd.DataFrame(round(df.groupby('cluster').mean(),1))
print('kmeans_mean_cluster: ', kmeans_mean_cluster)

kmeans = KMeans(n_clusters=3).fit(X)
centroids = kmeans.cluster_centers_
	
# Prediciendo los clusters
labels = kmeans.predict(X)
# Obteniendo los centros de los clusters
C = kmeans.cluster_centers_

# Graficando los clusters obtenidos 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
ax.scatter(C[:, 0], C[:, 1], C[:, 2])
plt.show()