# Cargando las librerias

import pandas as pd 
from scipy.io import arff
from sklearn import preprocessing
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 
from matplotlib.colors import ListedColormap 

# Cargando los datos
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
'class']


# Cargando las nuevas equitas, e imprimiendolas

df = load_data(nombres)
df.head()


# Transformando los datos categóricos de las etiquetas a valores entre 0 y 1

def preprocess_features(df, cols):
    """transform categorical features"""
    le = preprocessing.LabelEncoder()
    for clmn in cols:
        df[clmn] = le.fit_transform(df[clmn])
    
    return df


# Se comprueba que todas las columnas categóricas tienen más de una categoría

cat_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard', 'class']
df = preprocess_features(df, cat_cols)
for clmn in cat_cols:
    print(df[clmn].unique)


# Crear columnas que contienen un solo valor único

X = df.iloc[:, 0:17].values  
y = df.iloc[:, 18].values


# Dividiendo las X y las Y en 
# Datos de entrenamiento y datos de verificacion 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Desempeno de preprocesando 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 

# Funcion Applicando PCA para entrenamiento 
# Verifica el componente X 
pca = PCA(n_components = 2) 
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 
explained_variance = pca.explained_variance_ratio_ 

# Ajustando a la regresion Logistica para los datos de entrenamiento   
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train) 


# Prediciendo el analisis de los resultados usando  
# la funcion predictiva mediante la regresion logistica  
y_pred = classifier.predict(X_test) 

# Claculando la mtriz de confusion entre  
#  el conjunto de datos de analisis de Y y los valores predictivos. 
cm = confusion_matrix(y_test, y_pred) 

# Prediciendo el resultado del conjunto de valores 
# de entrenamiento a traves de scatter plot  
X_set, y_set = X_train, y_train 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                     stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1, 
                     stop = X_set[:, 1].max() + 1, step = 0.01)) 
  
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('yellow', 'white'))) 
  
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
  
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j) 
  
plt.title('Regresión Logistica(Conjunto de Entrenamiento)') 
plt.xlabel('PC1')  
plt.ylabel('PC2')  
plt.legend()   
  
# Mostrando scatter plot 
plt.show()

# Visualisando el conjunto de datos de analisis mediante scatter plot 
X_set, y_set = X_test, y_test 
  
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                     stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1, 
                     stop = X_set[:, 1].max() + 1, step = 0.01)) 
  
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), 
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
             cmap = ListedColormap(('yellow', 'white')))

plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
  
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j) 
  
# Etiquetas de la grafica (scatter plot) 
plt.title('Regresión Logistica (Conjunto Prueba)')  
plt.xlabel('PC1')  
plt.ylabel('PC2')  
plt.legend() 
  
# Mostrar scatter plot 
plt.show()