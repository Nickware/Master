# Cargando las librerias

from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix 


# Una pequeña funcion

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
df.head()


# Transformando los datos categóricos de las etiquetas a valores entre 0 y 1

def preprocess_features(df, cols):
    """Trasnformando las variables categoricas"""
    le = preprocessing.LabelEncoder()
    for clmn in cols:
        df[clmn] = le.fit_transform(df[clmn])
    
    return df

cat_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard', 'clase']
df = preprocess_features(df, cat_cols)


# Se comprueba que todas las columnas categóricas tienen más de una categoría

for clmn in cat_cols:
    print(df[clmn].unique())

# Crear columnas que contienen un solo valor único

def scale_features(df): 
    """Escalando todos los valores de las variables entre -1 y 1"""
    colnames = df.columns
    MinMaxScaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
    df = pd.DataFrame(MinMaxScaler.fit_transform(df))
    df.columns = colnames

    return df

#Escalando todos los valores de las variables entre -1 y 1 prevenimos la dominacion de variables en el modelo
df = scale_features(df)

def split_data(df, features):
    X = df[features]
    Y = df['clase'] 
    
    return train_test_split(X, Y, stratify=Y)

def train_model(x_train, y_train, scoring='recall'):
    svc = SVC(class_weight='balanced', probability=False)

    #Usando la validacion cruzada para encontrar buenos parametros
    tuning_parameters = [{'kernel': ['rbf'], 'gamma': [2**x for x in list(range(-10, 5))],
                         'C': [2**x for x in list(range(-10, 5))]}]

    clf = GridSearchCV(estimator=svc, param_grid=tuning_parameters, scoring=scoring, cv=10)
    clf.fit(x_train, y_train)
    
    return clf

def sensitivity(mat):
    """Usando la matriz de confusion para calcular la sensibilidad"""
    tp = mat[1][1]
    fn = mat[1][0]
    try:
        s = (1.0*tp) / (tp + fn)
    except:
        s = None
        
    return s

def specificity(mat):
    """Usando la matriz de confusión para calcular especificidad"""
    tn = mat[0][0]
    fp = mat[0][1]
    try:
        sp = (1.0*tn) / (tn + fp)
    except:
        sp = None
    
    return sp

features = [clmn for clmn in df.columns if clmn not in 'clase']
x_train, x_test, y_train, y_test = split_data(df, features)

clf = train_model(x_train, y_train, scoring='recall')
y_pred = clf.predict(x_test)

print("Optimizando la grilla de busqueda de scores")
print("Precisión : {0}",format(accuracy_score(y_test, y_pred)))
print("Sensibilidad : {0}",format(sensitivity(confusion_matrix(y_test, y_pred))))
print("Especificidad: {0}",format(specificity(confusion_matrix(y_test, y_pred))))

clf = train_model(x_train, y_train, scoring='roc_auc')
y_pred = clf.predict(x_test)

print ("Este calculo puede demorar. Depende de las especificaciones tecnicas de la maquina")
print ("Testiado en Debian 9.0 x86_64")
print ("Area dentro modelo ROC optimizada por la grilla de scores")
print("Accuracy : {0}",format(accuracy_score(y_test, y_pred)))
print("Sensibilidad : {0}",format(sensitivity(confusion_matrix(y_test, y_pred))))
print("Especificidad: {0}",format(specificity(confusion_matrix(y_test, y_pred))))