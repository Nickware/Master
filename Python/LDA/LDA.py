# Cargando las librerías
from pathlib import Path

from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_FILE = Path(__file__).resolve().parents[1] / 'data' / 'seismic-bumps.arff'
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f'No se encontró el dataset en: {DATA_FILE}. '
        'Verifica la estructura del repositorio.'
    )


# Función para cargar datos

def load_data(nombres, ruta=DATA_FILE):
    input_data, _ = arff.loadarff(str(ruta))
    df = pd.DataFrame(input_data)
    df.columns = nombres
    return df


# Nombres de las columnas
nombres = [
    'seismic',
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
    'clase'
]

# Cargar los datos
df = load_data(nombres)
print(df.head())

# Transformación de variables categóricas

def preprocess_features(df, cols):
    """Transforma columnas categóricas usando LabelEncoder."""
    le = preprocessing.LabelEncoder()
    for clmn in cols:
        df[clmn] = le.fit_transform(df[clmn])
    return df

cat_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard', 'clase']
df = preprocess_features(df, cat_cols)

# Verificar categorías
for clmn in cat_cols:
    print(f'{clmn}: {df[clmn].unique()}')

# Escalado de variables

def scale_features(df):
    """Escala todas las variables al rango [-1, 1]."""
    colnames = df.columns
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
    df = pd.DataFrame(scaler.fit_transform(df))
    df.columns = colnames
    return df

# Se evita que variables con mayor escala dominen el entrenamiento

df = scale_features(df)


def split_data(df, features):
    X = df[features]
    Y = df['clase']
    return train_test_split(X, Y, stratify=Y)


def train_model(x_train, y_train, scoring='recall'):
    svc = SVC(class_weight='balanced', probability=False)
    tuning_parameters = [{
        'kernel': ['rbf'],
        'gamma': [2**x for x in range(-10, 5)],
        'C': [2**x for x in range(-10, 5)]
    }]
    clf = GridSearchCV(estimator=svc, param_grid=tuning_parameters, scoring=scoring, cv=10)
    clf.fit(x_train, y_train)
    return clf


def sensitivity(mat):
    """Calcula la sensibilidad a partir de la matriz de confusión."""
    tp = mat[1][1]
    fn = mat[1][0]
    try:
        return (1.0 * tp) / (tp + fn)
    except ZeroDivisionError:
        return None


def specificity(mat):
    """Calcula la especificidad a partir de la matriz de confusión."""
    tn = mat[0][0]
    fp = mat[0][1]
    try:
        return (1.0 * tn) / (tn + fp)
    except ZeroDivisionError:
        return None


features = [clmn for clmn in df.columns if clmn != 'clase']
x_train, x_test, y_train, y_test = split_data(df, features)

clf = train_model(x_train, y_train, scoring='recall')
y_pred = clf.predict(x_test)

print('Optimizando la grilla de búsqueda de scores')
print('Precisión: {0}'.format(accuracy_score(y_test, y_pred)))
print('Sensibilidad: {0}'.format(sensitivity(confusion_matrix(y_test, y_pred))))
print('Especificidad: {0}'.format(specificity(confusion_matrix(y_test, y_pred))))

clf = train_model(x_train, y_train, scoring='roc_auc')
y_pred = clf.predict(x_test)

print('Este cálculo puede demorar según la máquina.')
print('Área bajo la curva ROC optimizada por la grilla de scores')
print('Accuracy: {0}'.format(accuracy_score(y_test, y_pred)))
print('Sensibilidad: {0}'.format(sensitivity(confusion_matrix(y_test, y_pred))))
print('Especificidad: {0}'.format(specificity(confusion_matrix(y_test, y_pred))))