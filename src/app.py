import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

try:
    with open('./data/raw/wbm.dat', 'rb') as file:
        data = pickle.load(file)
except FileNotFoundError:
    data = 'https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv'
    data = pd.read_csv(data)
    data.drop_duplicates(inplace=True)
    with open('./data/raw/wbm.dat', 'wb') as file:
        pickle.dump(data, file)
finally:
    print(data.head(10))

x = data.drop(columns='Outcome')
y = data['Outcome']

corr = data.corr()
refined_x = corr.loc[corr['Outcome'] > .2]
refined_x = data[refined_x.index.to_list()]
refined_x.drop(columns='Outcome', inplace=True)

xtr, xte, ytr, yte = train_test_split(x, y, train_size=0.2, random_state=42)
rxtr, rxte, rytr, ryte = train_test_split(refined_x, y, train_size=0.2, random_state=42)

model1 = RandomForestClassifier(random_state=42)
model2 = RandomForestClassifier(random_state=42)
models = [model1, model2]
datasets = [[xtr, xte], [rxtr, rxte]]

for i in range(len(models)):
    model = models[i]
    data = datasets[i]
    model.fit(data[0], ytr)
    pred = model.predict(data[1])
    acc_check = accuracy_score(pred, yte)
    cm = confusion_matrix(pred, yte)
    print(f'Model {i} accuracy: {acc_check}')
    sns.heatmap(cm, annot=True, cbar=True, fmt='.2f')
    plt.show()

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

for i in range(len(models)):
    model = models[i]
    data = datasets[i]
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid.fit(data[0], ytr)
    print(
        f'Grid Search Model {i}:\n\tModel {i} Best Parameters: {grid.best_params_}\n\tModel {i} Best Accuracy: {grid.best_score_}')
    mod = grid.best_estimator_
    pred = mod.predict(data[1])
    acc_check = accuracy_score(pred, yte)
    cm = confusion_matrix(pred, yte)
    print(f'Model {i} Best Estimator Prediction Accuracy: {acc_check}')
    sns.heatmap(cm, annot=True, cbar=True, fmt='.2f')
    plt.show()

with open('./models/hgml.dat', 'wb') as file:
    pickle.dump(model, file)