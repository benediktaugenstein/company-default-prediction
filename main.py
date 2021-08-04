import pandas as pd

from numpy import mean

from collections import Counter

from matplotlib import pyplot

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics

from xgboost import XGBClassifier

from datetime import datetime

# Erfassung der Anfangszeit zur Laufzeitbestimmung am Ende des Prozesses
start_time=datetime.now()

# Import Datensatz
file = pd.ExcelFile('Data.xlsx')
file.sheet_names
df1=file.parse('ExcelSheetName')
dataset = df1
array = dataset.values

# Datensatz in X (Indikatoren) und Y (Ergebnisse: insolvent -> 1 oder nicht -> 0) aufteilen
# Im Falle der Analyse von 20 Indikatoren muss hier die 64 zu 20 geändert werden
X = array[:,0:64]
y = array[:,64]

# Datensatz skalieren
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
X = min_max_scaler.fit_transform(X)

# Algorithmen festlegen für Cross Validation
models = []
#models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))
models.append(('KNN', KNeighborsClassifier(n_neighbors=50)))
models.append(('MLP', MLPClassifier(solver='adam', hidden_layer_sizes=(64), max_iter=500, activation='relu', random_state=1))) #hier nur ein hidden layer, aber auch mehrere möglich (z.B. hidden_layer_sizes=(64, 40, 20))
models.append(('DTC (CART)', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))
models.append(('GBC (XGBoost)', XGBClassifier(use_label_encoder=False, objective="binary:logistic", eval_metric="logloss", tree_method="exact", scale_pos_weight=13.4))) #scale_pos_weight kann hier je nach Datensatz angepasst werden

# Leere arrays für Ergebnisse und Namen der Algorithmen
results = []
names = []

# Cross Validation für jeden Algorithmus, Ergebnisausgabe als AUC-Mittelwert (Fläche unter ROC-Kurve) sowie Standardabweichung der Ergebnisse
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc') # Alternative: scoring='accuracy'
    results.append(cv_results)
    names.append(name)
    print(name + ' - mittlerer AUC-Score: %.3f' % mean(cv_results) + ' - Standardabweichung: %.3f' % cv_results.std())

# Algorithmen vergleichen durch Boxplot (optional)
pyplot.boxplot(results, labels=names)
pyplot.title('Boxplot Vergleich der Algorithmen')
pyplot.ylabel('AUC')
pyplot.show()

# Besten Algorithmus auswählen zur Bestimmung der Feature Importance und erneuten Vorhersagen
model = XGBClassifier(use_label_encoder=False, objective="binary:logistic", eval_metric="logloss", tree_method="exact", scale_pos_weight= 13.4)

model.fit(X, y)

# Feature importances ausgeben sowie Aufbereitung als bar chart
# ticks: die einzelnen Positionen auf der X-Achse
# x_labels: die Beschriftungen für die einzelnen Positionen der X-Achse -> in diesem Fall die Indikatoren (X1, X2,... X64)
# Im Falle der Analyse von 20 Indikatoren range(64) zu range(20) ändern
ticks_array = []
x_labels = []

for i in range(64):
    ticks_array.append(i)

j=0

for i in range(64):
    indicator_name = 'X' + str(i+1)
    x_labels.append(indicator_name)
    print(model.feature_importances_[j])
    j=j+1

pyplot.figure(figsize=(15, 3.5)) # Breite, Höhe
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_, align='center', width=0.8, color='#001AFF')
pyplot.xticks(ticks=ticks_array, labels=x_labels, rotation=90)
pyplot.show()

# Datensatz aufteilen in Trainingsdaten und Validierungsdaten/Testdaten
"""
Zur Erstellung von Vorhersagen für neue Unternehmen müssen die Kennzahlen der Unternehmen ganz unten in den Datensatz eingefügt werden.
Die Anzahl der Unternehmen, für die eine Prognose erstellt werden soll kann dann als integer (anstatt float) in test_size angegeben werden.
Diese Kennzahl repräsentiert dann keinen Prozentsatz mehr, sondern die Anzahl der Unternehmen, für welche eine Prognose erstellt werden soll.
"""
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.1, shuffle=True)

# Modell nochmal komplett neu trainieren (nur mit Trainingsdaten)
model.fit(X_train, Y_train)

# Vorhersagen treffen für Validierungsdaten/Testdaten (mit Wahrscheinlichkeiten)
predictions = model.predict(X_validation)
probability = model.predict_proba(X_validation)

# AUC-Score
# pos_prob: Nur Wahrscheinlichkeiten für den Fall, dass Unternehmen insolvent sind (nur 1, nicht 0)
pos_prob = probability[:, 1]
auc_score = roc_auc_score(Y_validation, pos_prob)
print('Gradient Boosting (XGBoostClassifier) ROC AUC-Score für den Validierungsdatensatz: %.3f' % (auc_score))

# False positive rate, true positive rate und thresholds für ROC-Kurve (optional)
#fpr, tpr, thresholds = metrics.roc_curve(Y_validation, lr_probs)
#print(thresholds)

print("PREDICTIONS:")
print(Counter(predictions))

print("VALIDATION:")
print(Counter(Y_validation))

# Confusion Matrix Ausgabe
confu = confusion_matrix(Y_validation, predictions)
print(confu)

# Confusion Matrix als Grafik
confu_heatmap = sns.heatmap(confu, vmax=50, annot=True, cmap='Oranges', fmt='g')
pyplot.ylabel("Validierung")
pyplot.xlabel("Vorhersagen")
pyplot.show()

# Zähler für Vorhersagen
correct_predictions = 0
false_predictions = 0
total_predictions = 0

# Wenn prediction stimmt, dann correct_predictions + 1, ansonsten false_predictions +1
for x in range(len(predictions)):
    total_predictions = total_predictions + 1
    if predictions[x] == Y_validation[x]:
        correct_predictions = correct_predictions + 1
    else:
        false_predictions = false_predictions +1

# Output
print("Total number of predictions:")
print(total_predictions)
print("Number of correct predictions:")
print(correct_predictions)
print("Number of false predictions:")
print(false_predictions)
print("Percentage of correct predictions:")
print(correct_predictions/total_predictions*100)
print("Runtime:")
print(datetime.now() - start_time)
