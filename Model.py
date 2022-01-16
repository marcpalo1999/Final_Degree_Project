#Importing libraries
from libraries import *


#Reading train and test dataframes
X_train = pd.read_csv('/Users/marcpalomercadenas/Desktop/TFG/TFG/TrainTestMatrices/X_train.csv', index_col = 0)
X_test = pd.read_csv('/Users/marcpalomercadenas/Desktop/TFG/TFG/TrainTestMatrices/X_test.csv', index_col = 0)
y_train = pd.read_csv('/Users/marcpalomercadenas/Desktop/TFG/TFG/TrainTestMatrices/y_train.csv', index_col = 0)
y_test = pd.read_csv('/Users/marcpalomercadenas/Desktop/TFG/TFG/TrainTestMatrices/y_test.csv', index_col = 0)

# Create a classifier: a support vector classifier
SVM = svm.SVC(gamma='scale', probability= True) #, probability = True
SVM.fit(X_train, y_train)

# Create 2nd classifier: a Random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)

#Predictions
predicted_SVM = SVM.predict(X_test)
y_pred_proba_SVM =SVM.predict_proba(X_test)[:,-1]
predicted_rfc = rfc.predict(X_test)
y_pred_proba_rfc =rfc.predict_proba(X_test)[:,-1]

#------- MODEL PERFORMANCE-------

# Roc and Auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba_SVM)
roc_curve_plot = metrics.plot_roc_curve(SVM, X_test, y_test, alpha = 0.8)
ax = plt.gca()
rfc_disp = metrics.plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
plt.title('ROC curve')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate (positive label: 1)')


plt.savefig('ROC curve')
plt.show()


# Ploting metrics extracted from Confusion matrix and CM itserlffor SVM
print(
    f"Classification report for classifier {SVM}:\n"
    f"{metrics.classification_report(y_test, predicted_SVM)}\n"
)

# true digit values and the predicted digit values.
disp = ConfusionMatrixDisplay.from_predictions(y_test, predicted_SVM)
disp.figure_.suptitle("Confusion Matrix for SVM")
print(f"Confusion matrix for SVM:\n{disp.confusion_matrix}")

plt.savefig('ConfusionMatrix for SVM')
plt.show()


# Ploting metrics extracted from Confusion matrix and CM itserlffor rfc
print(
    f"Classification report for classifier {rfc}:\n"
    f"{metrics.classification_report(y_test, predicted_rfc)}\n"
)

# true digit values and the predicted digit values.
disp = ConfusionMatrixDisplay.from_predictions(y_test, predicted_rfc)
disp.figure_.suptitle("Confusion Matrix for rfc")
print(f"Confusion matrix for rfc:\n{disp.confusion_matrix}")

plt.savefig('ConfusionMatrix for rfc')
plt.show()


 # %%
 #---------Feature importance--------
feature_names = ['ECG_HR', 'NIBP_MEAN','PROPO_CE', 'REMI_CE']
importances = rfc.feature_importances_
plt.pie(importances, labels = feature_names)

 # %%
 #Saving our model
with open("SVM_model.pickle", "wb") as file:
    pickle.dump(SVM, file)

with open("RandomForest_model.pickle", "wb") as file:
    pickle.dump(rfc, file)

print('Models Saved')

# %%
##Comentaris
# -1r agafar propo i remi ce, començar a agafar dades des d'on comença a infundir propo ce i si algun pacient no en té cap no agafarlo directament FET
#Que les dades estiguin balancejades, igual num de dades abans i despres de l verbal (pel pairing i perque el model no fagi lu de la classificació gats gossos) FET
#Interpolació de valors que no hi són FET
#Tots els moments d'abans del verbal que siguin 0 i els altres 1. Comprovar fent debug que els frames que donen son correctes FET


#Preguntes:
#1- Com miro els valors dun dataframe al debug? FET
#2- He de normalitzar respecte als maxims de cada pacient o respecte als maxims totals? PENDENT
#3- Com pot ser que la confusion matrix i els parametres del model siguin perfectes? Vaig pensar que era pel que m'havies comentat de fer el split per pacient i no, perque ara al canviarho passa el mateix. FET

#Al ficar REMI també les dades se'm redueixen a la meitat.
#No hi ha molta disparitat entre valors a prediure? No
#GUardar els train i test a arxius per que no canviin els resultats cada cop FET

#Ficar els pacients sencers a train o test FET
#No entrar la variable a predir al model FET
#Fer servir el predict_proba per que em doni la probabilitat FET
#Fer un RF (amb random state = 123, ens diu la importancia de les variables) o un Edgybus classifier i ferli la ROC curve PENDENT
# Usar la PEP-8 
#Hyper parameter tunning -> buscar la millor AUC
#Enviar-li la momoria com a molt tard el 9 (deadline meu el 27)
#Enviarli uns grafics ROC per exemple i dun pacient per un altre exemple amb els events marcats
#Solucionar el problema de les poques dades PRIORITAT

#Validació Jaramillo:
    #Demanarli tot i mirarmho

