#necessary imports
#exec(open("scripto.py").read())
from libraries import *


#Reading train and test dataframes
X_train = pd.read_csv('/Users/marcpalomercadenas/Desktop/TFG/TFG/TrainTestMatrices/X_train.csv', index_col = 0)
X_test = pd.read_csv('/Users/marcpalomercadenas/Desktop/TFG/TFG/TrainTestMatrices/X_test.csv', index_col = 0)
y_train = pd.read_csv('/Users/marcpalomercadenas/Desktop/TFG/TFG/TrainTestMatrices/y_train.csv', index_col = 0)
y_test = pd.read_csv('/Users/marcpalomercadenas/Desktop/TFG/TFG/TrainTestMatrices/y_test.csv', index_col = 0)

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.1) #, probability = True
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

##Comentaris
# -1r agafar propo i remi ce, començar a agafar dades des d'on comença a infundir propo ce i si algun pacient no en té cap no agafarlo directament FET
#Que les dades estiguin balancejades, igual num de dades abans i despres de l verbal (pel pairing i perque el model no fagi lu de la classificació gats gossos) FET
#Interpolació de valors que no hi són FET
#Tots els moments d'abans del verbal que siguin 0 i els altres 1. Comprovar fent debug que els frames que donen son correctes
# I ara que fer?

#Preguntes:
#1- Com miro els valors dun dataframe al debug?
#2- He de normalitzar respecte als maxims de cada pacient o respecte als maxims totals?
#3- Com pot ser que la confusion matrix i els parametres del model siguin perfectes? Vaig pensar que era pel que m'havies comentat de fer el split per pacient i no, perque ara al canviarho passa el mateix.

#Al ficar REMI també les dades se'm redueixen a la meitat.
#No hi ha molta disparitat entre valors a prediure? No
#GUardar els train i test a arxius per que no canviin els resultats cada cop FET