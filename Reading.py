from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from os import chdir
from scipy.interpolate import interp1d


# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from os import chdir, getcwd
#Hauria d'importar el primer CSV i veure a on hi ha events (pel training set)
#Després fer un data frame juntant tots els csv?
#I un model que m'indiques si el pacient està en fase de no resposta ciliar
#SIntroduir una serie de variables a determinar (quines? Com es trien?)
#Per tal que em dones una probabilitat de que la resposta palpedral fos negativa a  cada segon
#És supervised perque es una classificació (?)
#Com trio el model

#Vull un classificador que agafant els features () em digui a cada segon si està dormit o no (classificacció binària)
#No  ho farem amb ML temporal pk no dona millors resultats
#Agafaré com dos minuts abans i dos després del LOC (resopsta verbal)

path = '/Users/marcpalomercadenas/Desktop/TFG/TFG/archives/VARIABLES.xlsx'
source = pd.read_excel(path)
interpolation = ['ECG_HR', 'NIBP_MEAN','PROPO_CE']

#Jo veig dues opcions de preparar les dades. Una primera que és la que estic fent
#On a cada segon ens hauria de donar la probabilitat de que la resposta verbal fos negativa
#La segonaopció seria fer un dataframe on cada pacient fos una obsrvació i que tinguessim la
#Ce (concentració efecte) a on s'ha detectat la perdua de resposta verbal i els segons que s'ha trigat 
#des de l'inici de la infució. (Aixó estaria bé ficarho a analisis de possibilitats)
def Finder(patient_dataframe, word1, word2):#, propo = True, remif = True):
    
    #We initialize the local variables and the error counter
    #to know how many patients are being lost without verbal response
    idx = []
    a = 0
    b = 0

    patient_dataframe.reset_index()
    #We search for "verb'" inside the event feature
    try:
        idx = list(patient_dataframe['EVENT']).index(word1)
    except ValueError:
        a = 1

    #We search for "verbal" inside the event feature
    try :
        idx = list(patient_dataframe['EVENT']).index(word2)
    except ValueError:
        b = 1
    

    #If neither of the options has been useful to detecting the verbal response the error increases
    if a*b == 1:
        idx = []

    return idx

def print_full(x):
    pd.set_option('display.max_rows', 'display.max_cols')
    print(x)
    pd.reset_option('display.max_rows')

def data_preprocessing(patient_dataframe,idx):
    #Busquem el index del primer valor de propo i si no hi és descartem el pacient
    patient_dataframe.reset_index()
    fvi = patient_dataframe['PROPO_CE'].first_valid_index()
    

    if fvi != None:

        #Aqui està agafpd.ant el index de 60 abans del verbal
        a = patient_dataframe.iloc[fvi:(idx+fvi),:]
        a.reset_index()

        #Interpolation
        for element in interpolation:
            a[element] = a[element].interpolate(axis = 0, method = 'linear')
            #a[element].fillnull(mean(a[element]), inplace = True)
        
        #LoC binarisation
        vec1 = list(repeat(0,idx))
        vec2= repeat(1,idx)
        for element in vec2:
            vec1.append(element)
        #print(vec1+vec2)
        a['LoC'] = pd.DataFrame(vec1)

    else: 
        a = []
    
    #Que miri si vol mirar remi i propo o nomes propo
    return a

#Variable initialization
patient_list = []
Verbal_and_propo = 0
available_event = 0
no_event = 0


#Then we extract a list of filenames  from first csv to open all the others
for patient_ID, event_state in zip(source['ID'],source['EVENTS']):
    if event_state == 'AVAILABLE':
        
        #Counting the patients with record of EVENTS
        available_event = available_event +1

        #Reading the current archive of the patient
        path = '/Users/marcpalomercadenas/Desktop/TFG/TFG/archives'
        chdir(path)
        archive = pd.read_csv(patient_ID+'.csv')

        #Finder returns the index of the verbal response
        index = Finder(patient_dataframe = archive, word1 = "verb'", word2 ="verbal")

        if index != []:
            
            

            #We cut the patients' dataframe info  +/- 2 minutes from the lose of verbal response
            frame = data_preprocessing(archive,index)
            #Aqui hauriem de generar un nou parametre que tornes de data_preprocessing
            #i que fos la condició per fer l'append final
            if len(frame) != 0:
                
                #We count the number of patients with verbal response in propo induction

                Verbal_and_propo = Verbal_and_propo +1
                #We append the 240 seconds patients' dataframes for later concatenation
                patient_list.append(frame)

    else:
        no_event = no_event +1


print(f'There are {available_event}  patients from {available_event + no_event} with recorded events of any type')
print(f'There are {round((Verbal_and_propo)*100/available_event,1)}% of patients from the {available_event} with recorded events presenting negative verbal response at some point while propo induction')

#Concatenation the patients' data to introduce it to the ML algorithm
input = pd.concat(patient_list)

#Result feature declaration
#input['LoC'] = 0

#Input feature selection
selected_features = ['ECG_HR', 'NIBP_MEAN','PROPO_CE', 'LoC']
imput = input[selected_features]

#CUIDADO QUE AQUI ESTEM PERDENT INFO
imput = imput.dropna()

#Data scaling. There are a lot more types of scaling
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
imput = scaler.fit_transform(imput)

# Split data into train and test subsets

##AIXÒ NO FUNCIONA GENS
X_train, X_test, y_train, y_test = train_test_split(
    imput, imput[:,-1], test_size=0.5, shuffle=False
)


# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

##Comentaris
# -1r agafar propo i remi ce, començar a agafar dades des d'on comença a infundir propo ce i si algun pacient no en té cap no agafarlo directament FET
#Que les dades estiguin balancejades, igual num de dades abans i despres de l verbal (pel pairing i perque el model no fagi lu de la classificació gats gossos) FET
#Interpolació de valors que no hi són FET
#Tots els moments d'abans del verbal que siguin 0 i els altres 1. Comprovar fent debug que els frames que donen son correctes