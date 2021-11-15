from numpy import *
import pandas as pd
import matplotlib.pyplot as plt


# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from os import chdir
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
chdir('archives')
path = 'VARIABLES.xlsx'
source = pd.read_excel(path)




patient_list = []

#Then we extract a list of filenames  from first csv to open all the others
for element, eve in zip(source['ID'],source['EVENTS']):
    if eve == 'AVAILABLE':
        #Reading the current archive
        archive = pd.read_csv(element+'.csv')
        #we find the moment where there is no more verbal response
        #idx = archive.index[archive['EVENT'] == "verb'"].tolist()
        idx = 
        print(idx)
        if idx != []:
            #we cut the patients info from +/- 2 minutes from the lose of verbal response
            frame = archive.iloc[idx[0]-120:idx[0]+120,:]
            

            #(merge or concatenate function?)
            patient_list.append(frame)

#we merge all the patients
input = pd.concat(patient_list)

input['LoC'] = 0
selected_features = ['ECG_HR', 'NIBP_MEAN','PROPO_CP', 'REMI_CP', 'LoC']
input = input[selected_features]

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    input, input['LoC'], test_size=0.4, shuffle=False
)