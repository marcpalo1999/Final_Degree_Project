from libraries import *


path = '/Users/marcpalomercadenas/Desktop/TFG/TFG/archives/VARIABLES.xlsx'
source = pd.read_excel(path)

#Input feature selection
selected_features = ['ECG_HR', 'NIBP_MEAN','PROPO_CE', 'REMI_CE']

#Function to find the index and the time where LoC occurs
def finder(patient_dataframe, word1, word2):
    
    patient_dataframe.reset_index(drop = True)

    #We initialize the local variables
    idx = []
    a = 0
    b = 0
    
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
    

    #If neither of the options has been useful to detecting the verbal response nothing is returned
    if a*b == 1:
        idx = []
        LoC_time = 0
    else:
        LoC_time = patient_dataframe['TIME'].loc[idx]


    return [idx, LoC_time]

#Data preprocessing before entering it to the model
def data_preprocessing(patient_dataframe,idx,LoC_t,selected_features):
    
    #Busquem el index del primer valor de propo i si no hi és descartem el pacient
    fvi = patient_dataframe['PROPO_CE'].first_valid_index()
    
    
    #Patient is only valid it it has propo induction and LoC happend after propo induction
    if fvi != None and idx > fvi:

        #Cast of the df
        a = patient_dataframe.loc[fvi:(2*idx-fvi),:]
        
        
        #Interpolatio  Nans
        a.interpolate(method='linear', limit_direction='forward', axis=0, inplace = True)

        #LoC binarisation (creating a vector of 0 until negative verbal response, and the other are 1)
        index_LoC = list(a['TIME']).index(LoC_t)
        a.reset_index(drop = True, inplace = True)
        LoC_index = list(a['TIME']).index(LoC_t)
        a['LoC']=0
        a.loc[LoC_index:,'LoC']=1

        #Vdf of wanted variables
        a = a.loc[:,['ECG_HR', 'NIBP_MEAN','PROPO_CE', 'REMI_CE', 'LoC', 'TIME']]

        #NaN handling
        a.loc[:,'REMI_CE'] = a.loc[:,'REMI_CE'].fillna(0)
        a = a.dropna()
        a.reset_index(drop = True, inplace = True)

        #Variable assignation
        LoC = a['LoC']
        Time = a['TIME']
        a = a.loc[:,selected_features]

    else: 
        a = pd.DataFrame(columns= selected_features)
        LoC = pd.DataFrame()
        Time = pd.DataFrame()
    
    

    return [a, LoC, Time]

#Variable initialization
general_dataframe = pd.DataFrame()
X_train =[]
X_test = []
y_train = []
y_test = []
available_event = 0
no_event = 0
e = 0
available_index = 0
available_patient = 0

#Extracting patients df
for patient_ID, event_state in zip(source['ID'],source['EVENTS']):
    if event_state == 'AVAILABLE':
        
        #Counting the patients with record of EVENTS
        available_event = available_event +1

        #Reading the current archive of the patient
        path = '/Users/marcpalomercadenas/Desktop/TFG/TFG/archives'
        chdir(path)
        archive = pd.read_csv(patient_ID+'.csv')

        #Finder returns the index of the verbal response
        index, LoC_t = finder(patient_dataframe = archive, word1 = "verb'", word2 ="verbal")

        #If there is index of verbal response, there is verbal response
        if index != []:
            available_index = available_index+1
            
            #We cut the patients' dataframe info  +/- 2 minutes from the lose of verbal response
            frame, LoC, Time = data_preprocessing(archive,index,LoC_t,selected_features)
            

            #We append the 240 seconds patients' dataframes for later concatenation and visualization
            pd.concat([general_dataframe, pd.concat([frame, LoC], ignore_index = True)], ignore_index= True)

            #If the patient had verb resp on propo+remi, the dataframe shouldn't be empty: 
            if len(frame) != 0:
                available_patient = available_patient +1


                #Data scaling. There are a lot more types of scaling
                from sklearn.preprocessing import MaxAbsScaler
                scaler = MaxAbsScaler()
                frame = scaler.fit_transform(frame)

                # Split data into train and test subsets, randomly and putting whole patients in train or test, to avoid overfitting.
                x = random.uniform(0,1)
                if x < 0.5:
                    
                    X_train.append(frame)
                    y_train.append(LoC)
                else:
                    X_test.append(frame)
                    y_test.append(LoC)
                
 
            else:
                e = e+1
    else:
        no_event = no_event +1


print(f'There are {available_index}  patients from {available_index + no_event} with verbal response recording')
print(f'There are {round((available_patient)*100/available_index,1)}% of patients from the {available_index} with recorded neg. verbal response which data is handled')

#Concatenation the patients' data for model training
X_train = concatenate((X_train))
X_test = concatenate(X_test)
y_train = concatenate(y_train)
y_test = concatenate(y_test)

#Saving the matrices to csv so that the results don't change constantly
pd.DataFrame(X_train).to_csv('/Users/marcpalomercadenas/Desktop/TFG/TFG/TrainTestMatrices/X_train.csv')
pd.DataFrame(X_test).to_csv('/Users/marcpalomercadenas/Desktop/TFG/TFG/TrainTestMatrices/X_test.csv')
pd.DataFrame(y_train).to_csv('/Users/marcpalomercadenas/Desktop/TFG/TFG/TrainTestMatrices/y_train.csv')
pd.DataFrame(y_test).to_csv('/Users/marcpalomercadenas/Desktop/TFG/TFG/TrainTestMatrices/y_test.csv')