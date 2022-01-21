from libraries import *
from Train_Test_adquisition  import  data_preprocessing, finder

#Loading models
chdir("/Users/marcpalomercadenas/Desktop/TFG/TFG")
with open('SVM_model.pickle', "rb") as file:
    SVM = pickle.load(file)

with open('RandomForest_model.pickle', "rb") as file:
    rfc = pickle.load(file)


path = '/Users/marcpalomercadenas/Desktop/TFG/TFG/archives/CMA4_201217_083154.csv'
source = pd.read_csv(path)

#Input feature selection
selected_features = ['ECG_HR', 'NIBP_MEAN','PROPO_CE', 'REMI_CE']

#Processing the data
index, LoC_t  = finder(source, "verb'", "verbal")
data_processed, LoC, Time = data_preprocessing(source, index, LoC_t,selected_features)

data_processed.columns = ['0','1','2','3']

#Normalisation
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
data = scaler.fit_transform(data_processed)

#Model outcome
predicted_values = rfc.predict_proba(data)
pred_LoC = predicted_values[:,1]

#Plotting of the probability
plt.figure(dpi=1200) 
plt.plot(Time, pred_LoC)
plt.plot(Time, LoC)
plt.legend(['SVM Prediction','Loss of verbal response'])
plt.title('LoC Prediction Example')
plt.xlabel('time since init Propofol infusion (s)')
plt.ylabel('Probability of being unconscious')
plt.savefig('LoC Example')
plt.show()



#Plotting with a threshold
plt.figure(dpi=1200) 
plt.plot(Time, pred_LoC > 0.85)
plt.plot(Time, LoC)
plt.legend(['SVM Prediction','Loss of verbal response'])
plt.title('LoC Prediction Example')
plt.xlabel('time since init Propofol infusion (s)')
plt.ylabel('Probability of being unconscious. Threshold = 0.85')
plt.savefig('LoC Example with Threshold')
plt.show()
