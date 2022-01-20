from libraries import *
from Train_Test_adquisition  import  data_preprocessing, finder

chdir("/Users/marcpalomercadenas/Desktop/TFG/TFG")
with open('SVM_model.pickle', "rb") as file:
    SVM = pickle.load(file)

with open('RandomForest_model.pickle', "rb") as file:
    rfc = pickle.load(file)


path = '/Users/marcpalomercadenas/Desktop/TFG/TFG/archives/CMA4_210310_083940.csv'
source = pd.read_csv(path)

#Input feature selection
selected_features = ['ECG_HR', 'NIBP_MEAN','PROPO_CE', 'REMI_CE']

index, LoC_t  = finder(source, "verb'", "verbal")
data_processed, LoC, Time = data_preprocessing(source, index, LoC_t,selected_features)

data_processed.columns = ['0','1','2','3']

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
data = scaler.fit_transform(data_processed)

predicted_values = rfc.predict_proba(data)
plt.plot(Time, predicted_values[:,1])
plt.plot(Time, LoC)

plt.show()

pred_LoC = predicted_values[:,1]

#We set the threshold
plt.plot(Time, pred_LoC > 0.85)
plt.plot(Time, LoC)

plt.show()
