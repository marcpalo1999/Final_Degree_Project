from libraries import *
from Train_Test_adquisition  import  data_preprocessing, finder

chdir("/Users/marcpalomercadenas/Desktop/TFG/TFG")
with open('SVM_model.pickle', "rb") as file:
    SVM = pickle.load(file)

with open('RandomForest_model.pickle', "rb") as file:
    rfc = pickle.load(file)

# Ask the user to enter an area and calculate 
# its price using the imported model
path = '/Users/marcpalomercadenas/Desktop/TFG/TFG/archives/CMA4_201217_121621.csv'
source = pd.read_csv(path)
#Input feature selection
selected_features = ['ECG_HR', 'NIBP_MEAN','PROPO_CE', 'REMI_CE']

index  = finder(source, "verb'", "verbal")
source = data_preprocessing(source, index, selected_features)
t = linspace(0,150,len(source[1]))
data = pd.DataFrame(source[0])
data.columns = ['0','1','2','3']
predicted_values = rfc.predict(data)
plt.plot(t, predicted_values)
real_values = source[1]
real_values.reset_index(drop = True)
plt.plot(t, real_values)

plt.show()