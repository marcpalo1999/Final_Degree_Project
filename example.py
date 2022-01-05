from libraries import *

with open('SVM_model.pickle', "rb") as file:
    SVM = pickle.load(file)

with open('RandomForest_model.pickle', "rb") as file:
    rfc = pickle.load(file)

# Ask the user to enter an area and calculate 
# its price using the imported model
path = '/Users/marcpalomercadenas/Desktop/TFG/TFG/archives/CMA4_201217_134156.csv'
source = pd.read_csv(path)
#Input feature selection
selected_features = ['ECG_HR', 'NIBP_MEAN','PROPO_CE', 'REMI_CE']
source[selected_features]
proped_price = SVM.predict(source)
print ("Proped price:", round(proped_price[0], 2))