# Import Helper Library
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import operator

# Import data
data_df = pd.read_csv("kc_house_data.csv")
data_df.head()

# Make Dataset
dataset = data_df.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,17,18,19,20,2]]
# Menghapus data Nan 
dataset = dataset.fillna(0)

# Variabel
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,15].values

dataset.head()

# Membagi data menjadi Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

# Menggunakan Gradient booster
from sklearn.ensemble import GradientBoostingRegressor
estimasi = GradientBoostingRegressor(n_estimators=2000, max_depth=5, min_samples_split=2, learning_rate=0.1, loss="ls")

# Membuat model
model = estimasi.fit(X_train, Y_train)

# Memprediksi hasil Test set
prediksi = model.predict(X_test)

# Print your aktual and predict 
realval = pd.DataFrame(Y_test,columns=['real val'])
predval = pd.DataFrame(prediksi,columns=["predict val"])
compar = pd.concat([realval,predval],axis=1)
compar

# Cek the score
akurasi = model.score(X_test, Y_test)
print("nilai akurasi : {}".format(akurasi))

# Plot your predict with aktual
import matplotlib
matplotlib.rc('ytick', labelsize=10)
fig, ax = plt.subplots(figsize=(7,7))
plt.plot(Y_test, prediksi, 'g*')
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
plt.show()
