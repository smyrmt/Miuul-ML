import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%3f" % x)
pd.set_option("display.width", 500)

df = pd.read_csv("datasets/advertising.csv")
df.shape

################################################
# Simple Linear Regression
################################################

X = df[["TV"]]
y = df[["sales"]]

#################################################
# Model
################################################
reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*x

# sabit (b - bias)
reg_model.intercept_[0]

# tv'nin katsayısı (w1) - teta notasyonu- w notasyonu -coefficient
reg_model.coef_[0][0]

###################################################
# Tahmin
###################################################
# 150 birimlik tv harcaması olsa ne kadar satış olması beklenir?
reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

#modelin görselleştirilmesi
g = sns.regplot(x=X,y=y, scatter_kws={"color": "b", "s": 9}, ci=0, color="r") # ci = güven aralığı, color regresyon çizgisinin rengi
g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV *{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış sayısı")
g.set_xlabel("TV harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show(block=1)

###################################################
# Tahmin Başarısı
##################################################

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
y.mean() #bağımlı değişken yani satışların ortalaması
y.std()

# RMSE
np.sqrt(mean_squared_error(y, y_pred))

# MAE
mean_absolute_error(y, y_pred)

# R-KARE
reg_model.score(X, y) #bağımlı değişkenin bağımsız değişkeni açıklama yüzdesini verir

################################################
# Multiple Linear Regression
################################################
X = df.drop("sales", axis=1)
y = df[["sales"]]

#################################################
# Model
################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
reg_model = LinearRegression().fit(X_train, y_train)
# sabit (b - bias)
reg_model.intercept_

# tv'nin katsayısı (w1) - teta notasyonu- w notasyonu -coefficient
reg_model.coef_

"""Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir? model denklemini yazınız
TV: 30
radio: 10
newspaper: 40
b = 2.90794702
w = [0.0468431 , 0.17854434, 0.00258619]
"""
# Sales = 2.9 + TV * 0.04 + radio * 0.17 + newspaper * 0.002
2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619

# daha fonsiyonel olsun istersek;
yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T
reg_model.predict(yeni_veri)

#################################################
# Tahmin başarısını değerlendirme
################################################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.736902590147092

# Train R-Kare
reg_model.score(X_train, y_train)

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.4113417558581585 #normalde test hatası daha yüksek olur bu örnek başarılı bir örnek

# Train R-Kare
reg_model.score(X_test, y_test)

# 10 katlı Cros validation RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))
# 1.6913531708051797

############################################################################################
########################### BONUS ##############################
###########################################################################################
# Simple Linear Regression with Gradient Descent from Scratch
#############################################################

# Cost function MSE
def cost_func(Y, b, w, X):
    m = len(Y)
    sse = 0 # sse : sum of squared error
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2
    mse = sse /m
    return mse

# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate *1 / m * w_deriv_sum)
    return new_b, new_w

# train function
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w, cost_func(Y, initial_b, initial_w, X))) #ilk hatanın raporlandığı bölüm
    b = initial_b
    w = initial_w
    cost_history = []
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_func(Y, b, w, X) #değişikliklerden sonra hata nasıl bakıyoruz
        cost_history.append(mse)
        if i % 100 == 0: #her 100 döngüden sonra raporlama yaptırıyoruz
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4f}".format(i, b, w, mse))
    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters,b, w, cost_func(Y, b, w, X))) #son raporlama
    return cost_history, b, w

# not: parametre: modelin veriyi kullanarak veri setinden bulunur, hiperparametre: kullanıcı tarafından verilir
"""normal regresyon : ağırlıklar analitik şekilde çözülür, 
    gradient descent : optimizasyon yöntemidir, sürece bağlıdır, belirlenmesi gereken parametreler vardır
"""

X = df["radio"]
Y = df["sales"]
#hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
