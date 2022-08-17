import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_validate

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%3f" % x)
pd.set_option("display.width", 500)

###############################################################
# KNN
###############################################################

# EDA
df = pd.read_csv("diabetes.csv")
df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()

# Data Preprocessing
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_scaled = StandardScaler().fit_transform(X) #bu işlemde sütun isimleri silinmiş oluyor
X = pd.DataFrame(X_scaled, columns=X.columns) # standartlaştırılmış değerlerin sütun etiketlerini ekliyoruz

# Model
knn_model = KNeighborsClassifier().fit(X,y)
random_user = X.sample(1, random_state=45) # verisetinden rastgele bir kişi seçiyoruz
knn_model.predict(random_user) # bu kişinin diyabet olup olmama olasılığını tahmin ediyoruz

# model evaluation
#conf matrix için
y_pred = knn_model.predict(X) # tüm gözlem birimleri için tahmin değerlerini bulalım
#AUC için y_prob
y_prob = knn_model.predict_proba(X)[:, 1] # bağımsız değişkenlerin 1 sınıfına ait olma olasılıkları
print(classification_report(y, y_pred))
# AUC
roc_auc_score(y, y_prob)
# cross_validate: birden fazla doğruluk skoru elde etmemizi sağlar
cv_results = cross_validate(knn_model, X, y,cv=5, scoring=["accuracy", "f1", "roc_auc"])

#####################################################
# Hyperparameter Optimization
#####################################################
knn_model = KNeighborsClassifier()
knn_model.get_params()
knn_params = {"n_neighbors": range(2, 50)} #2-50 arası komşuluklardan hangisinin daha iyi sonuç vereceğini bulmaya çalışıyoruz
knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(X,y)
# n_jobs=-1 işlemciyi tam kapasite kullanır, verbose=1 rapor verir
knn_gs_best.best_params_ # en iyi komşuluk parametresi 17 imiş

#########################################
# Final Model
########################################
knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)
# not: key-value şeklinde bir değeri kullanmak için ** kullnılır
cv_results = cross_validate(knn_final, X, y,cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

# sonuç olarak komşuluk parametresini optimize ederek başarıyı arttırdık
