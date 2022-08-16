import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%3f" % x)
pd.set_option("display.width", 500)

###########################################################################################################
######################################### EDA #############################################################

#aşırı aykırı verileri almak için çeyreklik aralığı yüksek verildi
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, var):
    low_limit, up_limit = outlier_thresholds(dataframe, var)
    dataframe.loc[(dataframe[var] < low_limit), var] = low_limit
    dataframe.loc[(dataframe[var] > up_limit), var] = up_limit

df = pd.read_csv("diabetes.csv")
df.head()
df.shape

###########################
# Target Analizi
##########################
df["Outcome"].value_counts()
sns.countplot(x="Outcome", data=df)
plt.show(block=1)

#sınıf dağılımının oranlarını görmek için;
100 * df["Outcome"].value_counts() / len(df)

###########################
# Feature Analizi
##########################

df.describe().T

#sayısal değişkenlerin histogram grafiklerini göstermek için;
def plot_num_col(dataframe, num_col):
    dataframe[num_col].hist(bins=20)
    plt.xlabel(num_col)
    plt.show(block=1)

# sonuç değişkenini kaldıralım
cols = [col for col in df.columns if "Outcome" not in col]

for col in cols:
    plot_num_col(df, col)

###########################
# Target & Features
##########################

# hedefi sayısal değişkenlere özetle
def target_sum_with_num(dataframe, target, num_col):
    print(dataframe.groupby(target).agg({num_col: "mean"}), end="\n\n\n")

for col in cols:
    target_sum_with_num(df, "Outcome", col)

###########################################################################################################
######################################### Data Preprocessing ##############################################

df.shape
df.head()
df.isnull().sum()
df.describe().T

for col in cols:
    print(col, check_outlier(df, col))

# insülin değişkeninde aykırılık olduğu için bu değerleri eşik değerler ile değiştirelim
replace_with_thresholds(df, "Insulin")

#robust scaler tüm gözlem birimlerinden medyanı çıkarıp range değerine bölüyor (aykırı değerlerden etkilenmez)
for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()

#########################
# Model & Pred
#########################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
log_model = LogisticRegression().fit(X, y)
log_model.intercept_ #modelin sabiti
log_model.coef_ #bağımsız değişkenlerin katsayısı

y_pred = log_model.predict(X)

#########################
# Model Evaluation
#########################

def plot_conf_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy Score: {0}".format(acc), size=10)
    plt.show(block=1)

plot_conf_matrix(y, y_pred)
print(classification_report(y, y_pred))

#ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

############################
# Model Validation: Holdout
###########################

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.20, random_state=17)
log_model = LogisticRegression().fit(X_train, y_train)
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1] # bir sınıfa ait olma olasılıklarını verir
print(classification_report(y_test, y_pred)) # başarı düşmüş gibi gözüküyor. Model görmediği veride başarısız olabilir

plot_roc_curve(log_model, X_test, y_test)
plt.title("ROC CURVE")
plt.plot([0, 1], [0, 1], "r--")
plt.show(block=True)

######################################
# Model Validation: 5-fold cross val
#####################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
log_model = LogisticRegression().fit(X, y)
cv_results = cross_validate(log_model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()

######################################
# Prediction for a new observation
#####################################

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)
