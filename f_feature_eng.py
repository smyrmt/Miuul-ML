import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,StandardScaler, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%3f" % x)
pd.set_option("display.width", 500)

def load_application_train():
    data = pd.read_csv("application_train.csv")
    return data

df = load_application_train()
df.head()

def load():
    data = pd.read_csv("titanic.csv")
    return data

df = load()
df.head()

########################
# Aykırı değerleri yakalama
###########################

def outlier_thresholds(dataframe, col_name, q1 = 0.25, q3 = 0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 -quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit =quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age")
low, up =outlier_thresholds(df, "Fare")
df[(df["Fare"] < low) | (df["Fare"] > up)].head()

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_col_names(dataframe, cat_th=10, car_th=10):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat #tüm kategorikler
    cat_cols = [col for col in cat_cols if col not in cat_but_car]  #tüm kategoriklerden sahte kategorikleri çıkartarak gerçek kategorikler bulunur
    #numerik veriler
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categorics: {len(cat_cols)}")
    print(f"numerics: {len(num_cols)}")
    print(f"cat but car: {len(cat_but_car)}")
    print(f"num but cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    outliers = (dataframe[col_name] < low) | (dataframe[col_name] > up)
    if dataframe[outliers].shape[0] > 10:
        print(dataframe[outliers].head())
    else:
        print(outliers)

    if index:
        outlier_index = dataframe[outliers].index
        return outlier_index


grab_outliers(df, "Age", index=True)

########################################################################33
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers =dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
df.shape

for col in num_cols:
    new_df =remove_outlier(df, col)

df.shape[0] - new_df.shape[0]

#baskılama yöntemi (silmek istemediğimiz durumlarda)
def replace_with_thresholds(dataframe, var):
    low_limit, up_limit = outlier_thresholds(dataframe, var)
    dataframe.loc[(dataframe[var] < low_limit), var] = low_limit
    dataframe.loc[(dataframe[var] > up_limit), var] = up_limit


for col in num_cols:
    print(col, check_outlier(df,col))

for col in num_cols:
    replace_with_thresholds(df, col)

###########################################################
# Çok değişkenli Aykırı Değer Analizi: Local Outlier Factor
##########################################################
df = sns.load_dataset("diamonds")
df = df.select_dtypes(include={"float64", "int64"})
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]
np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style=".-")
plt.show()

th = np.sort(df_scores)[3] #elbow yöntemine göre grafiğe de bakarak 3. indekse sahip değeri eşik değer olarak belirleyebiliriz
df[df_scores < th] #negatif değerlerde olduğumuz için eşik değerden küçükler aykırıdır.

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

