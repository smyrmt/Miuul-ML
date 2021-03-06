import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%3f" % x)
pd.set_option("display.width", 500)

def dia():
    data = pd.read_csv("diabetes.csv")
    return data
def telco():
    data = pd.read_csv("Telco-Customer-Churn.csv")
    return data

def check_df(dataframe, head=5):
    print("####################### SHAPE ############################")
    print(dataframe.shape)
    print("####################### INFO ############################")
    print(dataframe.info())
    print("####################### DESCRIBE ############################")
    print(dataframe.describe().T)
    print("####################### NA ############################")
    print(dataframe.isnull().sum())
    print("####################### QUANTILES ############################")
    print(dataframe.describe([0,0.05,0.5,0.95,0.99,1]).T)
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.25, 0.50, 0.75, 1]
    print(dataframe[num_col].describe().T)
    if plot:
        dataframe[num_col].hist()
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show(block=True)
def target_summary_with_cat(dataframe, target, cat_col):
    print(pd.DataFrame({"Target Mean": dataframe.groupby(cat_col)[target].mean()}))
def target_summary_with_num(dataframe, target, num_col):
    print(dataframe.groupby(target).agg({num_col:"mean"}), end="\n")

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat  # t??m kategorikler
    cat_cols = [col for col in cat_cols if
                col not in cat_but_car]  # t??m kategoriklerden sahte kategorikleri ????kartarak ger??ek kategorikler bulunur
    # numerik veriler
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categorics: {len(cat_cols)}")
    print(f"numerics: {len(num_cols)}")
    print(f"cat but car: {len(cat_but_car)}")
    print(f"num but cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
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
def missing_values(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean(), }), end="\n\n\n")
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_col = [col for col in temp_df.columns if
                temp_df[col].dtypes == "O" and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_col:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare",
                                temp_df[var])  # rare_labels i??indeyse "Rare" yaz de??ilse bir??ey yapma

    return temp_df
def one_hot_encoder(dataframe, cat_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first)
    return dataframe

################################ G??rev 1: EDA ##############################
#Ad??m 1: Genel resmi inceleyiniz.
df = dia()
check_df(df)
#Ad??m 2: Numerik ve kategorik de??i??kenleri yakalay??n??z
cat_cols, num_cols, cat_but_car = grab_col_names(df,cat_th=10, car_th=20)
#Ad??m 3: Numerik ve kategorik de??i??kenlerin analizini yap??n??z

"""Ad??m 4: Hedef de??i??ken analizi yap??n??z. (Kategorik de??i??kenlere g??re hedef de??i??kenin ortalamas??, hedef de??i??kene g??re
numerik de??i??kenlerin ortalamas??)"""
#kategorik de??i??kenler
for col in cat_cols:
    target_summary_with_cat(df, "Outcome", col)
#numerik de??i??kenler
for col in num_cols:
    target_summary_with_num(df, "Outcome", col)
#Ad??m 5: Ayk??r?? g??zlem analizi yap??n??z
for col in df.columns:
    print(col, "    ", check_outlier(df, col))

#Ad??m 6: Eksik g??zlem analizi yap??n??z
missing_values(df)
#Ad??m 7: Korelasyon analizi yap??n??z
for col in df.columns:
    plt.hist(df[col])
    plt.title(col)
    plt.show(block=1)
################################ G??rev 2: Feature Eng ##############################
""" Ad??m 1: Eksik ve ayk??r?? de??erler i??in gerekli i??lemleri yap??n??z. Veri setinde eksik g??zlem bulunmamakta ama Glikoz, Insulin vb. 
de??i??kenlerde 0 de??eri i??eren g??zlem birimleri eksik de??eri ifade ediyor olabilir. ??rne??in; bir ki??inin glikoz veya insulin de??eri 0 
olamayacakt??r. Bu durumu dikkate alarak s??f??r de??erlerini ilgili de??erlerde NaN olarak atama yap??p sonras??nda eksik 
de??erlere i??lemleri uygulayabilirsiniz."""

#Ad??m 2: Yeni de??i??kenler olu??turunuz
#Ad??m 3: Encoding i??lemlerini ger??ekle??tiriniz
#Ad??m 4: Numerik de??i??kenler i??in standartla??t??rma yap??n??z
#Ad??m 5: Model olu??turunuz
