import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width",500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull.values.any()
df.isnull().sum

################################### Genel Resim ##########################################
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


check_df(df)

######################### Kategorik Değişken Analizi I ########################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width",500)
df = sns.load_dataset("titanic")

df["embarked"].value_counts() #belirtilen etikete ait verilerin sayısı
df["sex"].unique()  #benzersiz değişkenler
df["class"].nunique()   #benzersiz değişkenlerin sayısı

#kategorik değişkenler
cat_cols = [col for col in df.columns if str(df[col].dtype) in ["category", "bool", "object"]]
#sayısal gözüken ama kategorik olan değişkenler
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
#kategorik gözüken ama kategorik olmayan çok sayıda farklı değere sahip değişkenler (örneğin isim-soyisim gibi)
cat_but_car = [col for col in df.columns if (df[col].nunique() > 20) and (str(df[col].dtype) in ["category", "bool", "object"])]

cat_cols = cat_cols + num_but_cat

[col for col in df.columns if col not in cat_cols]  #kategorik olmayan değişkenleri verir

#girilen değişkenin toplam sayısı ve tüm dataframe içindeki oranını veren fonksiyon
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################################################")

#otomatik olarak tüm değişkenler için fonksiyonun uygulanması
for col in cat_cols:
    cat_summary(df,col)
    
######################### Kategorik Değişken Analizi II ########################################
#bir önceki kodlara ek olarak kutu grafiği çiziminin eklnemesi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    if(df[col].dtypes == "bool"):
        df[col] = df[col].astype(int)

    cat_summary(df, col, plot=True)
    
########################## Sayısal Değişken Analizi  ########################################

df[["age", "fare"]].describe().T    #sayısal değişkenler için özet bilgisini verir

num_cols = [col for col in df.columns if df[col].dtypes in ["float","int"]] #sayısal değişkenlerin tümünü verir
num_cols = [col for col in num_cols if col not in cat_cols] #sayısal değişkenlerden gerçekte kategorik olanlarını ayrıştırarak gerçek sayısal değişkenleri verir

def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.25, 0.50, 0.75, 1]
    print(dataframe[num_col].describe().T)
    if plot:
        dataframe[num_col].hist()
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show(block=True)


num_summary(df, "fare", True)

########################## Değişkenlerin Yakalanması  ########################################
   
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width",500)
df = sns.load_dataset("titanic")

#docstring

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        kategorik değişken listesi
    num_cols: list
        numerik değişken listesi
    cat_but_car: list
        kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    cat_cols = [col for col in df.columns if str(df[col].dtype) in ["category", "bool", "object"]]
    # sayısal gözüken ama kategorik olan değişkenler
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
    # kategorik gözüken ama kategorik olmayan çok sayıda farklı değere sahip değişkenler (örneğin isim-soyisim gibi)
    cat_but_car = [col for col in df.columns if (df[col].nunique() > 20) and (str(df[col].dtype) in ["category", "bool", "object"])]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in df.columns if col not in cat_but_car]  # kategorik olmayan değişkenleri verir

    num_cols = [col for col in df.columns if df[col].dtypes in ["float", "int"]]  # sayısal değişkenlerin tümünü verir
    num_cols = [col for col in num_cols if col not in cat_cols]  # sayısal değişkenlerden gerçekte kategorik olanlarını ayrıştırarak gerçek sayısal değişkenleri verir

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len((cat_but_car))}")
    print(f"num_but_cat: {len((num_but_cat))}")

    return cat_cols, num_cols, cat_but_car

grab_col_names(df)

#############################hedef değişkenin kategorik değişkenler ile analizi
def target_summary_with_cat(dataframe, target, cat_col):
    print(pd.DataFrame({"Target Mean": dataframe.groupby(cat_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

##############################hedef değişkenin sayısal değişkenler ile analizi
def target_summary_with_num(dataframe, target, num_col):
    print(dataframe.groupby(target).agg({num_col:"mean"}), end="\n")

for col in num_cols:
    target_summary_with_num(df, "survived", col)

