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

######################### Kategorik Değişken Analizi II ########################################
