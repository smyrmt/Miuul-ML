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

######################### Kategorik Değişken Analizi ########################################
