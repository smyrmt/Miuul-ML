import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width",500)
df = sns.load_dataset("titanic")
df.head()

######################
#Kategorik değişken görselleştirme
######################
df["sex"].value_counts().plot(kind="bar")
plt.show()

######################
#sayısal değişken görselleştirme
######################
plt.hist(df["age"])
plt.show()
plt.boxplot(df["fare"])
plt.show()
