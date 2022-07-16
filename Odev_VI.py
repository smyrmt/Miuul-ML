import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width",500)
df = pd.read_csv("persona.csv")
df.info()
df.nunique()
df["SOURCE"].nunique()
df["SOURCE"].value_counts()
df["PRICE"].nunique()
df["PRICE"].value_counts()
df["COUNTRY"].value_counts()
df["COUNTRY"].value_counts().sum()
df["COUNTRY"].value_counts().mean()
df["SOURCE"].value_counts().mean()
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})
#Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

#Görev 3: Çıktıyı PRICE’a göre sıralayınız
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})
agg_df.sort_values("PRICE", ascending=False)
agg_df.index


#Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.
agg_df.reset_index()

#Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz
df["AGE_CAT"] = pd.cut(df["AGE"], [0,18,23,30,40,70])
df.head(10)

#Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız
