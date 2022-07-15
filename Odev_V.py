import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width",500)
############################## Pandas Alıştırmalar ##########################
#görev1
df = sns.load_dataset("titanic")
#görev2
df["sex"].value_counts()
#görev3
df.nunique()
#görev4
df["pclass"].nunique()
#görev5
df[["pclass", "parch"]].nunique()
#görev6
df["embarked"].dtype
df["embarked"].astype("category")
#görev7
df.loc[df["embarked"] =="C"]
#görev8
df.loc[df["embarked"] !="S"]
#görev9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz
df.loc[(df["age"] < 30) & (df["sex"] == "female")]
#görev10: Fare'i 500'den büyük veya yaşı 70’den büyük yolcuların bilgilerini gösteriniz
df.loc[(df["fare"] > 500) | (df["age"] > 70)]
#görev11: Her bir değişkendeki boş değerlerin toplamını bulunuz
df.isnull().sum().sum()
#görev12: who değişkenini dataframe’den çıkarınız
df.drop("who", axis=1)
#görev13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz

#görev14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz

#görev15:  survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz
df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "mean", "count"]})
#görev16: 
"""30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 verecek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri
setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)"""

#görev17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız
tips = sns.load_dataset("tips")
#görev18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerinin sum, min, max ve mean değerlerini bulunuz
tips.info()
tips.head()
tips["time"].value_counts()
tips.groupby(["time"]).agg({"total_bill": ["sum", "mean", "min", "max"]})
#görev19:
tips.groupby(["day", "time"]).agg({"total_bill": ["sum", "mean", "min", "max"]})
#görev20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre sum, min, max ve mean değerlerini bulunuz

#görev21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)

#görev22: total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in toplamını versin

#görev23:
"""Total_bill değişkeninin kadın ve erkek için ayrı ayrı ortalamasını bulunuz. Bulduğunuz ortalamaların altında olanlara 0, üstünde ve eşit
olanlara 1 verildiği yeni bir total_bill_flag değişkeni oluşturunuz.
Kadınlar için Female olanlarının ortalamaları, erkekler için ise Male olanların ortalamaları dikkate alınacktır. Parametre olarak cinsiyet ve total_bill
alan bir fonksiyon yazarak başlayınız. (If-else koşulları içerecek)"""

#görev24: total_bill_flag değişkenini kullanarak cinsiyetlere göre ortalamanın altında ve üstünde olanların sayısını gözlemleyiniz

#görev25: Veriyi total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız
