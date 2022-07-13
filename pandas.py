import seaborn as sns

df = sns.load_dataset("titanic")
df.describe().T
df.isnull().values.any()    #hiç null değer var mı? varsa True döner
df.isnull().sum()   #hangi değişkenlerde kaç adet null değer olduğunu döndürür. (False'ları saydığı için)
df["sex"].head()
df["sex"].value_counts()
######################
df[0:13]    #0'dan 13'e kadar olan indexlerdeki verileri gösterir
df.drop(0,axis=0)   #0. satırı sil (axis=0 satırlardan siler)
#diğer bir silme yöntemi
delete_indexes = [1,3,5,6]
df.drop(delete_indexes,axis=0)
#işlemlerin kalıcı olması için inplace=True kullanılır
#df.drop(delete_indexes,axis=0, inplace=True)

#######################
#değişkeni indexe çevirmek
#######################
df.age.head()   #df["age"].head() için farklı bir kullanım
df.index = df.age
df.drop("age", axis=1, inplace=True) #yaş bilgisini özelliklerden kaldırıyoruz
#yaş bilgisini tekrar ekleyerek index bilgilerini varsayılana çeviriyoruz
df = df.reset_index().head()

#######################
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns',None)   #... yazımını kaldırarak tüm sütun değerlerini gösterir
df = sns.load_dataset("titanic")
df["age2"] = df["age"] ** 2
df.loc[:,df.columns.str.contains("age")]    #"age" ifadesini içeren etiketlere sahip tüm öğreleri gösterir. not: ~ değil anlamı katar.
#######################
#iloc: integer based selection (yalnızca integer değerler alır)
df.iloc[0:3]    #0dan 3e kadar olan indexleri getirir (3 hariç)
#loc: label based selection
df.loc[0:3]     #0dan 3e kadar olan indexleri getirir (3 dahil)
df.loc[0:3,"age"]   #"age etiketine sahip olan 0dan 3e kadar olan indexleri getirir (3 dahil)
#######################
df[df["age"] > 50].["age"].count()  #yaşı 50'den büyük olanların sayısını verir
df.loc[df["age"] > 50, ["age","class"]] #yaşı 50'den büyük olanların yaş ve sınıf bilgisi
df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age","class"]]   #yaşı 50'den büyük olan erkeklerin yaş ve sınıf bilgisi