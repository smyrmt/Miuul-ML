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
#######################
#groupby
#######################
df.groupby("sex")["age"].mean() #cinsiyete göre yaş ortalamasını verir
df.groupby("sex").agg({"age":["mean", "sum"]}) #cinsiyere göre yaş ortalaması ve yaşların toplamını verir
df.groupby("sex").agg({"age":["mean", "sum"], "survived": "mean"}) #ek olarak hayatta kalma oranını verir.
df.groupby(["sex", "embark_town", "class"]).agg({"age":["mean", "sum"], "survived": "mean"}) #gruplama işlemi birden fazla olabilir
######################
#groupby yerine pivot table da kullanılabilir.
#cinsiyet ve liman bilgisine göre hayatta kalma oranları şu şekilde bulunabilir
df.pivot_table("survived", "sex", "embarked", aggfunc="std") #varsayılan olarak ortalama verir. aggfunc="std" standart sapmayı verir.
df.pivot_table("survived", "sex", ["embarked", "class"])

#yaşları bölümlendirerek hangi yaş aralığında hayatta kalma oranı nasıl diye bakılabilir
df["new_age"] = pd.cut(df["age"],[0,10,18,25,40,90])
df.pivot_table("survived", "sex", ["new_age", "class"])

##püf nokta##
#çıktılarda tüm satırların yan yana gelmesi için
pd.set_option("display.width",500)

######################
#apply&lambda
#apply: fonksiyon kullanımı için
#lambda: kullan at fonksiyon
######################
df["age2"] = df["age"]*2
df["age3"] = df["age"]*5
(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

#"age" değeri içeren etiketlerin verilerini 10'a bölme
#geleneksel yöntem
for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10
#pratik yöntem
df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

#standartlaştırma fonksiyonu kullanım örneği
def standart_scaler(col_name):
    return(col_name - col_name.mean()) / col_name.std()
df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)
######################
##concat ile birleştirme##
#iki farklı data frame oluşturalım
m = np.random.randint(1, 30, size=(5, 3))
df1 = pd.DataFrame(m,columns=["var1", "var2", "var3"])
df2 = df1 + 99
#daha sora bu iki farklı dataframe'i birleştirelim
pd.concat([df1,df2], ignore_index=1)    #index değerlerinin yeniden oluşturulması için ignore_index=1 yapılır
##merge ile birleştirme##
df1 = pd.DataFrame({"emp":["ali", "veli", "ayla"],
                    "group": ["dr", "eng", "hr"]})
df2 = pd.DataFrame({"emp":["ali", "veli", "ayla"],
                    "start_date": [2010, 2009, 2014]})
pd.merge(df1, df2, on="emp")    #otomatik olarak emp üzerinden birleştirme yapılır. farklı bir etiket üzerinden birleştirme yapabilmek için on="" kullanılır.

