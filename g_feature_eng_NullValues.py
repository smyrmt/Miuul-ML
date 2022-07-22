#########################################################
#eksik değerleri yakalama
########################################################

def missing_values(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
df = load()
missing_values(df, True)

#########################################################
#Eksik değer prob. çözme
#######################################################
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0) #numerik sütunlardaki boş değerleri ortalama ile değiştir
dff.isnull().sum().sort_values(ascending=False) #boş değerleri tekrar tarayalım
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

########################################################
# Kategorik Değişken Kırılımında değer atama
######################################################
#cinsiyete göre kırılımda boş değerlere yaş ortalamalarının yazdırılması
df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]
df.isnull().sum()

########################################################
# Tahmine dayalı atama işlemi
######################################################
df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()

#değişkenlerin standardizasyonu
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

#KNN uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns) #standardizasyon işleminin geri alınması
df["age_imputed_knn"] = dff[["Age"]]
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
#####################################################
#Eksik veri yapısının grafikler ile analiz edilmesi
#####################################################
msno.bar(df)
plt.show(block=True)

msno.matrix(df)
plt.show(block=True)

msno.heatmap(df)
plt.show(block=True)

#####################################################
# Eksik Değerlerin Bağımlı Değişken ile ilişkisinin incelenmesi
#####################################################
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags =temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_values(df, True)
na_cols = missing_values(df, True)
missing_vs_target(df, "Survived", na_cols)
