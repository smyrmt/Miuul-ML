#####################################################################################################################
############# ENCODING SCALING ################
#####################################################################################################################

######################################################################
# Label Encoding & Binary Encoding
######################################################################
df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]
le.inverse_transform([0, 1])  # sayıların temsil ettiği etiket değerlerini gösterir


def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe


df = load()
binary_cols = [col for col in df.columns if (df[col].dtype not in ["int64", "float64"]) & (df[col].nunique() == 2)]
df.info()
binary_cols

for col in binary_cols:
    label_encoder(df, col)
df.head()

df = load_application_train()
df[binary_cols].head()


### not: Label encoder NA değerleri de doldurur!!!!!!

##########################################
# One-hot Encoding
#########################################
# drop_first değeri ile ilk sütun kaldırılır birbiri üzerinden oluşturulma durumunu kaldırmak için (dummy değişken tuzağı)
def one_hot_encoder(dataframe, cat_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first)
    return dataframe


df = load()
ohe_cols = [col for col in df.columns if 2 < df[col].nunique() <= 10]
one_hot_encoder(df, ohe_cols).head()

############################################
# Rare Encoding
############################################
df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)

#################################################
# Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analizi
################################################

df["NAME_INCOME_TYPE"].value_counts()
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean(),}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

#################################################
# Rare Encoder
################################################
def rare_encoder(dataframe, rare_perc):



new_df= rare_encoder(df, 0.01) #0.01 oranının altında kalan sınıfları birleştirir
rare_analyser((new_df, "TARGET", cat_cols))


###################################################
# Özellik Ölçeklendirme
###################################################
################
#Standart Scaler: z = (x-u) / s (ortalamayı çıkar standat sapmaya böll)
################
df = load()
ss = StandardScaler()
df["Age_standart_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

################
#Robust Scaler: medyanı çıkar iqr'a böll
################
rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])

################
#MinMax Scaler: verilen iki değer arasında değişken dönüşümü
################
mms = MinMaxScaler()
df["Age_minmax_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

age_cols = [col for col in df.columns if "Age" in col] #Age değişkenine ait verileri ve ölçeklendirme sonuçlarını almak için

def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    print(dataframe[num_col].describe(quantiles).T)

    if plot:
        dataframe[num_col].hist(bins=20)
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show(block=True)

#Scaler değişse de değerlerde değişme olmadığını grafik üzerinde görüntüleyelim.
#Veriler değişmez sadece ifade ediliş biçimleri değişir
for col in age_cols:
    num_summary(df,col,plot=True)


####################
#Num to Cat (Bining)
####################
df["Age_cut"] = pd.qcut(df["Age"], 5)
df.head()


#####################################################################################################################
############# FEATURE EXTRACTION ################
#####################################################################################################################

##################################
# Binary Features
##################################
df = load()
df.head()
df["New_Cabin_Bool"] = df["Cabin"].notnull().astype("int") #Cabin etiketine ait veriler dolu ise 1 boş ise 0 yazdır
df.groupby("New_Cabin_Bool").agg({"Survived": "mean"})

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["New_Cabin_Bool"] == 1, "Survived"].sum(), #kabin numarası olan hayatta kalan kaç kişi var
                                            df.loc[df["New_Cabin_Bool"] == 0, "Survived"].sum()],
                                      nobs=[df.loc[df["New_Cabin_Bool"] == 1, "Survived"].shape[0], #kabin numarası olan kaç kişi var
                                            df.loc[df["New_Cabin_Bool"] == 0, "Survived"].shape[0]])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue)) #p-value değerinin 0.5 ten küüçük olması iki değişken arasında fark olduğu anlamına gelir

df.loc[((df["SibSp"] + df["Parch"]) > 0), "New_Is_Alone"] = "NO" #yalnız ise no yazdır
df.loc[((df["SibSp"] + df["Parch"]) == 0), "New_Is_Alone"] = "YES" #ailesi varsa yes yazdır

df.groupby("New_Is_Alone").agg({"Survived": "mean"})

######################################
# Text'ler üzerindern özellik türetme
######################################

df.head()
###############
# Letter Count
df["New_Name_Count"] = df["Name"].str.len()

###############
# Word Count
df["New_Name_Word_Count"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

###############
# Özel yapıları yakalama
df["New_Name_Dr"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df.groupby("New_Name_Dr").agg({"Survived": ["mean", "count"]})

######################################
# Regex ile değişken türetme
######################################
df["New_Title"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False) #boşlukla başlayıp nokta ile biten değerleri ayrı bir sütuna al
df[["New_Title", "Survived", "Age"]].groupby(["New_Title"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

