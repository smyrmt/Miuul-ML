import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%3f" % x)
pd.set_option("display.width", 500)

def load():
    data = pd.read_csv("titanic.csv")
    return data
##############################################################################################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat  # tüm kategorikler
    cat_cols = [col for col in cat_cols if
                col not in cat_but_car]  # tüm kategoriklerden sahte kategorikleri çıkartarak gerçek kategorikler bulunur
    # numerik veriler
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Categorics: {len(cat_cols)}")
    print(f"numerics: {len(num_cols)}")
    print(f"cat but car: {len(cat_but_car)}")
    print(f"num but cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
    
def replace_with_thresholds(dataframe, var):
    low_limit, up_limit = outlier_thresholds(dataframe, var)
    dataframe.loc[(dataframe[var] < low_limit), var] = low_limit
    dataframe.loc[(dataframe[var] > up_limit), var] = up_limit
    
def missing_values(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
    
def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean(),}), end="\n\n\n")
        
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_col = [col for col in temp_df.columns if temp_df[col].dtypes == "O" and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_col:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var]) #rare_labels içindeyse "Rare" yaz değilse birşey yapma

    return temp_df

def one_hot_encoder(dataframe, cat_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first)
    return dataframe

#############################################################################################################################

df = load()
df.shape
df.head()
##########################################
# 1. Feature Engineering
##########################################
#Cabin bool
df["New_Cabin_Bool"] = df["Cabin"].notnull().astype("int")
#Name count
df["New_Name_Count"] = df["Name"].str.len()
#name word count
df["New_Name_Word_Count"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
#name dr
df["New_Name_Dr"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
#name title
df["New_Title"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)
#family size
df["New_Family_Size"] = df["SibSp"] + df["Parch"] + 1
#age_pclass
df["New_Age_Pclass"] = df["Age"] * df["Pclass"]
#is alone
df.loc[((df["SibSp"] + df["Parch"]) > 0), "New_Is_Alone"] = "NO"
df.loc[((df["SibSp"] + df["Parch"]) == 0), "New_Is_Alone"] = "YES"
#age level
df.loc[df["Age"] <= 18, "New_Age_Cat"] = "young"
df.loc[(df["Age"] >= 18) & (df["Age"] < 56) , "New_Age_Cat"] = "mature"
df.loc[df["Age"] > 56, "New_Age_Cat"] = "senior"
#Sex x Age
df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "New_Sex_Cat"] = "youngmale"
df.loc[(df["Sex"] == "male") & ((df["Age"] > 21) & df["Age"] <= 50), "New_Sex_Cat"] = "maturemale"
df.loc[(df["Sex"] == "male") & (df["Age"] > 50), "New_Sex_Cat"] = "seniormale"
df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), "New_Sex_Cat"] = "youngfemale"
df.loc[(df["Sex"] == "female") & ((df["Age"] > 21) & df["Age"] <= 50), "New_Sex_Cat"] = "maturefemale"
df.loc[(df["Sex"] == "female") & (df["Age"] > 50), "New_Sex_Cat"] = "seniorfemale"
df.head()
df.shape
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PassengerId" not in col]

######################################
# 2. Outliers
#####################################
for col in num_cols:
    print(col, check_outlier(df,col))
for col in num_cols:
    replace_with_thresholds(df,col)
for col in num_cols:
    print(col, check_outlier(df,col))

####################################
# 3. Missing Values
###################################
missing_values(df)
df.drop("Cabin", inplace=True, axis=1) #Cabil_bool değişkeni oluşturduğumuz için "Cabin" değişkenini siliyoruz
remove_cols = ["Ticket", "Name"]
df.drop(remove_cols, inplace=True, axis=1)

df["Age"] = df["Age"].fillna(df.groupby("New_Title")["Age"].transform("median")) # yaş değişkenindeki eksiklikler gider
# yaş değişkeninden oluşturulan değişkenlerin tekrar oluşturulması gerekir
#age_pclass
df["New_Age_Pclass"] = df["Age"] * df["Pclass"]
#age level
df.loc[df["Age"] < 18, "New_Age_Cat"] = "young"
df.loc[(df["Age"] >= 18) & (df["Age"] < 56), "New_Age_Cat"] = "mature"
df.loc[df["Age"] >= 56, "New_Age_Cat"] = "senior"
#Sex x Age
df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "New_Sex_Cat"] = "youngmale"
df.loc[(df["Sex"] == "male") & ((df["Age"] > 21) & df["Age"] <= 50), "New_Sex_Cat"] = "maturemale"
df.loc[(df["Sex"] == "male") & (df["Age"] > 50), "New_Sex_Cat"] = "seniormale"
df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), "New_Sex_Cat"] = "youngfemale"
df.loc[(df["Sex"] == "female") & ((df["Age"] > 21) & df["Age"] <= 50), "New_Sex_Cat"] = "maturefemale"
df.loc[(df["Sex"] == "female") & (df["Age"] > 50), "New_Sex_Cat"] = "seniorfemale"

#Embark değişkenindeki eksikleri (2) doldurmak için
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

# Not: ağaç yapılarında eksik değerleri doldurmaya gerek yok

############################################
# 4. Label Encoding
############################################
binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "int32", "float64"] and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

############################################
# 5. Rare Encoding
############################################
rare_analyser(df, "Survived", cat_cols)

df = rare_encoder(df, 0.01)
df["New_Title"].value_counts()

############################################
# 6. One-hot Encoding
############################################
ohe_cols = [col for col in df.columns if  10>= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)
df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PassengerId" not in col]
rare_analyser(df, "Survived", cat_cols) #one-hot encoding yapıldıktan sonra kullanışsız yeni sütunlar oluştu
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
#bu kullanışsız verileri silmek istersek;
#df.drop(useless_cols, axis=1, inplace=True)

############################################
# 7. Standart Scaler
############################################
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()

############################################
# 8. Model
############################################
y = df["Survived"] #bağımlı değişken
X = df.drop(["PassengerId", "Survived"], axis=1) #bağımsız değişkenler (survived ve passengerId dışındakiler
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import  RandomForestClassifier
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
#doğruluk: 0.8059

#Ham veriler üzerinden sınıflandırıcıyı çağırsaydık sonuç nasıl olurdu?
dff = load()
dff.dropna(inplace=True) #eksik değerler göz ardı edilmezse RF sınıflandırıcısı hata veriyor
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True) #RF sınıflandırıcısı kategorik veri kabul etmiyor sayısala çevirmemiz gerekli
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
# doğruluk: 0.7090

#yeni oluşturduğumuz etiketlerin önemini gözlemleyelim
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=0)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show(block=1)
    if save:
        plt.savefig("importances.png")

plot_importance(rf_model, X_train)
