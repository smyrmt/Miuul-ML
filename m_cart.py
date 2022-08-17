# pip install pydotplus
# pip install skompiler
# pip install astor
# pip install joblib

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%3f" % x)
pd.set_option("display.width", 500)

df = pd.read_csv("diabetes.csv")
df.head()
df.shape

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)
y_pred = cart_model.predict(X)
y_prob = cart_model.predict_proba(X)[:, 1]
print(classification_report(y, y_pred))
roc_auc_score(y, y_prob)
# doğruluk 1 çıktı, bunu valide etmeliyiz

# holdout ile başarı değerlendirme
############################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# train hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# test hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

# cv ile başarı değerlendirme
############################
cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)
cv_results = cross_validate(cart_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

# hyperparameter with gridsearchcv
cart_model.get_params()
cart_params = {"max_depth": range(1, 11), "min_samples_split": range(2, 20)}
cart_best_grid = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=-1, verbose=1, scoring="accuracy").fit(X, y)

cart_best_grid.best_params_
cart_best_grid.best_score_

random = X.sample(1, random_state=45)
cart_best_grid.predict(random)

#final model

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X,y)
cart_final.get_params()
# en iyi parametreyi kullanmak için diğer bir yol
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(cart_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()

#hiperparametre optimizasyonu ile başarı sonuçları arttı

###################################################
# Feature Importance
###################################################
cart_final.feature_importances_ #anlaşılabilir bir formatta değil

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig("importances.png")

plot_importance(cart_final, X)

###############################################
# Öğrenme eğrileriyle model karmaşıklığını analiz etme
######################################################
# her array bir parametre değerine karşı elde edilen 10 katlı sonuçların değerini verir
train_score, test_score = validation_curve(cart_final, X, y, param_name="max_depth", param_range=range(1,11), scoring="roc_auc", cv=10)

mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)
    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label="Training Score", color="b")
    plt.plot(param_range, mean_test_score, label="Validation Score", color="g")
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Nuumber of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show(block=True)

val_curve_params(cart_final, X, y, "max_depth", range(1,11), scoring="f1")

# birden fazla parametre seti için
cart_val_params = [["max_depth", range(1,11)], ["min_samples_split", range(2,20)]]
for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])

#############################################
# Visualization
#############################################
#conda install graphviz
#import graphviz

# karar ağacının görselleştirilmesi
def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")

#############################################
# Extracting Decission Rules
#############################################
tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)

#############################################
# Extracting Python codes of decission rules
#############################################
# pip install scikit-learn==0.23.1 (sorun çıkarsa bu kurulumu yapmalısın)
# import sklearn
# sklearn.__version__

print(skompile(cart_final.predict).to("python/code"))
# pip install sqlalchemy
print(skompile(cart_model.predict).to("sqlalchemy/sqlite"))

####################################################
# Prediction using Python codes
#############################################

def predict_with_rules(x):
    return() #return içine daha önceden bulduğumuz python kodlarını yazacağız

x = [12, 13, 20, 23, 4, 55, 12, 7]
predict_with_rules(x)

####################################################
# Saving and Loading Model
#############################################
# kaydetme
joblib.dump(cart_final, "cart_final.pkl")
# geri çağırma
cart_model_from_disk = joblib.load("cart_final.pkl")
x = [12, 13, 20, 23, 4, 55, 12, 7]
cart_model_from_disk.predict(pd.DataFrame(x).T) #predict methodunu kullandığımız için dataframe'e çevirmemiz gerekiyor

