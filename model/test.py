# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:51:57 2023

@author: tymot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import shap
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')
np.random.seed = 42



os.chdir(r"...")

df = pd.read_csv('train.csv')
df.head()


# PREPROCESSING
from sklearn import preprocessing



y = np.array(df['Credit_Score'])
def preprocess(df):
    
    
    def del__from_column(col):
        col = col.astype(str).replace("_",'')
        return pd.to_numeric(col, errors='coerce')
    
    
    # id i name - niepotrzebne

    df = df.drop(columns = ["ID", "Name"])
    
    # customer_id - przekształcamy na label_encoding
    
    le = preprocessing.LabelEncoder()
    le.fit(df["Customer_ID"])
    
    df["Customer_ID"] = le.transform(df["Customer_ID"])
    
    # month dict 
    # przekształcimy sobie miesiące
    month_dict = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5,
                  "June": 6, "July": 7, "August": 8}
    
    df["Month"] = df["Month"].map(month_dict)
    
    
    # przekształcimy sobie wiek
    
    df["Age"] = df['Age'].str.replace("\D+", '').astype(int)
    
    
    df["Age"] = np.where((0 > df["Age"]), -df["Age"], df["Age"])
    df['Age'] = np.where((0 <= df['Age']) & (df['Age'] <= 100), df['Age'], np.mean(df["Age"]))
    
    
    # SSN - social security number - raczej niepotrzebne, bedziemy sprawdzac czy jest taki goofy czy nie - moze to da jakies informacje
    # ale na razie drop
    
    df = df.drop(columns = ["SSN"])
    
    # occupation
    
    
    # annual income 
    
    df["Annual_Income"] = del__from_column(df['Annual_Income'])
    
    
    # num of loans
    
    df["Num_of_Loan"] = del__from_column(df['Num_of_Loan'])
    
    # type loan 
    
    df = df.drop(columns=["Type_of_Loan"])
    
    # changed_credit_limit
    
    
    df["Changed_Credit_Limit"] = del__from_column(df['Changed_Credit_Limit'])
    
    # credit_history
    
    def extract_age(age_string):
        if pd.isna(age_string):
            return 0
        else:
            return int(age_string.split()[0])*12 + int(age_string.split()[3])
    
    # Apply the lambda function to the 'age' column of the DataFrame
    df["Credit_History_Age_In_Months"] = df['Credit_History_Age'].apply(extract_age)
    
    
    df = df.drop(columns = ["Credit_History_Age"])
    
    # payment_behaviour
    
    #df["Payment_Behaviour"] = df["Payment_Behaviour"].replace('!@9#%8', 'Unknown_spent_Unknown_value_payments')
    #split_payment = lambda x: (x.split("_")[0], x.split("_")[2])
    #df[['Spent_Level', 'Value_Payment']] = df["Payment_Behaviour"].apply(split_payment)

    # Num_of_Delayed_Payment
    
    
    df["Num_of_Delayed_Payment"] = del__from_column(df["Num_of_Delayed_Payment"])
    df["Monthly_Balance"] = del__from_column(df["Monthly_Balance"])
    df["Amount_invested_monthly"] = del__from_column(df["Amount_invested_monthly"])
    df["Total_EMI_per_month"] = del__from_column(df["Total_EMI_per_month"])
    df["Outstanding_Debt"] = del__from_column(df["Outstanding_Debt"])

    
    
    return df.select_dtypes([np.number]) # to do usuniecia
    
    
df = preprocess(df)



y = np.array(df['Credit_Score'])
X = df.drop(['Credit_Score'],axis=1)
#X = df.fillna(0)


na_ratio_cols = data.isna().mean(axis=0)
na_ratio_cols



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)



pd.Series(y_val).hist()




from sklearn import metrics

def gini_roc(y_test, y_pred_proba, tytul):
    
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    
    plt.plot(fpr,tpr)
    plt.title(tytul)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    roc_auc = metrics.auc(fpr, tpr)
    gini = (2 * roc_auc) - 1

    return gini

def gini_train_val(model, X_train, y_train, X_val, y_val):
    
    y_pred_proba = model.predict_proba(X_train)[::,1]
    gini_train = gini_roc(y_train, y_pred_proba, "ROC Curve for Training Sample")
    print("gini_train: %.4f" % gini_train)
    
    y_pred_proba = model.predict_proba(X_val)[::,1]
    gini_val = gini_roc(y_val, y_pred_proba, "Roc Curve for Validation Sample")
    print("gini_val: %.4f" % gini_val)

    return

def shapley(model, X_train, X_val):
        
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    # model = lr
   
    explainer = shap.Explainer(model, X_train)
    
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train)
    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0])

    # freature importance    
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    
    shap.plots.bar(shap_values)
    shap.summary_plot(shap_values, plot_type='violin')
    shap.plots.bar(shap_values[0])
    shap.plots.waterfall(shap_values[0])
    shap.plots.force(shap_values[0])
    
    
    shap.plots.force(shap_values[1])
    
    shap.plots.heatmap(shap_values)
    
    # fig = shap.force_plot(explainer.expected_value, shap_values.values, X_train, feature_names = X_train.columns)
    # fig.savefig('testplot.png')
    # fig.plot()
    
    # fig = shap.force_plot(shap_values, X_train)
    # fig.plot()
   
    shap_values = explainer(X_val)
    shap.plots.beeswarm(shap_values)
    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0])
    
    # freature importance
    shap.summary_plot(shap_values, X_val, plot_type="bar")    
    


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=2000)

lr.fit(X_train, y_train)
y_proba = lr.predict_proba(X_val)
y_hat = lr.predict(X_val)
print("proba: " + str(y_proba[0:10,0]) + '\ny:     ' + str(y_hat[0:10]) + '\ny_hat: ' + str(y_val[0:10]))

gini_train_val(lr, X_train, y_train, X_val, y_val)
shapley(lr, X_train, X_val)

y_pred_proba = lr.predict_proba(X_val)[::,1]

b = pd.DataFrame(y_hat, columns=["y_hat"])
c = pd.DataFrame(y_pred_proba, columns=["PD"])
a = pd.merge(b, c, left_index=True, right_index=True)


score = metrics.accuracy_score(y_val, y_hat)
print(score)
metrics.confusion_matrix(y_val, y_hat)


from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text, export_graphviz
## biblioteka poniżej może być problematyczna na Windows
#import graphviz

tree1 = DecisionTreeClassifier()

# class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', 
#     splitter='best', 
#     max_depth=None, 
#     min_samples_split=2, 
#     min_samples_leaf=1, 
#     min_weight_fraction_leaf=0.0, 
#     max_features=None, random_state=None, 
#     max_leaf_nodes=None, min_impurity_decrease=0.0, 
#     class_weight=None, ccp_alpha=0.0)[source]

tree1.fit(X_train,y_train)
y_proba = tree1.predict_proba(X_val)
y_hat = tree1.predict(X_val)
print("proba: " + str(y_proba[0:10,0]) + '\ny:     ' + str(y_hat[0:10]) + '\ny_hat: ' + str(y_val[0:10]))

gini_train_val(tree1, X_train, y_train, X_val, y_val)

explainer = shap.TreeExplainer(tree1, X_train)
shap_values = explainer(X_train)
tmp = shap.Explanation(shap_values[:, :, 1], data=X_train, feature_names=X_train.columns)
shap.plots.beeswarm(tmp)

shap_values = explainer(X_val)
tmp = shap.Explanation(shap_values[:, :, 1], data=X_val, feature_names=X_train.columns)
shap.plots.beeswarm(tmp)

text_representation = export_text(tree1)
print(text_representation)

plt.figure(figsize=(25,20))
splits = plot_tree(tree1, filled=True)

# opcja 2
fig = plt.figure(figsize=(25,20))
_ = plot_tree(tree1, 
                    feature_names=X_train.columns,  
                    class_names="target",
                    filled=True)

print(cross_val_score(tree1, X, y, scoring="accuracy", cv = 7))

