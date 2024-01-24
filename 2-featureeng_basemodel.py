################### FEATURE ENGINEERING############

##MISSING VALUE ANALYSIS####
from libs import pd 
from helpers import missing_values_table

df = pd.read_csv("datasets\churn_telco2.csv")

df.isnull().sum()


na_columns = missing_values_table(df, na_name=True)
#df.drop(df[df["TotalCharges"].isnull()].index, axis=0)


# df[df["TotalCharges"].isnull()]["TotalCharges"] = df[df["TotalCharges"].isnull()]["MonthlyCharges"]

df.iloc[df[df["TotalCharges"].isnull()].index,19] = df[df["TotalCharges"].isnull()]["MonthlyCharges"]

df["tenure"] = df["tenure"] + 1
df[df["tenure"]==1]

# TotalCharges direkt sıfıra eşitleyebiliriz.


#ILLEGAL VALUE ANALYSIS 
from helpers import outlier_thresholds , check_outlier , replace_with_thresholds 
from helpers import grab_col_names
cat_cols, num_cols, cat_but_car = grab_col_names(df)


for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


################ BASE MODEL#######
from libs import LogisticRegression, KNeighborsClassifier , DecisionTreeClassifier,RandomForestClassifier,XGBClassifier,LGBMClassifier,CatBoostClassifier,pd
dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["Churn"]
X = dff.drop(["Churn","customerID"], axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    from libs import cross_validate
    cv_results =cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"],error_score='raise')
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

########## LR ##########
Accuracy: 0.8036
Auc: 0.8432
Recall: 0.542
Precision: 0.6583
F1: 0.5941
########## KNN ##########
Accuracy: 0.7632
Auc: 0.7464
Recall: 0.4468
Precision: 0.5694
F1: 0.5002
########## CART ##########
Accuracy: 0.728
Auc: 0.6586
Recall: 0.5077
Precision: 0.4886
F1: 0.4977
########## RF ##########
Accuracy: 0.792  
Auc: 0.8252      
Recall: 0.4842   
Precision: 0.6448
F1: 0.5529       
########## XGB ##########
Accuracy: 0.7833 
Auc: 0.8228      
Recall: 0.5072   
Precision: 0.6123
F1: 0.5542    
########## LightGBM ##########
Accuracy: 0.7982
Auc: 0.8373
Recall: 0.5281
Precision: 0.6482
F1: 0.5816   
########## CatBoost ##########
Accuracy: 0.797
Auc: 0.8401
Recall: 0.5051
Precision: 0.6531
F1: 0.5691


df.to_csv("churn_telco3.csv", index=False)