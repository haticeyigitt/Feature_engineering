################# FEATURE ENGINEERING #####################

#MEVCUT BİLGİLER KULLANILARAK YENİ LABELLAR ÜRETİLEREK MODEL GÜÇLENDİRİLEBİLİR.
from libs import  pd 
df=pd.read_csv("datasets\churn_telco3.csv")

# Tenure ile yıllık kategorik değişken oluşturulursa:

df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"


# Herhangi bir destek, yedek veya koruma almayan müşteriler değerlendirilebilir :

df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)


# Müşterinin toplam aldığı servis sayısı önemli olabilir:

df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)



# Herhangi bir streaming hizmeti alan müşteriler değerlendirilsin:

df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)


# Müşteri otomatik ödeme yapıyor mu?

df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)



# Ortalama aylık ödeme :
df["NEW_AVG_Charges"] = df["TotalCharges"] / df["tenure"]



# Güncel Fiyatın ort. fiyata göre artışı :
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]


# Servis başına ödenen ücret:
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)



df.head()
df.shape


############# ENCODING ###########
# SEPARATION OF VARIABLES ACCORDING TO TYPES
from helpers import grab_col_names
from libs import LabelEncoder
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# ONE - HOT ENCODING

# Update process of cat_cols list
from helpers import one_hot_encoder


cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols


df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()


#MODELLING

from libs import LogisticRegression, KNeighborsClassifier , DecisionTreeClassifier,RandomForestClassifier,XGBClassifier,LGBMClassifier,CatBoostClassifier,pd
y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)


models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    from libs import cross_validate
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"],error_score='raise')
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")


########## LR ##########
Accuracy: 0.8005
Auc: 0.8403
Recall: 0.503
Precision: 0.6645
F1: 0.5715
########## KNN ##########
Accuracy: 0.7694
Auc: 0.7537
Recall: 0.4644
Precision: 0.5833
F1: 0.5163
########## CART ##########
Accuracy: 0.7258
Auc: 0.6554
Recall: 0.5013
Precision: 0.4841
F1: 0.4924
########## RF ##########
Accuracy: 0.7945
Auc: 0.8282
Recall: 0.4917
Precision: 0.65
F1: 0.5595
########## XGB ##########
Accuracy: 0.7808
Auc: 0.8226
Recall: 0.4965
Precision: 0.6073
F1: 0.5458
########## LightGBM ##########
Accuracy: 0.7947
Auc: 0.8366
Recall: 0.527
Precision: 0.6376
F1: 0.5768
########## CatBoost ##########
Accuracy: 0.7982 
Auc: 0.8421      
Recall: 0.5163   
Precision: 0.6519
F1: 0.576  