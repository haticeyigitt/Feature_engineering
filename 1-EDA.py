
from libs import pd 
from libs import warnings


warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("datasets\churn_telco.csv")
df.head()
df.shape
df.info()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
#pd.to_numeric() ile sütun veya seriyi sayısal formata dönüştürür, errors='coerce' ise hatalı dönüşüm olursa Nan (not a number) olarak doldurur.

df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)
df.head()

########
#EDA (Exploratory data analysis-Keşifçi Veri Analizi)
#######

from helpers import check_df
check_df(df)


#GRAB CATEGORICAL AND NUMERICAL VARIABLES

from helpers import grab_col_names

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car



#CATEGORICAL VARIABLES 

from helpers import cat_summary



for col in cat_cols:
    cat_summary(df, col)


##### GÖZLEM ####
    #Kadın - erkek sayısı yaklaşık olarak eşit
    # Customerların ½50 'si evli 
    # %30 'luk kısmın bakmakla yükümlü olduğu kişi var.
    # %90 telefon hizmeti var
    # %90 lık kısmın %53' ü tek hatta sahip
    # %21 ' lik kısımda internet servisi yok.
    # Çok sayıda müşteri aydan aya sözleşme yapıyor
    # %60 kağıtsız ( e- fatura) bulunmakta
    # %26 sı geçen ay ayrılmış
    # %16 yaşlı çoğu genç




#NUMERICAL VARIABLES 

from helpers import num_summary

for col in num_cols:
    num_summary(df, col, plot=True)


##### GÖZLEM ######
    # Tenure'e bakıldığında 1 aylık müşterilerin çok fazla olduğunu görebiliriz.



#ANALYSIS OF NUMERICAL VARIABLES BY TARGET
    from helpers import target_summary_with_num

for col in num_cols:
    target_summary_with_num(df, "Churn", col)


##### GÖZLEM #####
# Tenure ve Churn ilişkisine baktığımızda Churn olmayan müşterilerin daha uzun süredir müşteri olduklarını gözlemleyebilriz.
# monthlycharges ve Churn incelendiğinde Churn olan müşterilerin ortalama aylık ödemeleri daha fazla.
    


#ANALYSIS OF CATEGORICAL VARIABLES BY TARGET
    from helpers import target_summary_with_cat

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)



##### GÖZLEM #####
    # Kadın- erkek churn oranı eşit neredeyse
    # Partner ve dependents'i olan müşterilerin churn oranı daha düşük
    # PhoneServise ve MultipleLines'da fark yok
    # Fiber Optik İnternet Servislerinde kayıp oranı çok daha yüksek
    # Kağıtsız faturalandırmaya sahip olanların churn oranı daha fazla
    # ElectronicCheck PaymentMethod'a sahip müşteriler, diğer seçeneklere kıyasla platformdan daha fazla ayrılma eğiliminde
    # Yaşlı müşterilerde churn yüzdesi daha yüksektir
    # No OnlineSecurity , OnlineBackup ve TechSupport gibi hizmetleri olmayan müşterilerin churn oranı yüksek
    # Bir veya iki yıllık sözleşmeli Müşterilere kıyasla, aylık aboneliği olan Müşterilerin daha büyük bir yüzdesi churn



#CORRELATION

df[num_cols].corr()

# Korelasyon Matrisi
from libs import plt , sns 
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# TotalChargers'in aylık ücretler ve tenure ile yüksek korelasyonlu olduğu görülmekte

df.to_csv("churn_telco2.csv", index=False)


