import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from eda import perform_eda

df= pd.read_csv("./titanic.csv")

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
cols= list(df.columns)
num_attr=df.select_dtypes(include="number").columns.to_list()
cat_attr=df.select_dtypes(include=["object","category"]).columns.to_list()
print("Numerical attributes:{} [{}]".format(num_attr,df.describe().shape[1]))
print("Categorical Attributes:{} [{}]".format(cat_attr, len(cat_attr)))

print("Number of Null values in each column:")
for a in cols:
    print("{}:{}".format(a, df[a].isnull().sum()))

corr_mtx= df[num_attr].corr()
print("Correlation Matrix:\n{}".format(corr_mtx))

sns.pairplot(df)
plt.savefig("./plots/pairplot.png")
print("You can check the pairplot generated here:- /plots")

print("Duplicate rows:")
print("\tNumber:",df.duplicated().sum())
print("\tPercentage:", df.duplicated().sum()/len(df),"% of the entire dataset")
print("\tName:", end=" ")
print(df[df.duplicated()])
df_cleaned= df.drop_duplicates()
print("\tAfter dropping:\n", df_cleaned.head(5))

print("Removing Nan values from all columns (by replacing them with their medians):")
num_ppl= Pipeline([
    ('num_ppl', SimpleImputer(strategy="median"))
])
for a in num_attr:
    a_list= list(df_cleaned[a].unique())
    if set(a_list)==set([0,1]):
        cat_attr.append(a)
        num_attr.remove(a)
        continue
    df_cleaned[num_attr]= pd.DataFrame(num_ppl.fit_transform(df[num_attr]), columns= num_attr)
print(df[num_attr].isnull().sum())

cat_ppl= Pipeline([
    ("cat_ppl", SimpleImputer(strategy="most_frequent"))
])
df_cleaned[cat_attr]= pd.DataFrame(cat_ppl.fit_transform(df_cleaned[cat_attr]), columns=cat_attr)
df_cleaned.info()

print("Outliers:\n------------------------")
for a in num_attr:
    q1= df_cleaned[a].quantile(0.25)
    q3= df_cleaned[a].quantile(0.75)
    iqr= q3-q1
    if iqr==0:
        print("\n{}\n------".format(a))
        print("Normal outlier calculation not possible for this feature\n")
    else:
        lb = q1- 1.5*iqr
        ub= q3 + 1.5*iqr
        num_outliers= len(df_cleaned[a][(df_cleaned[a]<lb) | (df_cleaned[a]>ub)])
        print("\n{}\n------".format(a))
        print("Number:", num_outliers)
        print("q1:", q1)
        print("q3:",q3)
        print("iqr:", iqr)
        print("\n")


print("Calculations for numerical attributes:\n--------------------")
print("Skewness\n------------")
for a in num_attr:
    print("{}:{}".format(a, df_cleaned[a].skew()))

perform_eda(df_cleaned, cols)






