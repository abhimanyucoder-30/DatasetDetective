import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, SGDClassifier,SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def model_finder(X,y, job):
    class_list= [LogisticRegression(),RandomForestClassifier(random_state=42), SGDClassifier(random_state=42)]
    reg_list= [LinearRegression(), Lasso(), Ridge(), SGDRegressor(random_state=42)]

    d1={}
    match job:
        case "regression":
            for a in reg_list:
                a.fit(X,y)
                score= cross_val_score(a,X,y, cv=5).mean()
                d1[a]= score
                print("{}\n------\nScore: {}".format(a,score))

        case "classification":
            for a in class_list:
                a.fit(X,y)
                score= cross_val_score(a,X,y, cv=5).mean()
                d1[a]= score
                print("{}\n------\nScore: {}".format(a,score))
    return d1

def scale(df):
    
    scaler= StandardScaler()
    for a in num_attr:
        scaler.fit_transform(np.array(df[a]))

def perform_eda(df, cols):
    target= input("Enter your target feature:").lower()
    for a in cols:
        if target==a.lower():
            target=a
            break

    #scaling the values
    df_scaled= scale(df)
    
    #splitting into target and independent features
    X= df_scaled.drop(target, axis=1)
    y= df_scaled[[target]]

    #splitting into train and test set
    X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2)
    #find the best suited model for it
    job= input("Enter if your job is regression or classification:").lower()
    model_finder(X_train, y_train, job)

   

if __name__=="__main__":
    df= pd.read_csv("./titanic.csv")
    cols= df.columns
    num_attr=df.select_dtypes(include="number").columns.to_list()
    cat_attr= df.select_dtypes(include=["category","object"]).columns.to_list()
    for a in num_attr:
        if set(list(df[a].unique()))==set([0,1]):
            cat_attr.append(a)
            num_attr.remove(a)
    
    
