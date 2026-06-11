import pandas as pd
from sklearn.preprocessing import LabelEncoder  #LabelEncoder -> converts text (categories) into numbers

def preprocess(df):

    # Fill missing values
    df = df.ffill() #ffill() : forward fill, missing values are filled with the previous value in the column

    # Encode categorical columns
    le = LabelEncoder() #LabelEncoder() : Converts text (categories) into numbers
    for col in df.select_dtypes(include='object'): #Select columns with data type 'object' (categorical columns)
        df[col] = le.fit_transform(df[col]) #fit(): find unique values and transform(): convert text to numbers

    return df