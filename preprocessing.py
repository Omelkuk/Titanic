import pandas as pd

def clean_data(df):
    df = df.copy()
    #feature engineering (ДЕКОМПОЗИЦИЯ ФИЧЕЙ) признаки статуса
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z] +)\.",expand=False)
    rare_titles = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    #interaction 
    df['Age'] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))
    df["age_Pclass"] = df["Age"] * df["Pclass"]

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    #Признаки семьи (КОМБИНАЦИЯ ФИЧЕЙ)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    #палуба
    df["Deck"] = df["Cabin"].str.get(0).fillna("Unknown")

    df = pd.get_dummies(df, columns=["Sex", "Embarked","Title", "Deck"], drop_first=True)
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df = df.fillna(0)
    return df