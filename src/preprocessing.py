import pandas as pd

def preprocess(df:pd.DataFrame, stats:dict, frequent_tickets:list, config:dict) -> pd.DataFrame:

    """
    Предобработка данных
    """
    df = df.copy()
    pp_cfg = config.preprocessing

    df["Initial"] = df["Name"].str.extract(r"([A-Za-z]+)\.")
    df["Initial"] = df["Initial"].replace(pp_cfg.title_mapping)

    df["Ticket_type"] = df["Ticket"].apply(
        lambda x: x.split()[0] if isinstance(x, str) else "Unknown"
    )
    df["Ticket_type"] = df["Ticket_type"].apply(
        lambda x: x if x in frequent_tickets else "Other"
    )

    for title, median_val in stats["title_medians"].items():
        mask = (df["Initial"] == title) & (df["Age"].isnull())
        df.loc[mask, "Age"] = median_val

    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=list(pp_cfg.age_bins),
        labels=list(pp_cfg.age_labels),
        right=False,
    )

  
    for pclass, median_val in stats["fare_medians"].items():
        mask = (df["Pclass"] == pclass) & (df["Fare"].isnull())
        df.loc[mask, "Fare"] = median_val
    df["Fare"] = df["Fare"].fillna(train_df['Fare'].median() if 'train_df' in dir() else stats["fare_medians"].get(3, 14.45))

    df["Embarked"] = df["Embarked"].fillna(stats["embarked_mode"])


    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Pclass"] = df["Pclass"].astype(str) 

    cat_cols = ["Age_Group", "Ticket_type", "Initial", "Embarked", "Sex", "Pclass"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    df = df.drop(columns=["Name", "Cabin", "Age", "Ticket", "SibSp", "Parch"], axis=1)

    return df