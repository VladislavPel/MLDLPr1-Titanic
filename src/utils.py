import json
from pathlib import Path
import pandas as pd

def get_frequent_tickets(df:pd.DataFrame, min_count:int = 10) -> list:
    """Вычисляем самые частые типы билетов"""
    ticket_types = df['Ticket'].apply(lambda x: x.split()[0]
    )
    counts = ticket_types.value_counts()
    return counts[counts >= min_count].index.tolist()

def get_train_stats(train_df:pd.DataFrame, title_mapping:dict) -> dict:
    "Вычисляем медиану возраста по титулу на тренировочных данных"
    temp = train_df.copy()
    temp['Initial'] = temp['Name'].str.extract(' ([A-Za-z]+)\.')
    temp['Initial'] = temp['Initial'].replace(title_mapping)

    fare_medians = train_df.groupby('Pclass')['Fare'].median().to_dict()

    embarked_mode = train_df['Embarked'].mode()[0]

    title_medians = temp.groupby('Initial')['Age'].median().to_dict()
    return {
        'title_medians': title_medians,
        'fare_medians': fare_medians,
        'embarked_mode': embarked_mode,
    }

def save_submission(predictions, passenger_ids, path: str) -> None:
    """
    Сохраняет предсказания в формате CSV
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions,
    })
    submission.to_csv(path, index=False)

def save_metrics(metrics: dict, path: str) -> None:
    """
    Сохраняет метрики эксперимента и гиперпараметры в JSON файл.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)