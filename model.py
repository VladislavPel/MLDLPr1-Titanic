import pandas as pd
from config import config

def get_baseline_model(test_df: pd.DataFrame) -> pd.DataFrame:
    predictions = (test_df['Sex'] == 'female').astype(int)
    return pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
    
