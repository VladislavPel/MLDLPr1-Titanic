import pandas as pd
from config import config
from model import get_baseline_model


def main():
    test_df = pd.read_csv(config.paths.test_csv)
    baseline_model = get_baseline_model(test_df)
    baseline_model.to_csv(config.paths.baseline_submission_csv, index=False)

if __name__ == "__main__":
    main()
