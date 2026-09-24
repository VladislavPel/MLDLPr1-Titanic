import pandas as pd
from src.utils import get_frequent_tickets, get_train_stats
from src.preprocessing import preprocess


def load_data(cfg):
    train = pd.read_csv(cfg.paths.train_csv)
    test = pd.read_csv(cfg.paths.test_csv)
    return train, test


def prepare_data(train, test, cfg):
    stats = get_train_stats(train, cfg.preprocessing.title_mapping)
    frequent_tickets = get_frequent_tickets(train, cfg.preprocessing.min_ticket_counts)

    train_proc = preprocess(train, stats, frequent_tickets, cfg)
    test_proc = preprocess(test, stats, frequent_tickets, cfg)

    X_train = train_proc.drop(columns=["PassengerId", "Survived"])
    y_train = train_proc["Survived"]
    X_test = test_proc.drop(columns=["PassengerId"])
    test_ids = test["PassengerId"]


    return X_train, y_train, X_test, test_ids