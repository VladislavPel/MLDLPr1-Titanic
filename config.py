from omegaconf import OmegaConf

config = {
    'general': {
        'experiment_name': 'titanic_full_exp',
        'seed': 42,
    },
    
    'hyperparams': {
        'learning_rate': 0.05,
        'n_estimators': 300,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbose': False,
    },

    'paths': {
        'train_csv': 'data/train.csv',
        'test_csv': 'data/test.csv',
        'baseline_submission_csv': 'results/baseline_submission.csv',
        'metrics_json': 'results/metrics.json',
        'best_submission_csv': 'results/best_submission.csv',
    },

    'preprocessing': {
        'min_ticket_counts': 10,
        'age_bins': [0, 12, 18, 35, 60, 100],
        'age_labels': ['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior'],
        'title_mapping': {
            'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss',
            'Dr': 'Mr', 'Major': 'Mr', 'Lady': 'Mrs',
            'Countess': 'Mrs', 'Jonkheer': 'Mr', 'Col': 'Mr',
            'Rev': 'Mr', 'Capt': 'Mr', 'Sir': 'Mr', 'Don': 'Mr',
        },
    },

    'models': {
        'run_baseline': True,
        'run_logistic_regression': True,
        'run_lasso': True,
        'run_ridge': True,
        'run_elasticnet': True,
        'run_knn': True,
        'run_decision_tree': True,
        'run_random_forest': True,
        'run_xgb': True,
        'run_catboost': True,
        'run_lightgbm': True,
        'run_dnn': True,

        
        'baseline_params': {'C': 1.0, 'max_iter': 1000},
        'lasso_params': {'alpha': 1.0, 'max_iter': 1000},
        'ridge_params': {'alpha': 1.0, 'max_iter': 1000},
        'elasticnet_params': {'alpha': 1.0, 'l1_ratio': 0.5, 'max_iter': 1000},

        'knn_params': {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'minkowski'},

        'decision_tree_params': {'max_depth': 5, 'random_state': '${general.seed}'},
        'random_forest_params': {
            'n_estimators': '${hyperparams.n_estimators}',
            'max_depth': '${hyperparams.max_depth}',
            'random_state': '${general.seed}'
        },

        'xgb_params': {
            'n_estimators': '${hyperparams.n_estimators}',
            'max_depth': '${hyperparams.max_depth}',
            'learning_rate': '${hyperparams.learning_rate}',
            'subsample': '${hyperparams.subsample}',
            'colsample_bytree': '${hyperparams.colsample_bytree}',
            'verbose': '${hyperparams.verbose}',
            'random_state': '${general.seed}',
        },
        'catboost_params': {
            'iterations': '${hyperparams.n_estimators}',
            'depth': '${hyperparams.max_depth}',
            'learning_rate': '${hyperparams.learning_rate}',
            'verbose': '${hyperparams.verbose}',
            'random_state': '${general.seed}',
        },
        'lgbm_params': {
            'n_estimators': '${hyperparams.n_estimators}',
            'max_depth': '${hyperparams.max_depth}',
            'learning_rate': '${hyperparams.learning_rate}',
            'subsample': '${hyperparams.subsample}',
            'colsample_bytree': '${hyperparams.colsample_bytree}',
            'verbose': '${hyperparams.verbose}',
            'random_state': '${general.seed}',
        },

        'dnn_params': {
            'hidden_layers': [128, 64, 32],
            'dropout': 0.3,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': '${hyperparams.learning_rate}',
            'use_batchnorm': True,
            'optimizer': 'adam', # или 'sgd'
        },
    },

    'training': {
        'cv_folds': 5,
        'debug': False,
        'number_of_train_debug_samples': 100,
    },
    
    'logging': {
        'prints': True,
    }
}

config = OmegaConf.create(config)