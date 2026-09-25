from omegaconf import OmegaConf

config = {
    'general': {
        #'experiment_name': 'titanic_lasso',
        'experiment_id': 24,       
        'seed': 42,
        'active': 'xgb',         
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
        'gender_submission_csv': 'results/gender_submission.csv',
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
            'Dona': 'Mrs',
        },
    },

    'training': {
        'cv_folds': 5,
        'debug': False,
        'number_of_train_debug_samples': 100,
    },

    'logging': {
        'prints': True,
    },

    'models': {
        'logistic_regression': {
            'penalty': 'l2',
            'C': 1.0,
            'max_iter': 1000,
            'random_state': '${general.seed}',
        },
        'lasso': {
            'penalty': 'l1',
            'solver': 'liblinear',
            'C': 1.0,
            'max_iter': 1000,
            'random_state': '${general.seed}',
        },
        'ridge': {
            'penalty': 'l2',
            'C': 0.1,
            'max_iter': 1000,
            'random_state': '${general.seed}',
        },
        'elasticnet': {
            'penalty': 'elasticnet',
            'solver': 'saga',
            'l1_ratio': 0.5,
            'C': 100.0,
            'max_iter': 10000,
            'random_state': '${general.seed}',
        },

        'knn': {
            'n_neighbors': 5,
            'weights': 'uniform',
            'metric': 'minkowski',
        },

        'decision_tree': {
            'max_depth': '${hyperparams.max_depth}',
            'random_state': '${general.seed}',
        },
        'random_forest': {
            'n_estimators': '${hyperparams.n_estimators}',
            'max_depth': '${hyperparams.max_depth}',
            'random_state': '${general.seed}',
        },

        'xgb': {
            'n_estimators': '${hyperparams.n_estimators}',
            'max_depth': '${hyperparams.max_depth}',
            'learning_rate': '${hyperparams.learning_rate}',
            'subsample': '${hyperparams.subsample}',
            'colsample_bytree': '${hyperparams.colsample_bytree}',
            'verbosity': 0,
            'random_state': '${general.seed}',
            'use_label_encoder': False,
            'eval_metric': 'logloss',
        },
        'catboost': {
            'iterations': '${hyperparams.n_estimators}',
            'depth': '${hyperparams.max_depth}',
            'learning_rate': '${hyperparams.learning_rate}',
            'verbose': '${hyperparams.verbose}',
            'random_state': '${general.seed}',
        },
        'lightgbm': {
            'n_estimators': '${hyperparams.n_estimators}',
            'max_depth': '${hyperparams.max_depth}',
            'learning_rate': '${hyperparams.learning_rate}',
            'subsample': '${hyperparams.subsample}',
            'colsample_bytree': '${hyperparams.colsample_bytree}',
            'verbose': -1,
            'random_state': '${general.seed}',
        },

        'dnn': {
            'hidden_layers': [ 64, 32],
            'dropout': 0.4,
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': '${hyperparams.learning_rate}',
            'use_batchnorm': True,
            'optimizer': 'adam',
        },
        'ensemble_avg': {
            'models': ['xgb', 'catboost', 'lightgbm'],
            'weights': [1.0, 1.0, 1.0], 
        },
        
        'ensemble_voting': {
            'models': ['xgb', 'catboost', 'lightgbm'],
            'weights': [1.0, 1.0, 1.0],
        },
        
        'ensemble_stacking': {
            'base_models': ['xgb', 'catboost', 'lightgbm'],
            'meta_model': 'ridge',  
            'cv_folds': 5,
        },
    },
}

config = OmegaConf.create(config)