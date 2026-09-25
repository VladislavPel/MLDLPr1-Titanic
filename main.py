import pandas as pd
from omegaconf import OmegaConf
from config import config
from src.data import load_data, prepare_data
from src.models import create_model, train_dnn, predict, evaluate_cv, create_voting_predictions, create_ensemble_predictions, create_stacking_predictions
from src.utils import save_submission, save_metrics
import numpy as np


def main():
    model_name = config.general.active
    exp_id = config.general.experiment_id

    print(f"ExperimentID: {exp_id} | Model: {model_name}")

    train, test = load_data(config)
    X_train, y_train, X_test, test_ids = prepare_data(train, test, config)
    
    # === ДИАГНОСТИКА ===
    print(f"\n{'='*50}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Проверка колонок
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)

    if train_cols == test_cols:
        print(f"✅ Columns match: {len(train_cols)} features")
    else:
        print(f"❌ Columns MISMATCH!")
        print(f"   Только в train: {train_cols - test_cols}")
        print(f"   Только в test: {test_cols - train_cols}")

    # Проверка NaN
    print(f"\nNaN в X_train: {X_train.isnull().sum().sum()}")
    print(f"NaN в X_test: {X_test.isnull().sum().sum()}")

    if X_train.isnull().sum().sum() > 0:
        print(f"   Колонки с NaN в train: {X_train.columns[X_train.isnull().any()].tolist()}")
    if X_test.isnull().sum().sum() > 0:
        print(f"   Колонки с NaN в test: {X_test.columns[X_test.isnull().any()].tolist()}")

    print(f"{'='*50}\n")
    # === КОНЕЦ ДИАГНОСТИКИ ===


    if model_name == "dnn":
        result = train_dnn(X_train, y_train, config)
        model = result["model"]
        cv_mean = result["best_val_acc"]
        cv_std = 0.0
        is_nn = True
        trained_models = [{"model": model, "is_nn": True}]

    elif model_name in ["ensemble_avg", "ensemble_voting", "ensemble_stacking"]:
        if model_name == "ensemble_stacking":
            base_model_names = config.models.ensemble_stacking.base_models
            meta_model_name = config.models.ensemble_stacking.meta_model
            cv_folds = config.models.ensemble_stacking.cv_folds
            
            print(f" Stacking : {base_model_names}")
            print(f" Meta-model: {meta_model_name}")
            
            base_factories = []
            for m_name in base_model_names:
                base_factories.append(lambda name=m_name: create_model(name, config))
            
            meta_model = create_model(meta_model_name, config)

            test_preds = create_stacking_predictions(
                X_train, y_train, X_test, base_factories, meta_model, cv=cv_folds
            )
            
            cv_mean = 0.0
            cv_std = 0.0
            for m_name in base_model_names:
                m = create_model(m_name, config)
                cv_m, cv_s = evaluate_cv(m, X_train, y_train, config)
                cv_mean += cv_m
            cv_mean /= len(base_model_names)
            
            model = None
            is_nn = False
        
        else:
            ensemble_models = config.models[model_name].models
            ensemble_weights = list(config.models[model_name].weights)
            
            print(f"{model_name} with models: {ensemble_models}")
            
            trained_models = []
            cv_scores = []
            
            for m_name in ensemble_models:
                print(f"\n   --- Training {m_name} ---", flush = True)
                if m_name == "dnn":
                    result = train_dnn(X_train, y_train, config)
                    trained_models.append({"model": result["model"], "is_nn": True})
                    cv_scores.append(result["best_val_acc"])
                else:
                    m = create_model(m_name, config)
                    cv_m, cv_s = evaluate_cv(m, X_train, y_train, config)
                    m.fit(X_train, y_train)
                    trained_models.append({"model": m, "is_nn": False})
                    cv_scores.append(cv_m)
                    print(f"   CV: {cv_m:.4f} +/- {cv_s:.4f}")
            
            if model_name == "ensemble_avg":
                test_preds = create_ensemble_predictions(X_test, trained_models, ensemble_weights)
            elif model_name == "ensemble_voting":
                test_preds = create_voting_predictions(X_test, trained_models, ensemble_weights)
            
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            model = None
            is_nn = False

    else:
        model = create_model(model_name, config)
        cv_mean, cv_std = evaluate_cv(model, X_train, y_train, config)
        model.fit(X_train, y_train)
        is_nn = False
        trained_models = [{"model": model, "is_nn": False}]

    print(f"\nCV Accuracy: {cv_mean:.4f} +/- {cv_std:.4f}")

    if model_name not in ["ensemble_avg", "ensemble_voting", "ensemble_stacking"]:
        test_preds = predict(model, X_test, is_nn)

    submission_path = f"results/exp{exp_id}_{model_name}_submission.csv"
    metrics_path = f"results/exp{exp_id}_{model_name}_metrics.json"

    save_submission(test_preds, test_ids, submission_path)
    save_metrics({
        'experiment_id': exp_id, 
        'model_name': model_name, 
        'cv_mean': cv_mean, 
        'cv_std': cv_std, 
        'params': OmegaConf.to_container(config.models[model_name], resolve=True)
    }, metrics_path)


    print(f"{'Exp':<6} {'Model':<20} {'CV Mean':<12} {'CV Std':<12}")
    print(f"{'-'*50}")
    print(f"{exp_id:<6} {model_name:<20} {cv_mean:<12.4f} {cv_std:<12.4f}")


if __name__ == "__main__":
    main()
