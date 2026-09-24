import pandas as pd
from omegaconf import OmegaConf
from config import config
from src.data import load_data, prepare_data
from src.models import create_model, train_dnn, predict, evaluate_cv
from src.utils import save_submission, save_metrics



def main():
    model_name = config.general.active
    exp_id = config.general.experiment_id

    print(f"ExperimentID: {exp_id} | Model: {model_name}")

    train, test = load_data(config)
    X_train, y_train, X_test, test_ids = prepare_data(train, test, config)
    

    if model_name == "dnn":
        results = train_dnn(X_train, y_train, config)
        model = results["model"]
        cv_mean = results["best_val_acc"]
        cv_std = 0.0
        is_nn = True
    else:
        model = create_model(model_name, config)
        cv_mean, cv_std = evaluate_cv(model, X_train, y_train, config)
        model.fit(X_train, y_train)
        is_nn = False

    test_pred = predict(model, X_test, is_nn)

    submission_path = f"results/exp{exp_id}_{model_name}_submission.csv"
    metrics_path = f"results/exp{exp_id}_{model_name}_metrics.json"

    save_submission(test_pred, test_ids, submission_path)
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
