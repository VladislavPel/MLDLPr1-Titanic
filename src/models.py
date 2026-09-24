import pandas as pd
from config import config
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from omegaconf import OmegaConf

def get_baseline_model(test_df: pd.DataFrame) -> pd.DataFrame:
    ''' Правило женщина - выжила '''
    predictions = (test_df['Sex'] == 'female').astype(int)
    return pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
    
class TitanicDNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout, use_batchnorm):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def create_model(name:str, config):
    params = OmegaConf.to_container(config.models[name], resolve=True)
    
    factories = {'logistic_regression': lambda: LogisticRegression(**params),
                 'lasso': lambda: LogisticRegression(**params),
                 'ridge': lambda: LogisticRegression(**params),
                 'elasticnet': lambda: LogisticRegression(**params),
                 'knn': lambda: KNeighborsClassifier(**params),
                 'decision_tree': lambda: DecisionTreeClassifier(**params),
                 'random_forest': lambda: RandomForestClassifier(**params),
                 'gradient_boosting': lambda: GradientBoostingClassifier(**params),
                 'xgb': lambda: XGBClassifier(**params),
                 'lightgbm': lambda: LGBMClassifier(**params),
                 'catboost': lambda: CatBoostClassifier(**params),
                 }

    return factories[name]()


def train_dnn(X_train, y_train, config):

    params = OmegaConf.to_container(config.models.dnn, resolve=True)
    seed = config.general.seed
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    from sklearn.model_selection import train_test_split
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=seed,
        stratify=y_train, 
    )
    
    X_tr_t = torch.FloatTensor(X_tr.values if isinstance(X_tr, pd.DataFrame) else X_tr)
    y_tr_t = torch.FloatTensor(y_tr.values if isinstance(y_tr, pd.Series) else y_tr)
    X_val_t = torch.FloatTensor(X_val.values if isinstance(X_val, pd.DataFrame) else X_val)
    y_val_t = torch.FloatTensor(y_val.values if isinstance(y_val, pd.Series) else y_val)
    
    train_dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=params["batch_size"], shuffle=False
    )
    
    model = TitanicDNN(
        input_dim=X_tr_t.shape[1],
        hidden_layers=params["hidden_layers"],
        dropout=params["dropout"],
        use_batchnorm=params["use_batchnorm"],
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"], eta_min=1e-8)
    
    patience = 15
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    best_state = None 
    
  
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    log_every = 10  
    
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == yb).sum().item()
        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)
        
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        if epoch % log_every == 0 or epoch == 1:
            print(f"   Epoch {epoch:3d}/{params['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch} "
                      f"(best: epoch {best_epoch}, val_acc: {best_val_acc:.4f})")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"Best epoch: {best_epoch} | Best Val Acc: {best_val_acc:.4f}")
    
    return {
        "model": model,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "history": history,
    }


def evaluate_cv(model, X, y, cfg):
    ''' Кросвал '''
    cv = StratifiedKFold(
        n_splits=cfg.training.cv_folds, shuffle=True, random_state=cfg.general.seed
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()

def predict(model, X_test, is_nn=False):
    if is_nn:
        X_t = torch.FloatTensor(X_test.values if isinstance(X_test, pd.DataFrame) else X_test)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_t)).numpy()
        return (probs >= 0.5).astype(int)
    return model.predict(X_test)

def predict_proba(model, X_test, is_nn=False):
    """
    Возвращает вероятности класса 1.
    """
    if is_nn:
        X_t = torch.FloatTensor(X_test.values if isinstance(X_test, pd.DataFrame) else X_test)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_t)).numpy()
        return probs
    return model.predict_proba(X_test)[:, 1]


def create_ensemble_predictions(X_test, trained_models_info, weights=None):
    """
    Усредняет вероятности от нескольких обученных моделей.
    
    """
    if weights is None:
        weights = [1.0] * len(trained_models_info)
    
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    avg_proba = np.zeros(X_test.shape[0])
    for info, w in zip(trained_models_info, weights):
        proba = predict_proba(info['model'], X_test, is_nn=info['is_nn'])
        avg_proba += w * proba
    
    return (avg_proba >= 0.5).astype(int)


def create_voting_predictions(X_test, trained_models_info, weights=None):
    """
    Hard Voting: каждая модель голосует за класс, выбирается большинство.
    
    """
    if weights is None:
        weights = [1.0] * len(trained_models_info)
    
    votes = np.zeros(X_test.shape[0])
    for info, w in zip(trained_models_info, weights):
        pred = predict(info['model'], X_test, is_nn=info['is_nn'])
        votes += w * pred
    
    threshold = sum(weights) / 2
    return (votes >= threshold).astype(int)


def create_stacking_predictions(X_train, y_train, X_test, base_model_factories, meta_model, cv=5):
    """
    Stacking
    
    """
    from sklearn.model_selection import StratifiedKFold
    
    n_samples = X_train.shape[0]
    n_base_models = len(base_model_factories)
    
    oof_preds = np.zeros((n_samples, n_base_models))
    test_preds = np.zeros((X_test.shape[0], n_base_models))
    
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    print(f"   Training {n_base_models} base models with {cv}-fold CV for stacking...")
    
    for i, factory in enumerate(base_model_factories):
        test_fold_preds = np.zeros(X_test.shape[0])
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
  
            model = factory()
            model.fit(X_tr, y_tr)

            oof_preds[val_idx, i] = predict_proba(model, X_val)

            test_fold_preds += predict_proba(model, X_test)
        
        test_preds[:, i] = test_fold_preds / cv
        print(f"   Base model {i+1}/{n_base_models} done")
    
    print(f"   Training meta-model: {type(meta_model).__name__}")
    meta_model.fit(oof_preds, y_train)
    
    return meta_model.predict(test_preds)