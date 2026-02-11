import os
import pandas as pd
import numpy as np
import yaml
from datetime import datetime


from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# Бустинги
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


from omegaconf import OmegaConf


from preprocessing import clean_data
try:
    from factory import TitanicNN, create_data_loader, train_nn, predict_probs
except ImportError:
    print(" Проверь наличие factory.py!")



def get_model(m_cfg, seed, input_dim=None, trial=None):
    m_type = m_cfg.get('type', 'lr').lower()
    
    # Линейные модели
    if m_type == "lr":
        return LogisticRegression(C=m_cfg.get('C', 1.0), solver='liblinear', random_state=seed)
    elif m_type == "lasso":
        return LogisticRegression(penalty='l1', C=m_cfg.get('C', 1.0), solver='liblinear', random_state=seed)
    elif m_type == "ridge":
        return RidgeClassifier(alpha=m_cfg.get('alpha', 1.0), random_state=seed)
    elif m_type == "elasticnet":
        return LogisticRegression(penalty='elasticnet', C=m_cfg.get('C', 1.0), 
                                  l1_ratio=m_cfg.get('l1_ratio', 0.5), solver='saga', random_state=seed)
    
    # KNN и Решающее дерево
    elif m_type == "knn":
        return KNeighborsClassifier(n_neighbors=m_cfg.get('n_neighbors', 5), 
                                    weights=m_cfg.get('weights', 'uniform'),
                                    metric=m_cfg.get('metric', 'minkowski'))
    elif m_type == "dt":
        return DecisionTreeClassifier(max_depth=m_cfg.get('max_depth', None), 
                                     criterion=m_cfg.get('criterion', 'gini'), random_state=seed)
    
    # Лес и Бустинги
    elif m_type == "rf":
        return RandomForestClassifier(n_estimators=m_cfg.get('n_estimators', 100), 
                                      max_depth=m_cfg.get('max_depth', None), random_state=seed)
    elif m_type == "xgb":
        return XGBClassifier(n_estimators=m_cfg.get('n_estimators', 100), random_state=seed)
    elif m_type == "lgb":
        return LGBMClassifier(n_estimators=m_cfg.get('n_estimators', 100), random_state=seed, verbose=-1)
    
    # Нейросеть
    elif m_type == "nn":
        return TitanicNN(input_dim=input_dim, cfg_nn=m_cfg, trial=trial)
    
    return LogisticRegression(random_state=seed)

#АНСАМБЛИ 

def get_Stacking_Ensemble(cfg, seed):
    base_models = [
        ("lgb", get_model(cfg.models.lgb, seed)),
        ("xgb", get_model(cfg.models.xgb, seed)),
        ("rf",  get_model(cfg.models.rf, seed))
    ]

    meta_C = cfg.models.get('stacking', {}).get('meta_C', 1.0)
    meta = LogisticRegression(C=meta_C, random_state=seed)
    return StackingClassifier(estimators=base_models, final_estimator=meta, cv=5)


def get_Voting_Ensemble(cfg, seed):
    estimators = [
        ("lgb", get_model(cfg.models.lgb, seed)),
        ("xgb", get_model(cfg.models.xgb, seed)),
        ("rf",  get_model(cfg.models.rf, seed))
    ]
   
    v_cfg = cfg.models.get('voting', {})
    weights = [v_cfg.get('w_lgb', 1), v_cfg.get('w_xgb', 1), v_cfg.get('w_rf', 1)]
    return VotingClassifier(estimators=estimators, voting='soft', weights=weights)



def log_results(name, results):
   
    print(f"\n" + "="*30)
    print(f" ИТОГИ ДЛЯ {name.upper()}:")
    for m, v in results.items():
        print(f"{m:15}: {v:.4f}")
    print("="*30)
    
   
    df_res = pd.DataFrame([{"Model": name, **results, "Date": datetime.now()}] )
    if os.path.exists("leaderboard.csv"):
        df_res.to_csv("leaderboard.csv", mode='a', header=False, index=False)
    else:
        df_res.to_csv("leaderboard.csv", index=False)



def main():
    cfg = OmegaConf.load("config/config.yaml")
    active_name = cfg.active_model
    seed = cfg.data.seed

    df = pd.read_csv(cfg.data.raw)
    df_cleaned = clean_data(df)
    X = df_cleaned.drop("Survived", axis=1, errors='ignore')
    y = df_cleaned["Survived"]

    scaler = StandardScaler().set_output(transform="pandas")
    X_scaled = scaler.fit_transform(X)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=seed)

 
    if active_name == "stacking":
        model = get_Stacking_Ensemble(cfg, seed)
    elif active_name == "voting":
        model = get_Voting_Ensemble(cfg, seed)
    else:
        model = get_model(cfg.models[active_name], seed, input_dim=X_scaled.shape[1])

  
    print(f" Запуск обучения модели: {active_name.upper()}")
    
    if active_name == "nn":
   
        X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
            X_train, y_train, test_size=0.1, random_state=seed
        )
        
        train_loader = create_data_loader(X_train_nn, y_train_nn, batch_size=cfg.models.nn.batch_size)
        val_loader = create_data_loader(X_val_nn, y_val_nn, batch_size=cfg.models.nn.batch_size)
        
      
        model = train_nn(
            model, 
            train_loader, 
            val_loader, 
            epochs=cfg.models.nn.epochs, 
            lr=cfg.models.nn.best_params.lr,
            patience=cfg.models.nn.early_stopping.patience,
            min_delta=cfg.models.nn.early_stopping.min_delta
        )
        
        y_prob = predict_probs(model, X_test)
        y_pred = (y_prob > 0.5).astype(int)
        
    else:
    
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

  
    results = {
        "ROC-AUC":   float(roc_auc_score(y_test, y_prob)),
        "Accuracy":  float(accuracy_score(y_test, y_pred)),
        "F1-Score":  float(f1_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred)),
        "Recall":    float(recall_score(y_test, y_pred))
    }

    log_results(active_name, results)

if __name__ == "__main__":
    main()