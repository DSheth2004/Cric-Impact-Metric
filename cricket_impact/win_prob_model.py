"""
Module 4: ML Win Probability Model
Train XGBoost model to predict P(batting_team_wins) at any ball.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


FEATURES = [
    'innings', 'over', 'cumulative_runs', 'cumulative_wickets',
    'cumulative_balls', 'current_rr', 'required_rr', 'rr_pressure_gap',
    'wickets_remaining', 'target', 'venue_difficulty', 'phase_encoded',
    'is_chasing',
]

MODEL_DIR = os.path.join('cricket_impact', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'win_prob_model.pkl')


def prepare_training_data(ball_df, info_df):
    """Prepare features and labels for win probability training."""
    # Merge winner info
    df = ball_df.merge(info_df[['match_id', 'winner', 'no_result']], on='match_id', how='left')

    # Exclude no-result matches
    df = df[df['no_result'] != True].copy()
    df = df[df['winner'].notna() & (df['winner'] != '')].copy()

    # Only use innings 1 and 2
    df = df[df['innings'].isin([1, 2])].copy()

    # Label: 1 if batting_team won
    df['batting_team_won'] = (df['batting_team'] == df['winner']).astype(int)

    # Ensure features exist and fill NaN
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    # Remove rows with all-zero features (edge cases)
    X = df[FEATURES].values
    y = df['batting_team_won'].values
    match_ids = df['match_id'].values

    return X, y, match_ids, df


def train_model(ball_df, info_df, model_dir=None):
    """Train win probability models and select best."""
    if model_dir is None:
        model_dir = MODEL_DIR

    print("  Preparing training data...")
    X, y, match_ids, df = prepare_training_data(ball_df, info_df)
    print(f"  Training data: {len(X):,} samples, positive rate: {y.mean():.3f}")

    # Split by match_id (80/20)
    unique_matches = np.unique(match_ids)
    train_matches, test_matches = train_test_split(
        unique_matches, test_size=0.2, random_state=42
    )
    train_mask = np.isin(match_ids, train_matches)
    test_mask = np.isin(match_ids, test_matches)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")

    results = {}

    # 1. Logistic Regression (baseline)
    print("  Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    results['logistic'] = {
        'model': lr,
        'auc': roc_auc_score(y_test, lr_prob),
        'log_loss': log_loss(y_test, lr_prob),
        'brier': brier_score_loss(y_test, lr_prob),
    }
    print(f"    AUC: {results['logistic']['auc']:.4f}, Brier: {results['logistic']['brier']:.4f}")

    # 2. Random Forest
    print("  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    results['rf'] = {
        'model': rf,
        'auc': roc_auc_score(y_test, rf_prob),
        'log_loss': log_loss(y_test, rf_prob),
        'brier': brier_score_loss(y_test, rf_prob),
    }
    print(f"    AUC: {results['rf']['auc']:.4f}, Brier: {results['rf']['brier']:.4f}")

    # 3. XGBoost (primary)
    if HAS_XGB:
        print("  Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        results['xgboost'] = {
            'model': xgb_model,
            'auc': roc_auc_score(y_test, xgb_prob),
            'log_loss': log_loss(y_test, xgb_prob),
            'brier': brier_score_loss(y_test, xgb_prob),
        }
        print(f"    AUC: {results['xgboost']['auc']:.4f}, Brier: {results['xgboost']['brier']:.4f}")

    # Select best model by AUC
    best_name = max(results, key=lambda k: results[k]['auc'])
    best_model = results[best_name]['model']
    best_metrics = results[best_name]

    print(f"\n  Best model: {best_name}")
    print(f"    AUC-ROC:  {best_metrics['auc']:.4f}")
    print(f"    Log Loss: {best_metrics['log_loss']:.4f}")
    print(f"    Brier:    {best_metrics['brier']:.4f}")

    # Calibrate if Brier score is too high
    if best_metrics['brier'] > 0.22:
        print("  Calibrating model...")
        calibrated = CalibratedClassifierCV(best_model, cv=3, method='isotonic')
        calibrated.fit(X_train, y_train)
        cal_prob = calibrated.predict_proba(X_test)[:, 1]
        cal_brier = brier_score_loss(y_test, cal_prob)
        cal_auc = roc_auc_score(y_test, cal_prob)
        print(f"    Calibrated AUC: {cal_auc:.4f}, Brier: {cal_brier:.4f}")
        if cal_brier < best_metrics['brier']:
            best_model = calibrated
            best_metrics['auc'] = cal_auc
            best_metrics['brier'] = cal_brier
            print("    Using calibrated model.")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'win_prob_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"  Model saved to {model_path}")

    return best_model, best_metrics


def load_model(model_dir=None):
    """Load a previously saved model."""
    if model_dir is None:
        model_dir = MODEL_DIR
    model_path = os.path.join(model_dir, 'win_prob_model.pkl')
    if os.path.exists(model_path):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            return joblib.load(model_path)
    return None


def predict_win_prob(ball_df, model):
    """Add win_prob predictions to ball_df."""
    df = ball_df.copy()
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X = df[FEATURES].values
    probs = model.predict_proba(X)[:, 1]
    df['win_prob'] = probs

    return df
