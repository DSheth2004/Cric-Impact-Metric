"""
Module 4: ML Win Probability Model
Predict P(batting side wins) at any ball. v2: monotonic gradient boosting, scored with
5-fold cross-fitting (cross_fit_win_prob). train_model() is the original v1 routine.
"""

import os
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


# v2.1: 'over' and 'phase_encoded' removed (balls bowled already carries that, and they only
# added steps); raw run rate replaced by a par-smoothed run rate so early dot balls don't
# cause cliffs. Same accuracy as v2 (AUC 0.787), half the early-innings noise.
FEATURES = [
    'innings', 'cumulative_runs', 'cumulative_wickets',
    'cumulative_balls', 'run_rate_smooth', 'required_rr', 'rr_gap_smooth',
    'wickets_remaining', 'target', 'venue_difficulty',
    'is_chasing',
]

# Inputs of the original v1 Random Forest (kept for validate_model.py comparisons)
V1_FEATURES = [
    'innings', 'over', 'cumulative_runs', 'cumulative_wickets',
    'cumulative_balls', 'current_rr', 'required_rr', 'rr_pressure_gap',
    'wickets_remaining', 'target', 'venue_difficulty', 'phase_encoded',
    'is_chasing',
]

MODEL_DIR = os.path.join('cricket_impact', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'win_prob_model.pkl')


def prepare_training_data(ball_df, info_df, features=None):
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

    features = FEATURES if features is None else features
    # Ensure features exist and fill NaN
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    # Remove rows with all-zero features (edge cases)
    X = df[features].values
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


# ── v2: monotonic gradient boosting + cross-fitting ──
# Direction of each feature's effect on P(batting side wins), holding the others fixed.
# +1 = can only raise it, -1 = can only lower it, 0 = unconstrained.
MONOTONIC = {
    'cumulative_runs': 1, 'run_rate_smooth': 1, 'wickets_remaining': 1,
    'cumulative_wickets': -1, 'required_rr': -1, 'rr_gap_smooth': -1, 'target': -1,
    'cumulative_balls': -1,
}


def make_model():
    """Gradient boosting that respects cricket logic: a wicket never helps the batting side."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    # Heavy regularisation: deliveries within a match share one outcome, so there are only
    # ~1,000 independent labels. Settings chosen by 5-fold grouped cross-validation.
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.03, max_depth=3, min_samples_leaf=5000,
        l2_regularization=5.0, monotonic_cst=[MONOTONIC.get(f, 0) for f in FEATURES],
        random_state=42,
    )


def cross_fit_win_prob(ball_df, info_df, n_splits=5, model_dir=None):
    """
    Score every delivery with a model that never saw that match's result.

    Matches are split into n_splits groups; each group is predicted by a model trained
    on the others. A final model trained on all matches is saved for scoring new uploads
    and for deliveries without a result (no-result games, super overs).
    Returns (ball_df with win_prob, metrics dict).
    """
    from sklearn.model_selection import GroupKFold

    if model_dir is None:
        model_dir = MODEL_DIR
    df = ball_df.copy()
    df['_rid'] = np.arange(len(df))
    X, y, mids, lab = prepare_training_data(df, info_df)
    rid = lab['_rid'].values

    oof = np.full(len(X), np.nan)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups=mids), 1):
        m = make_model().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        print(f"    fold {k}/{n_splits}: {len(np.unique(mids[te]))} matches scored out-of-fold")

    metrics = {
        'model': 'HistGradientBoostingClassifier (monotonic)',
        'scoring': f'{n_splits}-fold cross-fitting by match',
        'oof_auc': round(float(roc_auc_score(y, oof)), 4),
        'oof_brier': round(float(brier_score_loss(y, oof)), 4),
        'oof_log_loss': round(float(log_loss(y, oof)), 4),
    }
    print(f"  Out-of-fold AUC {metrics['oof_auc']}, Brier {metrics['oof_brier']}")

    final = make_model().fit(X, y)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(final, os.path.join(model_dir, 'win_prob_model.pkl'))
    print(f"  Final model (all matches) saved to {os.path.join(model_dir, 'win_prob_model.pkl')}")

    # final model for anything without a label; out-of-fold where we have one
    out = predict_win_prob(df, final)
    out.loc[out['_rid'].isin(rid), 'win_prob'] = pd.Series(oof, index=rid).reindex(
        out.loc[out['_rid'].isin(rid), '_rid']).values
    out = out.drop(columns=['_rid'])
    return out, metrics


def load_model(model_dir=None):
    """Load a previously saved model."""
    if model_dir is None:
        model_dir = MODEL_DIR
    model_path = os.path.join(model_dir, 'win_prob_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def predict_win_prob(ball_df, model, features=None):
    """Add win_prob predictions to ball_df."""
    df = ball_df.copy()
    features = FEATURES if features is None else features
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    X = df[features].values
    probs = model.predict_proba(X)[:, 1]
    df['win_prob'] = probs

    return df
