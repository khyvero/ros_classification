import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from ament_index_python.packages import get_package_share_directory

from ml_project.data_utils import load_dataset


def load_and_featurize_cancer():
    # load breast_cancer dataset
    df = load_dataset('Breast_Cancer.csv')
    # drop identifier and target from features
    drop = [c for c in ('id','ID','Id','diagnosis') if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop]

    # imputation
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    ct = ColumnTransformer([('num', num_pipe, feature_cols)], remainder='drop')

    X = ct.fit_transform(df)
    X_df = pd.DataFrame(X, columns=feature_cols)
    y_sr = df['diagnosis'].copy()
    return df, X_df, y_sr


def load_and_featurize_penguin():
    # Load Penguin dataset, preprocessing
    df = load_dataset('Penguin.csv')
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    target_col = 'species' if 'species' in obj_cols else obj_cols[-1]
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c]) and 'id' not in c.lower()]
    cat_cols = [c for c in obj_cols if c != target_col]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    ct = ColumnTransformer([
        ('num', numeric_pipeline, num_cols),
        ('cat', categorical_pipeline, cat_cols),
    ], remainder='drop')

    X = ct.fit_transform(df)
    feat_names = num_cols + list(ct.named_transformers_['cat'].get_feature_names_out(cat_cols))
    X_df = pd.DataFrame(X, columns=feat_names)
    y_sr = df[target_col].copy()
    return df, X_df, y_sr


def visualize_both(df_cancer: pd.DataFrame, df_penguin: pd.DataFrame) -> str:
    # Output directory
    out_dir = os.path.join(
        get_package_share_directory('ml_project'),
        'data_processed', 'comparison'
    )
    os.makedirs(out_dir, exist_ok=True)

    # pick two numeric features from each
    cov = [c for c in df_cancer.columns if is_numeric_dtype(df_cancer[c]) and c.lower() not in ('id','diagnosis')][:2]
    pov = [c for c in df_penguin.columns if is_numeric_dtype(df_penguin[c])][:2]
    target_c = 'diagnosis'
    target_p = 'species' if 'species' in df_penguin.columns else df_penguin.select_dtypes(include=['object']).columns[-1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Cancer: scatter
    ax = axes[0, 0]
    for lab, grp in df_cancer.groupby(target_c):
        ax.scatter(grp[cov[0]], grp[cov[1]], label=lab, alpha=0.7)
    ax.set_title(f'Cancer: {cov[0]} vs {cov[1]}')
    ax.legend()

    # Cancer: histogram
    ax = axes[0, 1]
    df_cancer[cov[0]].hist(ax=ax, bins=20)
    ax.set_title(f'{cov[0]} Distribution (Cancer)')

    # Cancer: bar counts
    ax = axes[0, 2]
    df_cancer[target_c].value_counts().plot.bar(ax=ax)
    ax.set_title('Cancer Diagnosis Counts')

    # Penguin: scatter
    ax = axes[1, 0]
    for lab, grp in df_penguin.groupby(target_p):
        ax.scatter(grp[pov[0]], grp[pov[1]], label=lab, alpha=0.7)
    ax.set_title(f'Penguin: {pov[0]} vs {pov[1]}')
    ax.legend()

    # Penguin: histogram
    ax = axes[1, 1]
    df_penguin[pov[0]].hist(ax=ax, bins=20)
    ax.set_title(f'{pov[0]} Distribution (Penguin)')

    # Penguin: bar counts
    ax = axes[1, 2]
    df_penguin[target_p].value_counts().plot.bar(ax=ax)
    ax.set_title('Penguin Species Counts')

    # Layout, show, save
    fig.tight_layout()
    plt.show()

    path = os.path.join(out_dir, 'overview_cancer_penguin.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def train_cancer(X_df, y_sr):
    # Split Cancer data (80% train, 20% test)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_df, y_sr, test_size=0.2, random_state=42, stratify=y_sr
    )
    # Train Linear SVM on Cancer data
    model = LinearSVC(random_state=42, max_iter=10000)
    model.fit(X_tr, y_tr)
    return model, (X_te, y_te)


def train_penguin(X_df, y_sr):
    # Split Penguin data (80% train, 20% test)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_df, y_sr, test_size=0.2, random_state=42, stratify=y_sr
    )
    # Train Linear SVM on Penguin data
    model = LinearSVC(random_state=42, max_iter=10000)
    model.fit(X_tr, y_tr)
    return model, (X_te, y_te)


def evaluate_both(model_c, holdout_c, model_p, holdout_p) -> (dict, str):
    # Evaluate both SVMs, plotting 6 subplots: CM, report, counts for each
    Xc_te, yc_te = holdout_c
    Xp_te, yp_te = holdout_p
    yc_pr = model_c.predict(Xc_te)
    yp_pr = model_p.predict(Xp_te)

    acc_c = accuracy_score(yc_te, yc_pr)
    acc_p = accuracy_score(yp_te, yp_pr)

    out_dir = os.path.join(
        get_package_share_directory('ml_project'),
        'data_processed', 'comparison'
    )
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Cancer: confusion matrix
    cm_c = confusion_matrix(yc_te, yc_pr)
    ax = axes[0, 0]
    cax = ax.matshow(cm_c)
    fig.colorbar(cax, ax=ax)
    ax.set_title(f'Cancer CM (acc={acc_c:.2f})')

    # Cancer: classification report
    rep_df_c = pd.DataFrame(classification_report(yc_te, yc_pr, output_dict=True)).transpose()
    ax = axes[0, 1]
    ax.axis('off')
    tbl = ax.table(
        cellText=rep_df_c.round(2).values,
        colLabels=rep_df_c.columns,
        rowLabels=rep_df_c.index,
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    ax.set_title('Cancer Report')

    # Cancer: counts
    labels_c = model_c.classes_
    counts_true_c = [sum(yc_te == lbl) for lbl in labels_c]
    counts_pred_c = [sum(yc_pr == lbl) for lbl in labels_c]
    ax = axes[0, 2]
    x_c = range(len(labels_c))
    ax.bar([i - 0.2 for i in x_c], counts_true_c, width=0.4, label='Actual')
    ax.bar([i + 0.2 for i in x_c], counts_pred_c, width=0.4, label='Predicted')
    ax.set_title('Cancer Counts')
    ax.legend()

    # Penguin: confusion matrix
    cm_p = confusion_matrix(yp_te, yp_pr)
    ax = axes[1, 0]
    cax = ax.matshow(cm_p)
    fig.colorbar(cax, ax=ax)
    ax.set_title(f'Penguin CM (acc={acc_p:.2f})')

    # Penguin: classification report
    rep_df_p = pd.DataFrame(classification_report(yp_te, yp_pr, output_dict=True)).transpose()
    ax = axes[1, 1]
    ax.axis('off')
    tbl = ax.table(
        cellText = rep_df_p.round(2).values,
        colLabels = rep_df_p.columns,
        rowLabels = rep_df_p.index,
        loc = 'center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    ax.set_title('Penguin Report')

    # Penguin: counts
    labels_p = model_p.classes_
    counts_true_p = [sum(yp_te == lbl) for lbl in labels_p]
    counts_pred_p = [sum(yp_pr == lbl) for lbl in labels_p]
    ax = axes[1, 2]
    x_p = range(len(labels_p))
    ax.bar([i - 0.2 for i in x_p], counts_true_p, width=0.4, label='Actual')
    ax.bar([i + 0.2 for i in x_p], counts_pred_p, width=0.4, label='Predicted')
    ax.set_title('Penguin Counts')
    ax.legend()

    fig.tight_layout()
    plt.show()

    eval_path = os.path.join(out_dir, 'evaluation_cancer_penguin.png')
    fig.savefig(eval_path, bbox_inches='tight')
    plt.close(fig)

    return {'cancer_acc': acc_c, 'penguin_acc': acc_p}, eval_path


def feature_importance_both(model_c, X_c_df: pd.DataFrame,
                            model_p, X_p_df: pd.DataFrame) -> str:
    # Get feature names and coefficients
    coefs_c = np.abs(model_c.coef_[0])
    names_c = X_c_df.columns
    coefs_p = np.abs(model_p.coef_[0])
    names_p = X_p_df.columns

    # Make output directory
    out_dir = os.path.join(
        get_package_share_directory('ml_project'),
        'data_processed', 'comparison'
    )
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot cancer feature importances
    ax = axes[0]
    ax.barh(names_c, coefs_c)
    ax.set_title('Cancer Feature Importances')
    ax.set_xlabel('Absolute Coefficient')

    # Plot penguin feature importances
    ax = axes[1]
    ax.barh(names_p, coefs_p)
    ax.set_title('Penguin Feature Importances')
    ax.set_xlabel('Absolute Coefficient')

    # Layout, show, save
    fig.tight_layout()
    plt.show()

    path = os.path.join(out_dir, 'feature_importance_cancer_penguin.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path
