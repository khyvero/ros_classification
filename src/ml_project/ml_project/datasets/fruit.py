import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from ament_index_python.packages import get_package_share_directory

from ml_project.data_utils import load_dataset


def load_and_featurize():
    # Load fruit dataset and apply feature engineering with imputation
    df = load_dataset('fruits_weight_sphercity.csv')

    num_cols = ['Weight', 'Sphericity']
    cat_cols = ['Color']

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    pipe = ColumnTransformer([
        ('num', numeric_pipeline, num_cols),
        ('cat', categorical_pipeline, cat_cols),
    ], remainder='drop')

    X = pipe.fit_transform(df)
    y = df['labels'].values

    feat_names = num_cols + list(pipe.named_transformers_['cat']
                                 .get_feature_names_out(cat_cols))
    X_df = pd.DataFrame(X, columns=feat_names)
    y_sr = pd.Series(y, name='labels')

    return df, X_df, y_sr


def visualize(df: pd.DataFrame) -> str:
    # Output directory
    out_dir = os.path.join(
        get_package_share_directory('ml_project'),
        'data_processed', 'fruits'
    )
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter
    ax = axes[0, 0]
    for lbl, group in df.groupby('labels'):
        ax.scatter(group['Weight'], group['Sphericity'], label=lbl, alpha=0.7)
    ax.set_xlabel('Weight')
    ax.set_ylabel('Sphericity')
    ax.set_title('Weight vs Sphericity')
    ax.legend()

    # Weight histogram
    ax = axes[0, 1]
    df['Weight'].hist(ax=ax, bins=15)
    ax.set_title('Weight Distribution')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Count')

    # Sphericity histogram
    ax = axes[1, 0]
    df['Sphericity'].hist(ax=ax, bins=15)
    ax.set_title('Sphericity Distribution')
    ax.set_xlabel('Sphericity')
    ax.set_ylabel('Count')

    # Label counts
    ax = axes[1, 1]
    df['labels'].value_counts().plot.bar(ax=ax)
    ax.set_title('Label Counts')
    ax.set_xlabel('Fruit Label')
    ax.set_ylabel('Count')

    # Layout, show, save
    fig.tight_layout()
    plt.show()

    path = os.path.join(out_dir, 'fruit_overview.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def train(X_df: pd.DataFrame, y_sr: pd.Series):
    # Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_sr, test_size=0.2, random_state=42, stratify=y_sr
    )
    # train a Logistic Regression (linear classifier)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    return model, (X_test, y_test)


def evaluate(model, holdout):
    X_test, y_test = holdout
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    out_dir = os.path.join(
        get_package_share_directory('ml_project'),
        'data_processed', 'fruits'
    )
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    ax = axes[0, 0]
    cax = ax.matshow(cm)
    fig.colorbar(cax, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xticks(range(len(model.classes_)))
    ax.set_xticklabels(model.classes_)
    ax.set_yticks(range(len(model.classes_)))
    ax.set_yticklabels(model.classes_)

    # Classification report table
    rep_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    ax = axes[0, 1]
    ax.axis('off')
    tbl = ax.table(
        cellText=rep_df.round(2).values,
        colLabels=rep_df.columns,
        rowLabels=rep_df.index,
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    ax.set_title('Classification Report')

    # Actual vs Predicted counts
    labels = model.classes_
    counts_true = [sum(y_test == lbl) for lbl in labels]
    counts_pred = [sum(y_pred == lbl) for lbl in labels]
    ax = axes[1, 0]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], counts_true, width=0.4, label='Actual')
    ax.bar([i + 0.2 for i in x], counts_pred, width=0.4, label='Predicted')
    ax.set_title('Actual vs Predicted Counts')
    ax.legend()

    # Error distribution histogram (feature Weight)
    ax = axes[1, 1]
    df_err = pd.DataFrame({
        'Weight': X_test.iloc[:, 0],
        'Correct': (y_test == y_pred)
    })
    for corr, grp in df_err.groupby('Correct'):
        ax.hist(grp['Weight'], alpha=0.7, label=str(corr), bins=15)
    ax.set_title('Weight Dist: Correct vs Incorrect')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Count')
    ax.legend()

    fig.tight_layout()
    plt.show()

    eval_path = os.path.join(out_dir, 'fruit_evaluation.png')
    fig.savefig(eval_path, bbox_inches='tight')
    plt.close(fig)
    return {'accuracy': acc, 'report': report}, eval_path


def feature_importance(model, X_df: pd.DataFrame) -> str:
    # Get feature names and coefficients
    coefs = np.abs(model.coef_[0])
    feat_names = X_df.columns

    # Make output directory
    out_dir = os.path.join(
        get_package_share_directory('ml_project'),
        'data_processed', 'fruits'
    )
    os.makedirs(out_dir, exist_ok=True)

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feat_names, coefs)
    ax.set_title('Feature Importances')
    ax.set_xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.show()

    path = os.path.join(out_dir, 'fruit_feature_importance.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path