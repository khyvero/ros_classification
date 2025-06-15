import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from sklearn.inspection import permutation_importance
from ament_index_python.packages import get_package_share_directory

from ml_project.data_utils import load_dataset


def load_and_featurize():
    # Load Iris dataset, prepare features with imputation
    df = load_dataset('Iris.csv')
    target_col = 'Species' if 'Species' in df.columns else df.columns[-1]
    feature_cols = [
        c
        for c in df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    numeric_pipeline = Pipeline(
        [('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]
    )
    ct = ColumnTransformer([('num', numeric_pipeline, feature_cols)], remainder='drop')

    X = ct.fit_transform(df)
    X_df = pd.DataFrame(X, columns=feature_cols)
    y_sr = df[target_col].copy()
    return df, X_df, y_sr


def visualize(df: pd.DataFrame) -> str:
    target_col = 'Species' if 'Species' in df.columns else df.columns[-1]
    feature_cols = [
        c
        for c in df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Output directory
    out_dir = os.path.join(
        get_package_share_directory('ml_project'), 'data_processed', 'iris'
    )
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # scatter: first two features
    ax = axes[0, 0]
    for sp, grp in df.groupby(target_col):
        ax.scatter(grp[feature_cols[0]], grp[feature_cols[1]], label=sp, alpha=0.7)
    ax.set_title(f"{feature_cols[0]} vs {feature_cols[1]}")
    ax.legend()

    # scatter: last two features
    ax = axes[0, 1]
    for sp, grp in df.groupby(target_col):
        ax.scatter(grp[feature_cols[2]], grp[feature_cols[3]], label=sp, alpha=0.7)
    ax.set_title(f"{feature_cols[2]} vs {feature_cols[3]}")
    ax.legend()

    # histogram of first feature
    ax = axes[1, 0]
    df[feature_cols[0]].hist(ax=ax, bins=15)
    ax.set_title(f"{feature_cols[0]} Distribution")

    # bar chart of species counts
    ax = axes[1, 1]
    df[target_col].value_counts().plot.bar(ax=ax)
    ax.set_title("Species Counts")

    # Layout, show, save
    fig.tight_layout()
    plt.show()

    vis_path = os.path.join(out_dir, "iris_dataset_overview.png")
    fig.savefig(vis_path, bbox_inches="tight")
    plt.close(fig)
    return vis_path


def train(X_df: pd.DataFrame, y_sr: pd.Series, model_name: str):
    # Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_sr, test_size=0.2, random_state=42, stratify=y_sr
    )
    # train
    if model_name == "knn":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name in ("forest", "random_forest"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model '{model_name}'")
    model.fit(X_train, y_train)
    return model, (X_test, y_test)


def evaluate(model, holdout):
    X_test, y_test = holdout
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    out_dir = os.path.join(
        get_package_share_directory("ml_project"), "data_processed", "iris"
    )
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    ax = axes[0, 0]
    cax = ax.matshow(cm)
    fig.colorbar(cax, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(model.classes_)))
    ax.set_xticklabels(model.classes_)
    ax.set_yticks(range(len(model.classes_)))
    ax.set_yticklabels(model.classes_)

    # classification report table
    rep_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    ax = axes[0, 1]
    ax.axis("off")
    tbl = ax.table(
        cellText=rep_df.round(2).values,
        colLabels=rep_df.columns,
        rowLabels=rep_df.index,
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    ax.set_title("Classification Report")

    # actual vs predicted counts
    labels = model.classes_
    counts_true = [sum(y_test == lbl) for lbl in labels]
    counts_pred = [sum(y_pred == lbl) for lbl in labels]
    ax = axes[1, 0]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], counts_true, width=0.4, label="Actual")
    ax.bar([i + 0.2 for i in x], counts_pred, width=0.4, label="Predicted")
    ax.set_title("Actual vs Predicted Counts")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # error histogram
    ax = axes[1, 1]
    errors = pd.Series(y_test != y_pred).map({False: "Correct", True: "Incorrect"})
    errors.value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Correct vs Incorrect Predictions")

    fig.tight_layout()
    plt.show()

    eval_path = os.path.join(out_dir, f"evaluation_{model.__class__.__name__}.png")
    fig.savefig(eval_path, bbox_inches="tight")
    plt.close(fig)
    return {"accuracy": acc, "report": report}, eval_path


def feature_importance(model, X_df: pd.DataFrame, y_sr: pd.Series) -> str:
    # determine importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
    else:
        # permutation importance
        result = permutation_importance(
            model, X_df, y_sr, n_repeats=10, random_state=42
        )
        importances = result.importances_mean

    feat_names = X_df.columns

    # Make output directory
    out_dir = os.path.join(
        get_package_share_directory("ml_project"), "data_processed", "iris"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feat_names, importances)
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.show()

    path = os.path.join(out_dir, f"iris_feature_importance_{model.__class__.__name__}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
