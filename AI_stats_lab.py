import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)


# -------------------------------------------------
# Question 1: Dataset generation and visualization
# -------------------------------------------------

def generate_clean_data(n_samples=500, noise=20, random_state=42):
    X, y, coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=noise,
        coef=True,
        random_state=random_state
    )
    return X, y, coef


def add_outliers(X, y, n_outliers=25, random_state=42):
    rng = np.random.RandomState(random_state)

    X_out = X.copy()
    y_out = y.copy()

    X_out[:n_outliers] = 10 + 0.75 * rng.randn(n_outliers, 1)
    y_out[:n_outliers] = -15 + 20 * rng.randn(n_outliers)

    return X_out, y_out


def plot_dataset_with_outliers(X, y, n_outliers=25):
    fig, ax = plt.subplots()

    ax.scatter(X[n_outliers:], y[n_outliers:], label="Normal Data")
    ax.scatter(X[:n_outliers], y[:n_outliers], label="Outliers")

    ax.set_title("Dataset with Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


# -------------------------------------------------
# Question 2: Fit regression models
# -------------------------------------------------

def fit_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return float(model.coef_[0])


def fit_huber_regression(X, y):
    model = HuberRegressor()
    model.fit(X, y)
    return float(model.coef_[0])


def fit_ransac_regression(X, y, random_state=42):
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)
    return float(model.estimator_.coef_[0])


def fit_theilsen_regression(X, y, random_state=42):
    model = TheilSenRegressor(random_state=random_state)
    model.fit(X, y)
    return float(model.coef_[0])


def coefficient_errors(coef_dict, true_coef):
    errors = {}
    for key, value in coef_dict.items():
        errors[key] = abs(value - true_coef)
    return errors


def best_robust_model(errors):
    robust_models = {
        k: v for k, v in errors.items()
        if k != "linear_regression"
    }
    return min(robust_models, key=robust_models.get)


def ransac_outlier_summary(X, y, n_outliers=25, random_state=42):
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    total_outliers_detected = np.sum(outlier_mask)

    added_outliers_detected = np.sum(outlier_mask[:n_outliers])

    return total_outliers_detected, added_outliers_detected


# -------------------------------------------------
# Visualization functions
# -------------------------------------------------

def plot_regression_fits(X, y, random_state=42):
    fig, ax = plt.subplots()

    ax.scatter(X, y, label="Data")

    line_X = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

    models = {
        "Linear": LinearRegression(),
        "Huber": HuberRegressor(),
        "RANSAC": RANSACRegressor(random_state=random_state),
        "Theil-Sen": TheilSenRegressor(random_state=random_state)
    }

    for name, model in models.items():
        model.fit(X, y)

        if name == "RANSAC":
            coef = model.estimator_.coef_[0]
            intercept = model.estimator_.intercept_
        else:
            coef = model.coef_[0]
            intercept = model.intercept_

        line_y = coef * line_X + intercept
        ax.plot(line_X, line_y, label=name)

    ax.set_title("Regression Model Fits")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


def plot_ransac_inliers_outliers(X, y, random_state=42):
    model = RANSACRegressor(random_state=random_state)
    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    fig, ax = plt.subplots()

    ax.scatter(X[inlier_mask], y[inlier_mask], label="Inliers")
    ax.scatter(X[outlier_mask], y[outlier_mask], label="Outliers")

    ax.set_title("RANSAC Inliers vs Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig