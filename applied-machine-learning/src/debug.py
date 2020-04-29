import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    preds = []

    for d in [1, 3, 6, 9]:
        poly = PolynomialFeatures(degree=d)
        X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
        model = LinearRegression().fit(X_train_poly, y_train)
        X_test_poly = poly.transform(np.linspace(0, 10, 100).reshape(-1, 1))
        preds.append(model.predict(X_test_poly))

    return np.vstack(preds)


def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    train_scores = np.array([])
    test_scores = np.array([])
    for d in range(10):
        poly = PolynomialFeatures(degree=d)
        X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
        X_test_poly = poly.transform(X_test.reshape(-1, 1))
        model = LinearRegression().fit(X_train_poly, y_train)
        train_scores = np.append(train_scores, model.score(X_train_poly, y_train))
        test_scores = np.append(test_scores, model.score(X_test_poly, y_test))

    return (train_scores, test_scores)


def answer_three():
    _, test_scores = answer_two()
    best = np.argmax(test_scores)
    return (best - 1, best + 1, best)


def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    poly = PolynomialFeatures(degree=12)
    X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
    X_test_poly = poly.transform(X_test.reshape(-1, 1))

    scores = []
    for m in [LinearRegression(), Lasso(alpha=0.01, max_iter=10000)]:
        model = m.fit(X_train_poly, y_train)
        scores.append(model.score(X_test_poly, y_test))

    return tuple(scores)


def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    classes = model.feature_importances_.argsort()[-5:][::-1]

    return X_train2.columns[classes].values.tolist()


def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    model = SVC(random_state=0)
    train_scores, test_scores = validation_curve(
        model,
        X_subset,
        y_subset,
        param_name="gamma",
        param_range=np.logspace(-4, 1, 6),
        scoring="accuracy",
    )
    return (train_scores.mean(axis=1), test_scores.mean(axis=1))


def answer_seven():
    train_scores, test_scores = answer_six()
    gammas = np.logspace(-4, 1, 6)
    best = test_scores.argmax()
    under = test_scores[:best].argmin()
    over = test_scores[best:].argmin() + best
    return (gammas[under], gammas[over], gammas[best])


if __name__ == "__main__":
    mush_df = pd.read_csv("../mushrooms.csv")
    mush_df2 = pd.get_dummies(mush_df)

    X_mush = mush_df2.iloc[:, 2:]
    y_mush = mush_df2.iloc[:, 1]

    # use the variables X_train2, y_train2 for Question 5
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_mush, y_mush, random_state=0
    )

    # For performance reasons in Questions 6 and 7, we will create a smaller version of the
    # entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
    # the 25% test split created above as the representative subset.
    #
    # Use the variables X_subset, y_subset for Questions 6 and 7.
    X_subset = X_test2
    y_subset = y_test2

    res = answer_three()

    print("debug")
