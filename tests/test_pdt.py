import numpy as np

from app.ml.pdt import PermutationDecisionTreeRegressor


def test_pdt_fits_simple_nonlinear_pattern() -> None:
    rng = np.random.default_rng(1)
    x = np.linspace(-2, 2, 220)
    X = np.column_stack([x, x**2, np.sin(3 * x)])
    y = np.where(x < 0, -0.05 + 0.02 * x, 0.04 + 0.03 * np.sin(3 * x))
    y = y + rng.normal(0, 0.003, size=len(y))

    model = PermutationDecisionTreeRegressor(max_depth=4, min_samples_leaf=8, min_samples_split=20)
    model.fit(X, y)
    pred = model.predict(X)
    mse = np.mean((pred - y) ** 2)

    assert mse < 0.001
    assert model.feature_importances_ is not None
    assert np.isclose(model.feature_importances_.sum(), 1.0)