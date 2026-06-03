from dataclasses import dataclass
from math import factorial, log
from typing import Any

import numpy as np


@dataclass
class _TreeNode:
    value: float
    samples: int
    depth: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None or self.left is None or self.right is None


class PermutationDecisionTreeRegressor:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 24,
        min_samples_leaf: int = 10,
        n_thresholds: int = 18,
        permutation_order: int = 3,
        order_weight: float = 0.25,
        max_features: int | None = None,
        min_impurity_decrease: float = 1e-10,
        random_state: int | None = 42,
    ) -> None:
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_thresholds = int(n_thresholds)
        self.permutation_order = int(permutation_order)
        self.order_weight = float(order_weight)
        self.max_features = max_features
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.random_state = random_state

        self.root_: _TreeNode | None = None
        self.n_features_in_: int = 0
        self.feature_importances_: np.ndarray | None = None
        self._col_medians: np.ndarray | None = None
        self._rng: np.random.Generator = np.random.default_rng(random_state)
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PermutationDecisionTreeRegressor":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(X_arr) != len(y_arr):
            raise ValueError("X and y have incompatible lengths")
        if len(y_arr) < max(2, self.min_samples_leaf):
            raise ValueError("Not enough samples to fit PermutationDecisionTreeRegressor")

        self.n_features_in_ = X_arr.shape[1]
        self._col_medians = np.nanmedian(np.where(np.isfinite(X_arr), X_arr, np.nan), axis=0)
        self._col_medians = np.where(np.isfinite(self._col_medians), self._col_medians, 0.0)
        X_arr = self._impute(X_arr)

        y_arr = np.where(np.isfinite(y_arr), y_arr, np.nan)
        valid = np.isfinite(y_arr)
        X_arr = X_arr[valid]
        y_arr = y_arr[valid]
        if len(y_arr) < max(2, self.min_samples_leaf):
            raise ValueError("Not enough finite target values to fit PermutationDecisionTreeRegressor")

        self._X = X_arr
        self._y = y_arr
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)

        self.root_ = self._build(np.arange(len(y_arr)), depth=0)

        total_importance = float(np.sum(self.feature_importances_))
        if total_importance > 0:
            self.feature_importances_ = self.feature_importances_ / total_importance

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("Model is not fitted")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_arr.shape[1]}")

        X_arr = self._impute(X_arr)
        return np.asarray([self._predict_row(row, self.root_) for row in X_arr], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        if self.root_ is None:
            return {}
        return self._node_to_dict(self.root_)

    def _node_to_dict(self, node: _TreeNode) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "value": node.value,
            "samples": node.samples,
            "depth": node.depth,
        }

        if node.is_leaf:
            payload["leaf"] = True
            return payload

        payload.update(
            {
                "leaf": False,
                "feature_index": node.feature_index,
                "threshold": node.threshold,
            }
        )
        payload["left"] = self._node_to_dict(node.left) if node.left is not None else None
        payload["right"] = self._node_to_dict(node.right) if node.right is not None else None
        return payload

    def _impute(self, X: np.ndarray) -> np.ndarray:
        medians = self._col_medians
        if medians is None:
            medians = np.zeros(X.shape[1], dtype=float)
        return np.where(np.isfinite(X), X, medians)

    def _build(self, indices: np.ndarray, depth: int) -> _TreeNode:
        assert self._y is not None

        y = self._y[indices]
        node = _TreeNode(
            value=float(np.mean(y)),
            samples=int(len(indices)),
            depth=depth,
        )

        if (
            depth >= self.max_depth
            or len(indices) < self.min_samples_split
            or len(indices) < 2 * self.min_samples_leaf
            or float(np.var(y)) <= 1e-14
        ):
            return node

        best = self._best_split(indices)
        if best is None:
            return node

        feature_idx, threshold, left_idx, right_idx, gain = best
        if gain <= self.min_impurity_decrease:
            return node

        assert self.feature_importances_ is not None
        self.feature_importances_[feature_idx] += max(gain, 0.0) * len(indices)

        node.feature_index = int(feature_idx)
        node.threshold = float(threshold)
        node.left = self._build(left_idx, depth + 1)
        node.right = self._build(right_idx, depth + 1)
        return node

    def _best_split(self, indices: np.ndarray) -> tuple[int, float, np.ndarray, np.ndarray, float] | None:
        assert self._X is not None and self._y is not None

        X_node = self._X[indices]
        y_node = self._y[indices]
        parent_impurity = self._impurity(y_node)
        n = len(indices)

        feature_ids = np.arange(self.n_features_in_)
        if self.max_features is not None and 0 < self.max_features < self.n_features_in_:
            feature_ids = self._rng.choice(feature_ids, size=self.max_features, replace=False)

        best_score = np.inf
        best_payload: tuple[int, float, np.ndarray, np.ndarray, float] | None = None

        for feature_idx in feature_ids:
            values = X_node[:, feature_idx]
            finite_values = values[np.isfinite(values)]
            if len(finite_values) < 2:
                continue

            unique = np.unique(finite_values)
            if len(unique) <= 1:
                continue

            if len(unique) <= self.n_thresholds:
                thresholds = (unique[:-1] + unique[1:]) / 2.0
            else:
                qs = np.linspace(0.05, 0.95, self.n_thresholds)
                thresholds = np.unique(np.quantile(finite_values, qs))

            for threshold in thresholds:
                left_mask = values <= threshold
                left_count = int(np.sum(left_mask))
                right_count = n - left_count

                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                left_indices = indices[left_mask]
                right_indices = indices[~left_mask]

                left_score = self._impurity(self._y[left_indices])
                right_score = self._impurity(self._y[right_indices])
                score = (left_count * left_score + right_count * right_score) / n

                if score < best_score:
                    gain = parent_impurity - score
                    best_score = score
                    best_payload = (
                        int(feature_idx),
                        float(threshold),
                        left_indices,
                        right_indices,
                        float(gain),
                    )

        return best_payload

    def _impurity(self, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) == 0:
            return 0.0

        var = float(np.var(y))
        if len(y) < self.permutation_order + 1 or var <= 1e-14 or self.order_weight <= 0:
            return var

        entropy = self._permutation_entropy(y, order=self.permutation_order)
        return var * (1.0 + self.order_weight * entropy)

    @staticmethod
    def _permutation_entropy(sequence: np.ndarray, order: int = 3) -> float:
        seq = np.asarray(sequence, dtype=float).reshape(-1)
        if order < 2 or len(seq) < order:
            return 0.0

        patterns: dict[tuple[int, ...], int] = {}
        for i in range(0, len(seq) - order + 1):
            window = seq[i : i + order]
            pattern = tuple(np.argsort(window, kind="mergesort"))
            patterns[pattern] = patterns.get(pattern, 0) + 1

        counts = np.asarray(list(patterns.values()), dtype=float)
        probs = counts / counts.sum()

        entropy = -float(np.sum(probs * np.log(probs + 1e-15)))
        max_entropy = log(factorial(order))
        if max_entropy <= 0:
            return 0.0

        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    @staticmethod
    def _predict_row(row: np.ndarray, node: _TreeNode) -> float:
        current = node

        while not current.is_leaf:
            assert current.feature_index is not None and current.threshold is not None

            if row[current.feature_index] <= current.threshold:
                assert current.left is not None
                current = current.left
            else:
                assert current.right is not None
                current = current.right

        return float(current.value)