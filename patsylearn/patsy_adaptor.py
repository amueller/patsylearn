import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import column_or_1d
from patsy import dmatrix, dmatrices


# TODO: take care of intercepts

class PatsyModel(BaseEstimator):
    def __init__(self, estimator, formula):
        self.estimator = estimator
        self.formula = formula

    def fit(self, data):
        """Fit the scikit-learn model using the formula.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Contains features and possible labels.
            Column names need to match variables in formula.
        """
        design_y, design_X = dmatrices(self.formula, data)
        self.design_y_ = design_y.design_info
        self.design_X_ = design_X.design_info
        # convert to 1d vector so we don't get a warning
        # from sklearn.
        design_y = column_or_1d(design_y)
        est = clone(self.estimator)
        self.estimator_ = est.fit(design_X, design_y)
        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, data):
        X = np.array(dmatrix(self.design_X_, data))
        return self.estimator_.predict(X)

    @if_delegate_has_method(delegate='estimator')
    def transform(self, data):
        X = np.array(dmatrix(self.design_X_, data))
        return self.estimator_.transform(X)

    @if_delegate_has_method(delegate='estimator')
    def score(self, data):
        design_infos = (self.design_y_, self.design_X_)
        design_y, design_X = dmatrices(design_infos, data)
        return self.estimator_.score(design_X, design_y)


class PatsyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, formula):
        self.formula = formula

    def fit(self, data):
        """Fit the scikit-learn model using the formula.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Contains features and possible labels.
            Column names need to match variables in formula.
        """
        self.fit_transform(data)
        return self

    def fit_transform(self, data):
        """Fit the scikit-learn model using the formula and transform it.

        Parameters
        ----------
        data : dict-like (pandas dataframe)
            Input data. Contains features and possible labels.
            Column names need to match variables in formula.

        Returns
        -------
        X_transform : ndarray
            Transformed data
        """
        design = dmatrix(self.formula, data)
        self.design_ = design.design_info
        return np.array(design)

    def transform(self, data):
        return np.array(dmatrix(self.design_, data))
