import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.metaestimators import if_delegate_has_method
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
        return self

    def transform(self, data):
        pass
