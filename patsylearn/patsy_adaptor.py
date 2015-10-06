import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
from patsy import dmatrix, dmatrices


### TODO: take care of intercepts

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
        X =  np.array(design_X)
        y =  np.array(design_y)
        est = clone(self.estimator)
        self.estimator_ = est.fit(X, y)
        return self

    def predict(self, data):
        X = np.array(dmatrix(self.design_X_, data))
        return self.estimator.predict(X)

    def transform(self, X):
        pass

    def score(self, data):
        pass

class PatsyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, formula):
        self.formula = formula

    def fit(self, data):
        pass

    def transform(self, data):
        pass
