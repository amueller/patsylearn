Patsy-Learn
===========
A simple patsy to scikit-learn adaptor.
Be aware that the content of this package is pretty experimental. Feedback welcome.

This package provides two classes:

* A scikit-learn meta-estimator PatsyModel, that feeds the design matrix created by patsy into a scikit-learn estimator.
* A scikit-learn transformer, that uses a patsy formula to transform and select features.

Both classes work on pandas DataFrames. If you want to use PatsyModel for a
supervised estiamator, the input dataframe is expected to contain both the data
and the target.

Example
-------
    
    # put the iris dataset into a pandas dataframe
    from sklearn.datasets import load_iris
    import pandas as pd

    iris = load_iris()
    names = [f_name.replace(" ", "_").strip("_(cm)") for f_name in iris.feature_names]
    iris_df = pd.DataFrame(iris.data, columns=names)
    iris_df['species'] = iris.target
    
    # create logistic regression with two features
    from patsylearn import PatsyModel
    from sklearn.linear_model import LogisticRegression
    model = PatsyModel(LogisticRegression(), "species ~ sepal_length + petal_length")
    # model is an sklearn estimator.

    from sklearn.cross_validation import train_test_split
    data_train, data_test = train_test_split(iris_df)
    model.fit(data_train)
    print(model.score(data_test))

    # use PatsyTransformer to just select and transform features
    transformer = PatsyTransformer("sepal_length + np.log(petal_length) + petal_length:sepal_width")
    # transformer is a scikit-learn transformer

    transformer.fit(data)
    print(transformer.transform(data).shape)
