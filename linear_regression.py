"""Import packages"""
import logging
from pandas import DataFrame
import numpy as np

# Sci-kit learn library
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

# Configuration of logging
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()
logger.setLevel(level=logging.INFO)


def training(input_df: DataFrame):
    """_Linear Regression training

    Args:
        input_df (DataFrame): _description_

    Returns:
        _type_: train, test, model
    """
    test_size_value = 0.30
    seed_value = 42
    logger.info("Splitting input datrafame in train and test samples")
    logger.info("Test size %s with randon_state=%s", test_size_value, seed_value)
    train, test = train_test_split(input_df, test_size=test_size_value, random_state=seed_value)
    df = train
    logger.info("Train dataframe dimension %s x %s", train.shape[0], train.shape[1])
    logger.info("Test dataframe dimension %s x %s", test.shape[0], test.shape[1])

    field_to_predict = "TARGET_MD"
    logger.info("Field to predict %s", field_to_predict)
    y_df = df[field_to_predict]
    x_df = df.drop([field_to_predict], axis=1)
    x, y = np.array(x_df), np.array(y_df)

    logger.info("Starting fitting of Linear Regression Model")
    model = LinearRegression().fit(x, y)
    logger.info("Linear Regression model has been built!")
    logger.info("Printing metrics of the model")
    r_sq = model.score(x, y)
    print("coefficient of determination:", r_sq)
    print("intercept:", model.intercept_)
    print("slope:", model.coef_)

    return train, test, model


def testing(model, test):
    """Linear Regression Testing

    Args:
        model (_type_): model input
        test (_type_): test dataframe input

    Returns:
        _type_: predicted values as dataframe
    """
    field_to_predict = "TARGET_MD"
    y_test = test[field_to_predict]
    x_test = test.drop([field_to_predict], axis=1)
    logger.info("Running prediction")
    prediction = model.predict(x_test)
    logger.info("Computing prediction metrics")
    print("The r2 of prediction is:", r2_score(y_test, prediction))
    print("The MSE of prediction is:", root_mean_squared_error(y_test, prediction) ** 2)
    print("The RMSE of prediction is:", root_mean_squared_error(y_test, prediction))
    print("The MAE of prediction is:", mean_absolute_error(y_test, prediction))
    logger.info("Run finished!")
    return prediction
