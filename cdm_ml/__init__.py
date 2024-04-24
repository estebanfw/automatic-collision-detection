# MACHINE LEARNING MODULE
# IN THIS SCRIPT YOU WILL FIND ALL THE FUNCTIONS NEEDED FOR COMPUTING THE LGBM REGRESSION MODEL

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import datetime as dt
import pickle
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import preparing_data as F


# Bayesian optimization function using Light Gradient Boosting Regression Model
def bayesian_opt_lgbm(
    X,
    y,
    init_iter=3,
    n_iters=7,
    random_state=11,
    seed=101,
    num_iterations=100,
    evalm="lgb_r2",
):
    """Computes the optimal values for the LGBM model

    Parameters
    ----------
    X : dataframe
        Train dataset
    y : target dataframe
        Values to be predicted
    init_iter: int
        Initial number of iterations
    n_iters: int
        Total number of iterations of the Bayesian Optimization
    random_state: int
    seed: int
        Initial number to compute random number to startup calculation
    num_iterations: int
        Number of trees
    evalm: lgb_r2 lgb_rmse lgb_mae lgb_adjusted_r2
        String use to indicate which metric will be used as objective of the optimization

    Returns
    -------
    list
        Optimal hyperparemeters for the regression model.
    dictionary
        Input values for the model
    """
    dtrain = lgb.Dataset(data=X, label=y)

    # Metric evaluation functions
    def lgb_r2(preds, dtrain):  # R2
        labels = dtrain.get_label()
        return "metric", r2_score(labels, preds), True

    def lgb_rmse(preds, dtrain):  # RMSE
        labels = dtrain.get_label()
        return "metric", mean_squared_error(labels, preds, squared=False), True

    def lgb_mae(preds, dtrain):  # MAE
        labels = dtrain.get_label()
        return "metric", mean_absolute_error(labels, preds), True

    def lgb_adjusted_r2(preds, dtrain):  # ADJUSTED R2
        labels = dtrain.get_label()
        n = dtrain.num_data()
        k = dtrain.num_feature()
        return "metric", ((1 - r2_score(labels, preds)) * (n - 1)) / (n - k - 1), True

    metrics_dict = {
        "lgb_r2": lgb_r2,
        "lgb_rmse": lgb_rmse,
        "lgb_mae": lgb_mae,
        "lgb_adjusted_r2": lgb_adjusted_r2,
    }
    # Select metric
    metric = str(evalm)
    metric_feval = metrics_dict.get(str(evalm))

    # Objective Function
    def hyp_lgbm(
        num_leaves,
        feature_fraction,
        learning_rate,
        bagging_fraction,
        max_depth,
        min_split_gain,
        min_child_weight,
    ):
        params = {
            "application": "regression",
            "num_iterations": num_iterations,
            "early_stopping_round": 50,
            "verbose": -1,
            "metric": metric,
        }  # Default parameters
        params["num_leaves"] = int(round(num_leaves))
        params["learning_rate"] = learning_rate
        params["feature_fraction"] = max(min(feature_fraction, 1), 0)
        params["bagging_fraction"] = max(min(bagging_fraction, 1), 0)
        params["max_depth"] = int(round(max_depth))
        params["min_split_gain"] = min_split_gain
        params["min_child_weight"] = min_child_weight
        cv_results = lgb.cv(
            params,
            dtrain,
            nfold=5,
            seed=seed,
            categorical_feature=[],
            stratified=False,
            verbose_eval=None,
            feval=metric_feval,
        )
        # print(cv_results)
        return np.max(cv_results["metric-mean"])

        # Domain space-- Range of hyperparameters

    pds = {
        "num_leaves": (2, 120),
        "feature_fraction": (0.1, 0.9),
        "bagging_fraction": (1, 1),
        "max_depth": (7, 15),
        "learning_rate": (0.001, 0.05),
        "min_split_gain": (0.001, 0.1),
        "min_child_weight": (10, 35),
    }
    # Surrogate model
    optimizer = BayesianOptimization(hyp_lgbm, pds, random_state=random_state)

    # Optimize
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)

    # Output dictionary
    output_dict = optimizer.max["params"]
    output_dict["num_iterations"] = n_iters #of bayesian optimization
    output_dict["n_estimators"] = num_iterations  #number of trees

    # Save dictionary to file
    filename = "./opt_parameters_bo/param_{}_{}.pkl".format(
        dt.datetime.now().strftime("%Y%m%d_%H%M%S"), metric
    )
    a_file = open(filename, "wb")
    pickle.dump(output_dict, a_file)
    a_file.close()

    return optimizer, output_dict


# Function to create and validate a new model
def create_and_validate_model(
    X,
    y,
    X_test,
    Y_test,
    init_iter=5,
    n_iters=500,
    random_state=77,
    seed=101,
    num_iterations=300,
    evalm="lgb_r2",
    hp_metric="regression_L2",
):
    """Creates a new model LGBM regression model and validates it with the testing dataset

    Parameters
    ----------
    X : dataframe
        Train dataset
    y : target dataframe
        Values to be predicted
    X_test : dataframe
        Dataframe used to test model
    y_test : target dataframe
        Values to be predicted on the testing dataframe
    init_iter: int
        Initial number of iterations
    n_iters: int
        Total number of iterations of the Bayesian Optimization
    random_state: int
    seed: int
        Initial number to compute random number to startup calculation
    num_iterations: int
        Number of trees
    evalm: lgb_r2 lgb_rmse lgb_mae lgb_adjusted_r2
        String use to indicate which metric will be used as objective of the optimization
    hp_metric: "regression_L2"
        Metric used to optimize model after the computation. It is recommended to use the one corresponding with the evaluation metric used to create the model at first place

    Returns
    -------
    model
        Light Gradient Boosting Model
    dataframe
        Dataframe comparing true vs predicted values
    """
    bayesian = bayesian_opt_lgbm(
        X, y, init_iter, n_iters, random_state, seed, num_iterations, evalm
    )
    opt_parameters = bayesian[1]
    print("------------------------ OPTIMAL PARAMETERS ------------------------")
    print(opt_parameters)
    print("-------------------------------------------------------------------")

    # LOAD OPTIMAL PARAMETERS FOR FURTHER COMPUTATION
    hyper_params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": str(hp_metric),
        "learning_rate": opt_parameters.get("learning_rate"),
        "feature_fraction": opt_parameters.get("feature_fraction"),
        "bagging_fraction": opt_parameters.get("bagging_fraction"),
        "verbose": -1,
        "max_depth": int(round(opt_parameters.get("max_depth"))),
        "num_leaves": int(round(opt_parameters.get("num_leaves"))),
        "min_split_gain": opt_parameters.get("min_split_gain"),
        "num_iterations": opt_parameters.get("num_iterations"),
        "n_estimators": opt_parameters.get("n_estimators"),
        "min_child_weight": opt_parameters.get("min_child_weight"),
    }
    # TRAIN MODEL WITH OPTIMAL PARAMETERS
    lgbm_train = lgb.Dataset(X, label=y)
    gbm = lgb.train(params=hyper_params, train_set=lgbm_train)

    # TEST MODEL WITH TESTING SUBPART OF DATASET
    Y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # REGRESION MODEL METRICS
    print("The r2 of prediction is:", r2_score(Y_test, Y_pred))
    print("The MSE of prediction is:", mean_squared_error(Y_test, Y_pred, squared=True))
    print(
        "The RMSE of prediction is:", mean_squared_error(Y_test, Y_pred, squared=False)
    )
    print("The MAE of prediction is:", mean_absolute_error(Y_test, Y_pred))

    # COMPARE TEST VALUES VS PREDICTED VALUES
    df_results = compare_true_vs_prediction(df_true=Y_test, df_pred=Y_pred)

    # WRITE TO A FILE

    outF = open(
        "./validation-results/r_evalm-{}_hp_metric-_{}_{}.txt".format(
            str(evalm), str(hp_metric), dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
        "w+",
    )
    # write line to output file
    outF.write(
        "------------------------- MODEL HYPER-PARAMETERS ------------------------- \n"
    )
    outF.write(str(hyper_params))
    outF.write("\n")
    outF.write("\n")
    outF.write(
        "------------------------ REGRESSION MODEL METRICS ------------------------ \n"
    )
    outF.write(
        str("The r2 of prediction is: ") + str(r2_score(Y_test, Y_pred)) + str("\n")
    )
    outF.write(
        str("The MSE of prediction is: ")
        + str(mean_squared_error(Y_test, Y_pred, squared=True))
        + str("\n")
    )
    outF.write(
        str("The RMSE of prediction is: ")
        + str(mean_squared_error(Y_test, Y_pred, squared=False))
        + str("\n")
    )
    outF.write(
        str("The MAE of prediction is: ")
        + str(mean_absolute_error(Y_test, Y_pred))
        + str("\n")
    )
    outF.write("\n")
    outF.write("\n")
    outF.write(
        "-------------------------- ADDITIONAL COMMENTS -------------------------- \n"
    )
    outF.write(
        str(
            "This model was created and validated at {}".format(
                dt.datetime.fromtimestamp(dt.datetime.timestamp(dt.datetime.now()))
            )
        )
    )
    outF.close()

    return gbm, df_results


# Function to load a previously created model and validate it
def load_and_validate_model(name, X, y, X_test, Y_test, hp_metric="regression_L2"):
    """Loads a model LGBM regression model and validates it with the testing dataset

    Parameters
    ----------
    pkl: python pickle file
        Input file where hyperparameters of the already computed model are stored.
    X : dataframe
        Train dataset
    y : target dataframe
        Values to be predicted
    X_test : dataframe
        Dataframe used to test model
    Y_test : target dataframe
        Values to be predicted on the testing dataframe
    hp_metric: "regression_L2"
        Metric used to optimize model after the computation. It is recommended to use the one corresponding with the evaluation metric used to create the model at first place

    Returns
    -------
    model
        Light Gradient Boosting Model
    dataframe
        Dataframe comparing true vs predicted values
    """
    filename = str("./opt_parameters_bo/{}.pkl".format(name))
    a_file = open(filename, "rb")
    opt_parameters = pickle.load(a_file)
    print("------------------------ OPTIMAL PARAMETERS ------------------------")
    print(opt_parameters)
    print("-------------------------------------------------------------------")

    # LOAD OPTIMAL PARAMETERS FOR FURTHER COMPUTATION
    hyper_params = {
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": str(hp_metric),
        "learning_rate": opt_parameters.get("learning_rate"),
        "feature_fraction": opt_parameters.get("feature_fraction"),
        "bagging_fraction": opt_parameters.get("bagging_fraction"),
        "verbose": -1,
        "max_depth": int(round(opt_parameters.get("max_depth"))),
        "num_leaves": int(round(opt_parameters.get("num_leaves"))),
        "min_split_gain": opt_parameters.get("min_split_gain"),
        "num_iterations": opt_parameters.get("num_iterations"),
        "n_estimators": opt_parameters.get("n_estimators"),
        "min_child_weight": opt_parameters.get("min_child_weight"),
    }
    # TRAIN MODEL WITH OPTIMAL PARAMETERS
    lgbm_train = lgb.Dataset(X, label=y)
    gbm = lgb.train(params=hyper_params, train_set=lgbm_train)

    # TEST MODEL WITH TESTING SUBPART OF DATASET
    Y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # REGRESION MODEL METRICS
    print("The r2 of prediction is:", r2_score(Y_test, Y_pred))
    print("The MSE of prediction is:", mean_squared_error(Y_test, Y_pred, squared=True))
    print(
        "The RMSE of prediction is:", mean_squared_error(Y_test, Y_pred, squared=False)
    )
    print("The MAE of prediction is:", mean_absolute_error(Y_test, Y_pred))

    # COMPARE TEST VALUES VS PREDICTED VALUES
    df_results = compare_true_vs_prediction(df_true=Y_test, df_pred=Y_pred)

    # WRITE TO A FILE

    outF = open(
        "./validation-results/r_hp_metric-_{}_{}.txt".format(
            str(hp_metric), dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
        "w+",
    )
    # write line to output file
    outF.write(
        "------------------------- MODEL HYPER-PARAMETERS ------------------------- \n"
    )
    outF.write(str(hyper_params))
    outF.write("\n")
    outF.write("\n")
    outF.write(
        "------------------------ REGRESSION MODEL METRICS ------------------------ \n"
    )
    outF.write(
        str("The r2 of prediction is: ") + str(r2_score(Y_test, Y_pred)) + str("\n")
    )
    outF.write(
        str("The MSE of prediction is: ")
        + str(mean_squared_error(Y_test, Y_pred, squared=True))
        + str("\n")
    )
    outF.write(
        str("The RMSE of prediction is: ")
        + str(mean_squared_error(Y_test, Y_pred, squared=False))
        + str("\n")
    )
    outF.write(
        str("The MAE of prediction is: ")
        + str(mean_absolute_error(Y_test, Y_pred))
        + str("\n")
    )
    outF.write("\n")
    outF.write("\n")
    outF.write(
        "-------------------------- ADDITIONAL COMMENTS -------------------------- \n"
    )
    outF.write(
        str(
            "This model was created and validated at {}".format(
                dt.datetime.fromtimestamp(dt.datetime.timestamp(dt.datetime.now()))
            )
        )
    )
    outF.close()

    return gbm, df_results

# Function to compare true vs predicted values
def compare_true_vs_prediction(df_true, df_pred):
    """Compares results of the prediction vs true values

    Parameters
    ----------
    df_true : dataframe
        Dataframe with true values
    df_pred : target dataframe
        Dataframe with predicted values

    Returns
    -------
    dataframe
        Dataframe comparing true values vs predicted values
    """
    aux_y = pd.DataFrame(df_true)
    aux_y.reset_index(inplace=True)
    aux_y.drop(["index"], inplace=True, axis=1)
    aux_y_pred = pd.DataFrame(df_pred)
    aux_y_pred.reset_index(inplace=True)
    aux_y_pred.drop(["index"], inplace=True, axis=1)
    frames = [aux_y, aux_y_pred]
    result = pd.concat(frames, axis=1)
    result.columns = ["y_true", "y_predicted"]
    result["y_true_10"] = 10 ** result.y_true
    result["y_predicted_10"] = 10 ** result.y_predicted
    result[result["y_true_10"] > 0.00001]
    result[result["y_true_10"] > 0.0001]
    return result