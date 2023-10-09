import numpy as np
import io
import pandas as pd
import gc
import inspect #use to find the name of a dataframe

from math import prod

def df_analysis(df, name_df, *args, **kwargs):
    """
    Method used to analyze on the DataFrame.

    Parameters:
    -----------------
        df : pandas.DataFrame
        Dataset to analyze.
        
        name_df : str
        Dataset name.
        
        *args, **kwargs:
        -----------------
            columns : list
            Dataframe keys in list format.
            
            flag : str
            Flag to show complete information about the dataset to analyse
            "complete" shows all information about the dataset.

    Returns:
    -----------------
        None. 
        Print the analysis on the Dataset. 
        
    """
    
    # Getting the variables
    columns = kwargs.get("columns", None)
    type_analysis = kwargs.get("type_analysis", None)
    
    ORDERING_COMPLETE = [
        "name", "type", "records", "unique", "# NaN", "% NaN", "mean", "min", "25%", "50%", "75%", "max", "std"
    ]
    
    # Calculating the memory usage based on dataframe.info()
    buf = io.StringIO()
    df.info(buf=buf)
    memory_usage = buf.getvalue().split('\n')[-2]

    if df.empty:
        print("The", name_df, "dataset is empty. Please verify the file.")
    else:
        empty_cols = [col for col in df.columns if df[col].isna().all()] # identifying empty columns
        df_rows_duplicates = df[df.duplicated()] #identifying full duplicates rows
        
        # Creating a dataset based on Type object and records by columns
        type_cols = df.dtypes.apply(lambda x: x.name).to_dict() 
        df_resume = pd.DataFrame(list(type_cols.items()), columns = ["name", "type"])
        df_resume["records"] = list(df.count())
        df_resume["# NaN"] = list(df.isnull().sum())
        df_resume["% NaN"] = list(((df.isnull().sum() / len(df.index))*100).round(2))
        
        print("\nAnalysis of", name_df, "dataset")
        print("--------------------------------------------------------------------")
        print("- Dataset shape:                 ", df.shape[0], "rows and", df.shape[1], "columns")
        print("- Total of NaN values:           ", df.isna().sum().sum())
        print("- Percentage of NaN:             ", round((df.isna().sum().sum() / prod(df.shape)) * 100, 2), "%")
        print("- Total of full duplicates rows: ", df_rows_duplicates.shape[0])
        print("- Total of empty rows:           ", df.shape[0] - df.dropna(axis="rows", how="all").shape[0]) if df.dropna(axis="rows", how="all").shape[0] < df.shape[0] else \
                    print("- Total of empty rows:            0")
        print("- Total of empty columns:        ", len(empty_cols))
        print("  + The empty column is:         ", empty_cols) if len(empty_cols) == 1 else \
                    print("  + The empty column are:         ", empty_cols) if len(empty_cols) >= 1 else None
        print("- Unique indexes:                ", df.index.is_unique)
        
        if columns is not None:
            print("\n- The key(s):", columns, "is not present multiple times in the dataframe.\n  It CAN be used as a primary key.") if df.size == df.drop_duplicates(columns).size else \
                print("\n- The key(s):", columns, "is present multiple times in the dataframe.\n  It CANNOT be used as a primary key.")
        
        if type_analysis == "summarized":
            print("\n")
        
        if type_analysis is None or type_analysis != "summarized":
            pd.set_option("display.max_rows", None) # show full of showing rows
            pd.set_option("display.max_columns", None) # show full of showing cols
            pd.set_option("display.max_colwidth", None) # show full width of showing cols
            pd.set_option("display.float_format", lambda x: "%.5f" % x) # show full content in cell    
            
            if type_analysis is None or type_analysis != "complete":
                print("\n- Type object and records by columns      (",memory_usage,")")
                print("--------------------------------------------------------------------")
            elif type_analysis == "complete":
                df_resume["unique"] = list(df.nunique())
                df_desc = pd.DataFrame(df.describe().T).reset_index()
                df_desc = df_desc.rename(columns={"index": "name"})
                df_resume = df_resume.merge(right=df_desc[["name", "mean", "min", "25%", "50%", "75%", "max", "std"]], on="name", how="left")
                df_resume = df_resume[ORDERING_COMPLETE]
                print("\n- Type object and records by columns      (",memory_usage,")")
                print("--------------------------------------------------------------------")
                
            display(df_resume.sort_values("records", ascending=False))
            
            pd.reset_option("display.max_rows") # reset max of showing rows
            pd.reset_option("display.max_columns") # reset max of showing cols
            pd.reset_option("display.max_colwidth") # reset width of showing cols
            pd.reset_option("display.float_format") # reset show full content in cell
            
        # deleting dataframe to free memory
        if type_analysis == "complete":
            del [[df_resume, df_desc]]
            gc.collect()
            df_resume, df_desc = (pd.DataFrame() for i in range(2))
        else:
            del df_resume
            gc.collect()
            df_resume = pd.DataFrame()




def plot_validation_curve_by(estimator, name_model, validation_by, X_train, y_train, param_name, param_range, param_name_short, scoring=None, cv=None, enable=True):
    """
    Generate 1 plots: 
        1. The test and training validation curve
    
    Parameters
    -----------------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.
        
    name_model : str
        Name of the model as title for the chart. 
        
    validation_by : str 
        Name of the metric to plot the curve. Possible values ["R2SCORE", "MAE"]
        
    X_train : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y_train : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.      
        
    param_name : str
        Name of the parameter that will be varied.
    
    param_range : array-like of shape (n_values,)
        The values of the parameter that will be evaluated.
    
    param_name_short : str
        Short name for param_name
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    
    enable: bool, if False, don't execute the function 
        
    Returns:
    -----------------
        None. 
        Plot the graph. 
        
    """    
    # Validation of execution of the function
    if enable :
        
        # Initializing figure
        fig = plt.figure(figsize=(8, 6))

        if validation_by == "R2SCORE":

            # Get the validation_curves results
            train_scores, test_scores = validation_curve(estimator, X_train, y_train, param_name=param_name, param_range=param_range, cv=kfold)

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            plot = sns.lineplot(x=param_range, y=train_scores_mean, label="Train", marker="o")
            plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="blue")

            plot = sns.lineplot(x=param_range, y=test_scores_mean, label="Validation", marker="o")
            plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="orange")

            plt.legend(bbox_to_anchor=(1.22, 1), borderaxespad=0)

            plt.title(name_model + " Validation curve", fontdict={ "fontsize": 16, "fontweight": "normal" })
            plot.set(xlabel=param_name_short, ylabel="R2-score".translate(SUP))

        elif validation_by == "MAE":

            # Get the validation_curves results
            train_scores, test_scores = validation_curve(estimator, X_train, y_train, param_name=param_name, param_range=param_range, scoring=scoring, cv=kfold)

            train_errors, test_errors = -train_scores, -test_scores

            plot = plt.errorbar(param_range, train_errors.mean(axis=1), yerr=train_errors.std(axis=1), label="Train")
            plot = plt.errorbar(param_range, test_errors.mean(axis=1), yerr=test_errors.std(axis=1), label="Validation")

            plt.legend(bbox_to_anchor=(1.235, 1), borderaxespad=0)

            plt.ylabel("MAE\n(smaller is better)")
            plt.xlabel(param_name_short)
            _ = plt.title(name_model + " Validation curve", fontdict={ "fontsize": 16, "fontweight": "normal" })

        plt.tight_layout()
        plt.savefig("img/" + name_model + "-validation-curve-by-" + validation_by + ".png")
        sns.despine(fig)
        plt.show()
    
            
def plot_learning_curve(estimator, name_model, X_train, y_train, cv=None, train_sizes=np.linspace(0.2, 1.0, 10), enable=True):
    """
    Generate 3 plots: 
        1. The test and training learning curve
        2. The training samples vs fit times curve
        3. The fit times vs score curve
    
    Parameters
    -----------------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.
        
    name_model : str
        Name of the model as title for the chart.        
        
    X_train : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y_train : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.        
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5) or (0.2, 1.0, 10))
        
    enable: bool, if False, don't execute the function
        
    Returns:
    -----------------
        None. 
        Plot the graphs. 
        
    """
    # Validation of execution of the function
    if enable :
        
        # Get the learning_curves results
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X_train, y_train, cv=cv, \
                                                                               train_sizes=train_sizes, n_jobs=-1, return_times=True)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Initializing figure
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))

        # Plot learning curve
        plot = sns.lineplot(x=train_sizes, y=train_scores_mean, label="Train", marker="o", ax=ax1)
        ax1.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="blue")

        plot = sns.lineplot(x=train_sizes, y=test_scores_mean, label="Validation", marker="o", ax=ax1)
        ax1.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="orange")

        ax1.legend(loc="best")
        ax1.set_title(" Learning curve (" + name_model + ")", fontdict={ "fontsize": 16, "fontweight": "normal" })
        plot.set(xlabel="Training examples", ylabel="R2-score".translate(SUP))

        # Plot Scalability of the model
        plot = sns.lineplot(x=train_sizes, y=fit_times_mean, marker="o", ax=ax2)
        ax2.fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1, color="blue")

        ax2.set_title("Scalability of the model", fontdict={ "fontsize": 16, "fontweight": "normal" })
        plot.set(xlabel="Training examples", ylabel="Fit times")

        # Performance of the model
        plot = sns.lineplot(x=fit_times_mean, y=test_scores_mean, marker="o", ax=ax3)
        ax3.fill_between(fit_times_mean, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1)

        ax3.set_title("Performance of the model", fontdict={ "fontsize": 16, "fontweight": "normal" })
        plot.set(xlabel="Fit times examples", ylabel="R2-score".translate(SUP))

        plt.savefig("img/" + name_model + "-performance-model.png")
        sns.despine(fig)
        plt.show()


def plot_cross_val_predi(estimator, name_model, target_variable, X_train, y_train, cv=None, enable=True):
    """
    
    
    Generate 1 plots: 
        1. The validation between real values vs predicted values
    
    Parameters
    -----------------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.
        
    name_model : str
        Name of the model as title for the chart.   

    target_variable : str
        Name of the target variable.    
        
    X_train : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y_train : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.      
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
        
    enable: bool, if False, don't execute the function
        
    Returns:
    -----------------
        None. 
        Plot the graph. 
        
    """   

    # Validation of execution of the function
    if enable :
        
        # Get the predicted values
        predicted = cross_val_predict(estimator, X_train, y_train, cv=kfold)

        # Initializing figure
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))

        # Main title
        plt.suptitle("Real values vs Predicted values", size=24)

        ax1.scatter(predicted, y_train, edgecolors=(0, 0, 0))
        ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "--k", lw=4)

        ax1.text(min(y_train)+0.2, 0.98*max(y_train), r'$R^2$ = %.2f, RMSE = %.2f' % (
                round(r2_score(y_train, predicted), 3),
                round(math.sqrt(mean_squared_error(y_train, predicted)), 3)), 
                style="italic", fontsize=13,
                bbox={"facecolor": "grey", "alpha": 0.4, "pad": 5})
        ax1.set_title(name_model + " Cross-Values Predictions", fontdict={ "fontsize": 16, "fontweight": "normal" })
        ax1.set_xlabel("Predicted values")
        ax1.set_ylabel("Real values")


        ax2.scatter(predicted, (y_train - predicted), edgecolors=(0, 0, 0))
        ax2.hlines(y=0, xmin=predicted.min(), xmax=predicted.max(), colors="red", linestyles="--", lw=4)
        ax2.set_title(name_model + " Residuals", fontdict={ "fontsize": 16, "fontweight": "normal" })
        ax2.set_xlabel("Predicted values")
        ax2.set_ylabel("Standardized residuals")


        sns.kdeplot(y_train, color="r", label="Real values", ax=ax3)
        sns.kdeplot(predicted, color="b", label="Predicted values", ax=ax3)

        ax3.set_title(name_model + " Distribution plot based on density", fontdict={ "fontsize": 16, "fontweight": "normal" })
        ax3.set_xlabel("SiteEnergyUse(kBtu)")
        ax3.set_ylabel("Density")
        plt.legend(bbox_to_anchor=(0.99, 0.99), borderaxespad=0)

        plt.savefig("img/" + name_model + "-cross-val-predict.png")
        plt.show()
    

def plot_features_importance(estimator, name_model, X_train, y_train, scoring=None, enable=True):
    """
    Generate 1 plots: 
        1. The importance by feature
    
    Parameters
    -----------------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.
        
    name_model : str
        Name of the model as title for the chart.     
        
    X_train : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y_train : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning. 
        
    scoring : string, callable or None, default=None
        Scorer to use. It can be a single string or a callable. 
        If None, the estimator’s default scorer is used. 
        
    enable: bool, if False, don't execute the function
        
    Returns:
    -----------------
        None. 
        Plot the graph. 
        
    """
    
    # Validation of execution of the function
    if enable :
        
        # Get the importance by feature
        results = permutation_importance(estimator, X_train, y_train, scoring=scoring)

        # Making a dataframe to work easily
        df_importance = pd.DataFrame({
                            "Feature" : X_train.columns,
                            "Importance" : results.importances_mean
                        })

        # Sorting by importance before plotting
        df_importance = df_importance.sort_values("Importance")

        # Initializing figure    
        fig = plt.subplots(figsize=(10, 8))

        plot = sns.barplot(data=df_importance, y=df_importance["Feature"], x=df_importance["Importance"])

        plt.title(name_model + " Features Importance", fontdict={ "fontsize": 16, "fontweight": "normal" })
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig("img/" + name_model + "-feature-importance.png")
        plt.show()