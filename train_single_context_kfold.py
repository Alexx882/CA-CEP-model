import pandas as pd
from pandas import DataFrame

import numpy as np
import collections


from pandas import DataFrame
from typing import Iterator, Tuple, List, Dict

def chunks(lst: DataFrame, n) -> Iterator[DataFrame]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_k_folds(dataframe: DataFrame, k: int = 10) -> Iterator[Tuple[DataFrame, DataFrame]]:
    """
    Folds the dataframe k times and returns each fold for training
    :returns: k-1 folds for training, 1 fold for testing
    """
    
    fold_size = int(len(dataframe) / k)
    folds = [c for c in chunks(dataframe, fold_size)]
    
    if len(folds) != k:
        print(f"#folds={len(folds)} do not match k={k}! "\
            f"Merging last 2 folds with sizes={len(folds[-2])}, {len(folds[-1])}")
        folds[-2:] = [pd.concat([folds[-2], folds[-1]])]
        print(f"#folds={len(folds)}, new size last fold={len(folds[-1])}")
        
    for i in range(k):
        yield pd.concat([f for (idx, f) in enumerate(folds) if idx != i]), folds[i]
        

def remove_empty_community_class(df):
    '''Removes evolution_label -1 from dataset indicating the community stays empty.'''
    import warnings
    warnings.filterwarnings("ignore")
    df['evolution_label'] = df['evolution_label'].replace(-1.0, 0)
    warnings.filterwarnings("default")
    return df


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from entities.repeated_training_result import RepeatedTrainingResult
repeated_result: Dict[str, RepeatedTrainingResult] = {} 

from processing import DataSampler
sampler = DataSampler()


import sklearn.metrics

def export_report(layer_name: str):
    '''Exports the global RepeatedTrainingResult with all contents.'''
    fpath = f'data/{use_case}/ml_output/{approach}'
    Path(fpath).mkdir(parents=True, exist_ok=True)

    with open(f"{fpath}/results.csv", 'a') as file:
        for model_name, result in repeated_result.items():
            file.write(f"{layer_name}, {model_name}, {result.get_all_metrics_as_str()}\n")


def print_report(clfs: list, test_Xs: list, test_Y: 'y', titles: list):
    '''Adds the classification report result to the global RepeatedTrainingResult.'''
    for clf, test_X, title in zip(clfs, test_Xs, titles):
        pred_Y = clf.predict(test_X)        
        cls_report: dict = sklearn.metrics.classification_report(y_true=test_Y, y_pred=pred_Y, output_dict=True)
        if title not in repeated_result:
            repeated_result[title] = RepeatedTrainingResult()
        repeated_result[title].add_classification_report(cls_report)


import pickle 
from pathlib import Path

def export_model(model, model_name):
    fpath = f'data/{use_case}/ml_output/{approach}/{layer_name}'
    Path(fpath).mkdir(parents=True, exist_ok=True)
    with open(f'{fpath}/{layer_name}_{model_name}.model', 'wb') as f:
        pickle.dump(model, f)


def run():
    from sklearn.naive_bayes import GaussianNB
    priors = None #np.array([19,16,16,74,74]) / (19+16+16+74+74)
    smoothing = 0

    clf = GaussianNB(priors=priors, var_smoothing=smoothing)
    clf.fit(train_X, train_Y)

    clf_p = GaussianNB(priors=priors, var_smoothing=smoothing)
    clf_p.fit(train_Xp, train_Y)

    print_report([clf, clf_p], [test_X, test_Xp], test_Y, ["nb X", "nb Xp"])

    # export_model(clf, 'nb_x')
    # export_model(clf_p, 'nb_xp')



    from sklearn.svm import LinearSVC
    c = 1
    dual = False
    tol = 1E-4

    svc = LinearSVC(C=c, dual=dual, tol=tol)
    svc.fit(train_X, train_Y)

    svc_p = LinearSVC(C=c, dual=dual, tol=tol)
    svc_p.fit(train_Xp, train_Y)

    print_report([svc, svc_p], [test_X, test_Xp], test_Y, ["svc X", "svc Xp"])

    # export_model(svc, 'svc_x')
    # export_model(svc_p, 'svc_xp')



    from sklearn.neighbors import KNeighborsClassifier
    n_neighbors = 20
    weights = 'uniform'
    algo = 'auto'
    leaf_size = 30

    knnc = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algo, leaf_size=leaf_size)
    knnc.fit(train_X, train_Y)

    knnc_p = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights,  algorithm=algo, leaf_size=leaf_size)
    knnc_p.fit(train_Xp, train_Y)

    print_report([knnc, knnc_p], [test_X, test_Xp], test_Y, ["knn X", "knn Xp"])

    # export_model(knnc, 'knn_x')
    # export_model(knnc_p, 'knn_xp')



    from sklearn.tree import DecisionTreeClassifier 
    criterion = 'gini'
    splitter = 'random'
    max_depth = 10
    min_samples_leaf = 1
    min_impurity_decrease = 1E-5 # impurity improvement needed to split
    ccp_alpha = 1E-3

    seed = 42

    dtc = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease, ccp_alpha=ccp_alpha, random_state=seed)
    dtc.fit(train_X, train_Y)

    dtc_p = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease, ccp_alpha=ccp_alpha, random_state=seed)
    dtc_p.fit(train_Xp, train_Y)

    print_report([dtc, dtc_p], [test_X, test_Xp], test_Y, ["dt X", "dt Xp"])

    # export_model(dtc, 'dt_x')
    # export_model(dtc_p, 'dt_xp')



    from sklearn.ensemble import RandomForestClassifier
    n_estimators = 100
    criterion = 'gini'
    max_depth = None
    min_samples_leaf = 2
    min_impurity_decrease = 1E-5
    bootstrap=True

    seed = 42

    rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, random_state=seed)
    rfc.fit(train_X, train_Y)

    rfc_p = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, random_state=seed)
    rfc_p.fit(train_Xp, train_Y)

    print_report([rfc, rfc_p], [test_X, test_Xp], test_Y, ["rf X", "rf Xp"])

    # export_model(rfc, 'rf_x')
    # export_model(rfc_p, 'rf_xp')



    from sklearn.ensemble import AdaBoostClassifier 
    from sklearn.svm import LinearSVC
    c = 1
    dual = False
    tol = 1E-4

    base_estimator = None
    n_estimators = 50
    algo = 'SAMME.R'
    learning_rate = .3

    bc = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, algorithm=algo, learning_rate=learning_rate)
    bc.fit(train_X, train_Y)

    bc_p = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, algorithm=algo, learning_rate=learning_rate)
    bc_p.fit(train_Xp, train_Y)

    print_report([bc, bc_p], [test_X, test_Xp], test_Y, ["bb X", "bb Xp"])

    # export_model(bc, 'boost_x')
    # export_model(bc_p, 'boost_xp')


if (__name__ == '__main__'):

    approach = 'single_context'

    use_case_data = {
        'youtube':
            [l[0] for l in [
            # ['CategoryLayer', 'category_id'],
            # ['ViewsLayer', 'views'],
            ['LikesLayer', 'likes'],
            # ['DislikesLayer', 'dislikes'],
            # ['CommentCountLayer', 'comment_count'],
            # ['CountryLayer', 'country_id'],  
            # ['TrendDelayLayer', 'trend_delay'],
            ]],
        'taxi':
            [l[0] for l in [
            # ['CallTypeLayer', 'call_type'],
            # ['DayTypeLayer', 'day_type'],
            ## ['TaxiIdLayer', 'taxi_id'],

            ## ['OriginCallLayer', ('call_type', 'origin_call')],
            ## ['OriginStandLayer', ('call_type', 'origin_stand')],
            ['StartLocationLayer', ('start_location_lat', 'start_location_long')],
            # ['EndLocationLayer', ('end_location_lat', 'end_location_long')],
            ]]
    }

    for use_case, layer_names in use_case_data.items():
        for layer_name in layer_names:
            print(use_case, layer_name)

            try:
                df: DataFrame = pd.read_csv(f'data/{use_case}/ml_input/single_context/{layer_name}.csv', index_col=0)
                
                df = remove_empty_community_class(df)

                # remove the new column containing abs.val. for regression
                df = df[df.columns[:-1]]

                for idx, (training, testing) in enumerate(get_k_folds(df, k=10)):
                    print(idx)

                    scaler = StandardScaler()
                    train_X = scaler.fit_transform(training)[:,:-1] # all except y
                    train_Y = training[training.columns[-1]]

                    test_X = scaler.transform(testing)[:,:-1] # all except y
                    test_Y = testing[testing.columns[-1]]

                    try:
                        train_X, train_Y = sampler.sample_median_size(train_X, train_Y, max_size=10000)
                    except Exception as ex:
                        print(f"### Failed median sampling for {layer_name}: {ex}")
                        continue # dont train with full dataset fold
                        
                    pca = PCA(n_components=8)
                    train_Xp = pca.fit_transform(train_X)
                    test_Xp = pca.transform(test_X)

                    run() # for layer for fold

                print(f"Exporting result for {layer_name}")
                export_report(layer_name)

            except Exception as e:
                print('### Exception occured:', e)
