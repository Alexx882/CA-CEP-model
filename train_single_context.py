import pandas as pd
from pandas import DataFrame

import numpy as np
import collections


def split_data(dataframe, test_dataset_frac=.2, shuffle=False) -> '(training_data, test_data)':
    if shuffle:
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    training_size = int(len(dataframe) * (1-test_dataset_frac))

    train = dataframe[:training_size]
    test = dataframe[training_size:]

    y_train = train[train.columns[-1]]
    y_test = test[test.columns[-1]]
  
    print(f"\nWorking with: {len(train)} training points + {len(test)} test points ({len(test)/(len(test)+len(train))} test ratio).")
    print(f"Label Occurrences: Total = {collections.Counter(y_train.tolist() + y_test.tolist())}, \n" \
          f"\tTraining = {collections.Counter(y_train)}, \n" \
          f"\tTest = {collections.Counter(y_test)}")
    # try:
    #     print(f"Label Majority Class: Training = {stat.mode(Y_train)}, Test = {stat.mode(Y_test)}\n")
    # except stat.StatisticsError:
    #     print(f"Label Majority Class: no unique mode; found 2 equally common values")

    return train, test



def remove_empty_community_class(df):
    '''Removes evolution_label -1 from dataset indicating the community stays empty.'''
    # res = df.loc[df['evolution_label'] != -1.0]
    # res = res.reset_index(drop=True)
    # return res
    import warnings
    warnings.filterwarnings("ignore")
    df['evolution_label'] = df['evolution_label'].replace(-1.0, 0)
    warnings.filterwarnings("default")
    return df



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



from processing import DataSampler
sampler = DataSampler()



from sklearn.decomposition import PCA
pca = PCA(n_components=8)



import sklearn.metrics

def print_report(clfs: list, test_Xs: list, test_Y: 'y', titles: list):
    """
    Prints all reports.
    :param clfs: list of classifiers to evaluate
    :param test_Xs: list of test_X for the corresponding classifier at idx
    :param test_Y: true classes:param titles: list of titles for the classifiers at idx
    """
    for clf, test_X, title in zip(clfs, test_Xs, titles):
        pred_Y = clf.predict(test_X)        
        print(f"### {layer_name} {title} ###\n", sklearn.metrics.classification_report(y_true=test_Y, y_pred=pred_Y))


import pickle 
from pathlib import Path

def export_model(model, model_name):
    fpath = f'data/{use_case}/ml_output/{approach}/{layer_name}'
    Path(fpath).mkdir(parents=True, exist_ok=True)
    with open(f'{fpath}/{layer_name}_{model_name}.model', 'wb') as f:
        pickle.dump(model, f)


def run():
    # from sklearn.naive_bayes import GaussianNB
    # priors = None #np.array([19,16,16,74,74]) / (19+16+16+74+74)
    # smoothing = 0

    # clf = GaussianNB(priors=priors, var_smoothing=smoothing)
    # clf.fit(train_X, train_Y)

    # clf_p = GaussianNB(priors=priors, var_smoothing=smoothing)
    # clf_p.fit(train_Xp, train_Y)

    # print_report([clf, clf_p], [test_X, test_Xp], test_Y, ["nb X", "nb Xp"])

    # export_model(clf, 'nb_x')
    # export_model(clf_p, 'nb_xp')



    # from sklearn.svm import LinearSVC
    # c = 1
    # dual = False
    # tol = 1E-4

    # svc = LinearSVC(C=c, dual=dual, tol=tol)
    # svc.fit(train_X, train_Y)

    # svc_p = LinearSVC(C=c, dual=dual, tol=tol)
    # svc_p.fit(train_Xp, train_Y)

    # print_report([svc, svc_p], [test_X, test_Xp], test_Y, ["X", "Xp"])

    # export_model(svc, 'svc_x')
    # export_model(svc_p, 'svc_xp')



    # from sklearn.neighbors import KNeighborsClassifier
    # n_neighbors = 20
    # weights = 'uniform'
    # algo = 'auto'
    # leaf_size = 30

    # knnc = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algo, leaf_size=leaf_size)
    # knnc.fit(train_X, train_Y)

    # knnc_p = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights,  algorithm=algo, leaf_size=leaf_size)
    # knnc_p.fit(train_Xp, train_Y)

    # print_report([knnc, knnc_p], [test_X, test_Xp], test_Y, ["knn X", "knn Xp"])

    # export_model(knnc, 'knn_x')
    # export_model(knnc_p, 'knn_xp')



    # from sklearn.tree import DecisionTreeClassifier 
    # criterion = 'gini'
    # splitter = 'random'
    # max_depth = 10
    # min_samples_leaf = 1
    # min_impurity_decrease = 1E-5 # impurity improvement needed to split
    # ccp_alpha = 1E-3

    # seed = 42

    # dtc = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease, ccp_alpha=ccp_alpha, random_state=seed)
    # dtc.fit(train_X, train_Y)

    # dtc_p = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease, ccp_alpha=ccp_alpha, random_state=seed)
    # dtc_p.fit(train_Xp, train_Y)

    # print_report([dtc, dtc_p], [test_X, test_Xp], test_Y, ["dt X", "dt Xp"])

    # export_model(dtc, 'dt_x')
    # export_model(dtc_p, 'dt_xp')



    # from sklearn.ensemble import RandomForestClassifier
    # n_estimators = 100
    # criterion = 'gini'
    # max_depth = None
    # min_samples_leaf = 2
    # min_impurity_decrease = 1E-5
    # bootstrap=True

    # seed = 42

    # rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, random_state=seed)
    # rfc.fit(train_X, train_Y)

    # rfc_p = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, random_state=seed)
    # rfc_p.fit(train_Xp, train_Y)

    # print_report([rfc, rfc_p], [test_X, test_Xp], test_Y, ["rf X", "rf Xp"])

    # export_model(rfc, 'rf_x')
    # export_model(rfc_p, 'rf_xp')



    from sklearn.ensemble import AdaBoostClassifier 
    from sklearn.svm import LinearSVC
    c = 1
    dual = False
    tol = 1E-4

    base_estimator = LinearSVC(C=c, dual=dual, tol=tol) # None #SVC(kernel='linear')
    n_estimators = 50
    algo = 'SAMME'
    learning_rate = .3

    bc = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, algorithm=algo, learning_rate=learning_rate)
    bc.fit(train_X, train_Y)

    bc_p = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, algorithm=algo, learning_rate=learning_rate)
    bc_p.fit(train_Xp, train_Y)

    print_report([bc, bc_p], [test_X, test_Xp], test_Y, ["bb X", "bb Xp"])

    export_model(bc, 'boost_x')
    export_model(bc_p, 'boost_xp')



if (__name__ == '__main__'):

    approach = 'single_context'

    use_case_data = {
        'youtube':
            [l[0] for l in [
            ['CategoryLayer', 'category_id'],
            ['ViewsLayer', 'views'],
            ['LikesLayer', 'likes'],
            ['DislikesLayer', 'dislikes'],
            ['CommentCountLayer', 'comment_count'],
            ['CountryLayer', 'country_id'],  
            ['TrendDelayLayer', 'trend_delay'],
            ]],
        'taxi':
            [l[0] for l in [
            ['CallTypeLayer', 'call_type'],
            ['DayTypeLayer', 'day_type'],
            ['TaxiIdLayer', 'taxi_id'],

            ['OriginCallLayer', ('call_type', 'origin_call')],
            ['OriginStandLayer', ('call_type', 'origin_stand')],
            ['StartLocationLayer', ('start_location_lat', 'start_location_long')],
            ['EndLocationLayer', ('end_location_lat', 'end_location_long')],
            ]]
    }

    for use_case, layer_names in use_case_data.items():
        for layer_name in layer_names:
            print(use_case, layer_name)

            try:
                df: DataFrame = pd.read_csv(f'data/{use_case}/ml_input/single_context/{layer_name}.csv', index_col=0)

                training, testing = split_data(df, shuffle=False)

                training = remove_empty_community_class(training)
                testing = remove_empty_community_class(testing)

                train_X = scaler.fit_transform(training)[:,:-1] # all except y
                train_Y = training[training.columns[-1]]

                test_X = scaler.transform(testing)[:,:-1] # all except y
                test_Y = testing[testing.columns[-1]]

                try:
                    train_X, train_Y = sampler.sample_median_size(train_X, train_Y, max_size=10000)
                except Exception as ex:
                    print(f"### Failed median sampling for {layer_name}: {ex}")
                    

                train_Xp = pca.fit_transform(train_X)
                test_Xp = pca.transform(test_X)

                run()
            except Exception as e:
                print('### Exception occured:', e)

            