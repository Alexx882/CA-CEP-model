def get_evolution_label(old_size: int, new_size: int) -> int:
    '''Returns the evolution label as int by mapping 0..4 to {continuing, shrinking, growing, dissolving, forming}.'''
    if old_size == 0 and new_size == 0:
        return 0 # STILL EMPTY
    if old_size == new_size:
        return 0 # continuing
    if old_size == 0 and new_size > 0:
        return 4 # forming
    if old_size > 0 and new_size == 0:
        return 3 # dissolving
    if old_size > new_size:
        return 1 # shrinking
    if old_size < new_size:
        return 2 # growing


from collections import Counter
def mode(lst):
    '''Returns any of the maximal occuring values, i.e., manual impl of mode(.) in py 3.8+'''
    counts = Counter(lst)
    return max(counts.items(), key=lambda e: e[1])[0] # max for count, return key


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


from entities.repeated_training_result import RepeatedTrainingResult
# contains the k-fold classification reports for a use_case with key=(layer_name, naive_model) 
repeated_result: Dict[Tuple, RepeatedTrainingResult] = {} 

import sklearn
import random

import sklearn.metrics

def show_majority_class_prediction():
    # print("### Majority Class Prediction: ###")

    majority_class_train = mode(Y_train)
    # print(f"Training majority class = {stat.mode(Y_train)}, Test majority class = {stat.mode(Y_test)}") 
        
    pred_Y = [majority_class_train] * len(Y_test)

    key = (layer_name, 'majority')
    if key not in repeated_result:
        repeated_result[key] = RepeatedTrainingResult()
    repeated_result[key].add_classification_report(sklearn.metrics.classification_report(y_true=Y_test, y_pred=pred_Y, output_dict=True))

    
def show_random_prediction():
    print("### Random Class Prediction: ###")

    classes = list(set(Y_train))
    print(f"Classes: {classes}")

    pred_Y = random.choices(classes, k=len(Y_test))
    print(sklearn.metrics.classification_report(y_true=Y_test, y_pred=pred_Y))


def run_persistence_prediction(): # last value
    pred_Y = history_test.apply(lambda row: get_evolution_label(old_size=row['cluster_size.1'], new_size=row['cluster_size.2']), axis=1)
    
    key = (layer_name, 'persistence')
    if key not in repeated_result:
        repeated_result[key] = RepeatedTrainingResult()
    repeated_result[key].add_classification_report(sklearn.metrics.classification_report(y_true=history_test['evolution_label'], y_pred=pred_Y, output_dict=True))


from pathlib import Path
def export_report():
    fpath = f'data/{use_case}/ml_output/{approach}'
    Path(fpath).mkdir(parents=True, exist_ok=True)

    with open(f"{fpath}/results_naive.csv", 'a') as file:
        for (l_name, model), result in repeated_result.items():
            file.write(f"{l_name}, {model}, {result.get_all_metrics_as_str()}\n")



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


import pandas as pd
from pandas import DataFrame

for use_case, layer_names in use_case_data.items():
    repeated_result = {}

    for layer_name in layer_names:
        print(use_case, layer_name)

        df: DataFrame = pd.read_csv(f'data/{use_case}/ml_input/single_context/{layer_name}.csv', index_col=0)
        df['evolution_label'] = df['evolution_label'].replace(-1.0, 0)

        for fold_nr, (train, test) in enumerate(get_k_folds(df, k=10)):
           
            Y_train = train[train.columns[-1]]
            Y_test = test[test.columns[-1]]
            history_test = test[['cluster_size', 'cluster_size.1', 'cluster_size.2', 'evolution_label']] # automatically loaded with .1, .2 for unique col names

            # naive prediction
            show_majority_class_prediction()
            run_persistence_prediction()

    export_report()    
