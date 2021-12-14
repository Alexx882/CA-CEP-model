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
# contains the k-fold classification reports for a use_case with key=(layer_name, reference_layer_name, naive_model) 
repeated_result: Dict[Tuple, RepeatedTrainingResult] = {} 


import sklearn
import statistics as stat
import random

import sklearn.metrics

def show_majority_class_prediction():
    # print("### Majority Class Prediction: ###")

    majority_class_train = mode(Y_train)
    majority_class_test = mode(Y_test)
    # print(f"Training majority class = {majority_class_train}, Test majority class = {majority_class_test}") 
    
    pred_Y = [majority_class_train] * len(Y_test) 

    key = (layer_name, reference_layer_name, 'majority')
    if key not in repeated_result:
        repeated_result[key] = RepeatedTrainingResult()
    repeated_result[key].add_classification_report(sklearn.metrics.classification_report(y_true=Y_test, y_pred=pred_Y, output_dict=True))

    
def show_random_prediction():
    print("### Random Class Prediction: ###")

    classes = list(set(Y_train))
    print(f"Classes: {classes}")

    pred_Y = random.choices(classes, k=len(Y_test))
    print(sklearn.metrics.classification_report(y_true=Y_test, y_pred=pred_Y))


def run_persistence_prediction():
    pred_Y = history_test.apply(lambda row: get_evolution_label(old_size=row['prev_cluster_size'], new_size=row['current_cluster_size']), axis=1)

    key = (layer_name, reference_layer_name, 'persistence')
    if key not in repeated_result:
        repeated_result[key] = RepeatedTrainingResult()
    repeated_result[key].add_classification_report(sklearn.metrics.classification_report(y_true=history_test['evolution_label'], y_pred=pred_Y, output_dict=True))


from pathlib import Path
def export_report():
    fpath = f'data/{use_case}/ml_output/{approach}'
    Path(fpath).mkdir(parents=True, exist_ok=True)

    with open(f"{fpath}/results_naive.csv", 'a') as file:
        for (l_name, rl_name, model), result in repeated_result.items():
            file.write(f"{l_name}, {rl_name}, {model}, {result.get_all_metrics_as_str()}\n")



approach = 'cross_context'
datasets = {
    'youtube':[
        # ('CategoryLayer', 'CountryLayer'),
        # ('ViewsLayer', 'CountryLayer'),

        # ('ViewsLayer', 'CategoryLayer'),

        # ('LikesLayer', 'ViewsLayer'),
        # ('DislikesLayer', 'ViewsLayer'),
        # ('CommentCountLayer', 'ViewsLayer'),
        ('TrendDelayLayer', 'ViewsLayer'),
        ],
    'taxi':[
        ('CallTypeLayer', 'DayTypeLayer'),

        # # ('OriginCallLayer', 'CallTypeLayer'),
        # # ('OriginStandLayer', 'CallTypeLayer'),

        # # ('TaxiIdLayer', 'OriginStandLayer'),
        # # ('StartLocationLayer', 'OriginStandLayer'),
        # # ('EndLocationLayer', 'OriginStandLayer'),

        # ('StartLocationLayer', 'DayTypeLayer'),
        # ('EndLocationLayer', 'DayTypeLayer'),
        ]
    }


import pandas as pd
from pandas import DataFrame

for use_case, layer_combinations in datasets.items():
    # reset results for the next use_case
    repeated_result = {}

    for layer_name, reference_layer_name in layer_combinations:
        print(use_case, layer_name, reference_layer_name) 

        df: DataFrame = pd.read_csv(f'data/{use_case}/ml_input/cross_context/{layer_name}_{reference_layer_name}.csv', index_col=0)
        df['evolution_label'] = df['evolution_label'].replace(-1.0, 0)

        for fold_nr, (train, test) in enumerate(get_k_folds(df, k=10)):
            print(fold_nr)
            
            Y_train = train['evolution_label']
            Y_test = test['evolution_label']
            history_test = test[['prev_cluster_size', 'current_cluster_size', 'evolution_label']]

            # naive prediction
            show_majority_class_prediction()
            run_persistence_prediction()
        
    export_report()
