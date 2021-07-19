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

import sklearn
import statistics as stat
import random

import sklearn.metrics

def show_majority_class_prediction():
    print("### Majority Class Prediction: ###")

    try:
        majority_class = stat.mode(Y_train)
        print(f"Training majority class = {stat.mode(Y_train)}, Test majority class = {stat.mode(Y_test)}") 
        
        pred_Y = len(Y_test) * [majority_class]
        print(sklearn.metrics.classification_report(y_true=Y_test, y_pred=pred_Y))
    except stat.StatisticsError:
        print(f"Label Majority Class: no unique mode; found 2 equally common values")


    
def show_random_prediction():
    print("### Random Class Prediction: ###")

    classes = list(set(Y_train))
    print(f"Classes: {classes}")

    pred_Y = random.choices(classes, k=len(Y_test))
    print(sklearn.metrics.classification_report(y_true=Y_test, y_pred=pred_Y))


use_case_data = {
    # 'youtube':
    #     [l[0] for l in [
    #     ['CategoryLayer', 'category_id'],
    #     ['ViewsLayer', 'views'],
    #     ['LikesLayer', 'likes'],
    #     ['DislikesLayer', 'dislikes'],
    #     ['CommentCountLayer', 'comment_count'],
    #     ['CountryLayer', 'country_id'],  
    #     ['TrendDelayLayer', 'trend_delay'],
    #     ]],
    'taxi':
        [l[0] for l in [
        # ['CallTypeLayer', 'call_type'],
        # ['DayTypeLayer', 'day_type'],
        # ['TaxiIdLayer', 'taxi_id'],

        # ['OriginCallLayer', ('call_type', 'origin_call')],
        # ['OriginStandLayer', ('call_type', 'origin_stand')],
        # ['StartLocationLayer', ('start_location_lat', 'start_location_long')],
        ['EndLocationLayer', ('end_location_lat', 'end_location_long')],
        ]]
}


for use_case, layer_names in use_case_data.items():
    for layer_name in layer_names:
        print(use_case, layer_name)

        import pandas as pd
        from pandas import DataFrame

        df: DataFrame = pd.read_csv(f'data/{use_case}/ml_input/single_context/{layer_name}.csv', index_col=0)
        df['evolution_label'] = df['evolution_label'].replace(-1.0, 0)

        # last value
        simple_df = df[['cluster_size', 'cluster_size.1', 'cluster_size.2', 'evolution_label']]
        simple_df['prediction'] = simple_df.apply(lambda row: get_evolution_label(old_size=row['cluster_size.1'], new_size=row['cluster_size.2']), axis=1)
        simple_df_test = simple_df.sample(frac=.2).reset_index(drop=True)
        print('### Persistence ###', sklearn.metrics.classification_report(y_true=simple_df_test['evolution_label'], y_pred=simple_df_test['prediction']))

        test_dataset_frac = .2
        dataframe = df
        training_size = int(len(dataframe) * (1-test_dataset_frac))

        train = dataframe[:training_size]
        test = dataframe[training_size:]

        Y_train = train[train.columns[-1]]
        Y_test = test[test.columns[-1]]

        # naive prediction
        show_majority_class_prediction()
        # show_random_prediction()

