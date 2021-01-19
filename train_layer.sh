#! /bin/bash

source venv/bin/activate

# create result folders
mkdir output/layer_metrics/5
mkdir output/layer_metrics/10
mkdir output/layer_metrics/15

# train
python3 train_layer.py CallTypeLayer DayTypeLayer

python3 train_layer.py OriginCallLayer CallTypeLayer 
python3 train_layer.py OriginStandLayer CallTypeLayer 

python3 train_layer.py TaxiIdLayer OriginCallLayer 
python3 train_layer.py StartLocationLayer OriginCallLayer 
python3 train_layer.py EndLocationLayer OriginCallLayer 
python3 train_layer.py TaxiIdLayer OriginStandLayer 
python3 train_layer.py StartLocationLayer OriginStandLayer 
python3 train_layer.py EndLocationLayer OriginStandLayer 
