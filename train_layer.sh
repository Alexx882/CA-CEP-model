#! /bin/bash

source venv/bin/activate

for depth in 5 10 15
do 
    python3 train.py CallTypeLayer DayTypeLayer $depth
    
    python3 train.py OriginCallLayer CallTypeLayer $depth
    python3 train.py OriginStandLayer CallTypeLayer $depth

    python3 train.py TaxiIdLayer OriginCallLayer $depth
    python3 train.py StartLocationLayer OriginCallLayer $depth
    python3 train.py EndLocationLayer OriginCallLayer $depth
    python3 train.py TaxiIdLayer OriginStandLayer $depth
    python3 train.py StartLocationLayer OriginStandLayer $depth
    python3 train.py EndLocationLayer OriginStandLayer $depth
done