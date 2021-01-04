#! /bin/bash

source venv/bin/activate

for layer in CallTypeLayer DayTypeLayer EndLocationLayer OriginCallLayer OriginStandLayer StartLocationLayer TaxiIdLayer
do 
    python3 train.py $layer
done