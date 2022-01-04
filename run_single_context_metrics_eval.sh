#!/bin/bash 

source venv-gpu2/bin/activate

# execute the single-context training without one metric each
for VARIABLE in size.1 sd.1 magnitude.1 scarcity.1 temp_center_distance.1 popularity.1 diversity.1 sin_t.1 cos_t.1 size.2 sd.2 magnitude.2 scarcity.2 temp_center_distance.2 popularity.2 diversity.2 sin_t.2 cos_t.2 size.3 sd.3 magnitude.3 scarcity.3 temp_center_distance.3 popularity.3 diversity.3 sin_t.3 cos_t.3
do
	python3 train_single_context_kfold.py $VARIABLE
done
