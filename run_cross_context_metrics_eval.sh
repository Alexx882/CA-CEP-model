#!/bin/bash 

source venv-gpu2/bin/activate

# execute the cross-context training without one metric each
for VARIABLE in size.1 variety.1 entropy.1 clst_sizes_min.1 clst_sizes_max.1 clst_sizes_avg.1 clst_sizes_sum.1 clst_popularities_min.1 clst_popularities_max.1 clst_popularities_avg.1 clst_popularities_sum.1 center_dists_min.1 center_dists_max.1 center_dists_avg.1 center_dists_sum.1 time_f1.1 time_f2.1 size.2 variety.2 entropy.2 clst_sizes_min.2 clst_sizes_max.2 clst_sizes_avg.2 clst_sizes_sum.2 clst_popularities_min.2 clst_popularities_max.2 clst_popularities_avg.2 clst_popularities_sum.2 center_dists_min.2 center_dists_max.2 center_dists_avg.2 center_dists_sum.2 time_f1.2 time_f2.2 cluster_id
do
	python3 train_cross_context_kfold.py $VARIABLE
done
