import os 
import sys

if len(sys.argv) < 2:
    print("Specify dataset name!")
    exit()

dataset = sys.argv[1] 
for folder in ['cluster_metrics', 'layer_metrics', 'raw', 'ml_input/single_context', 'ml_input/cross_context']:
    os.makedirs(os.path.join(f'data/{dataset}/',folder))

print("Created!")