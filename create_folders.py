import os 
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Specify dataset name!")
    exit()

dataset = sys.argv[1] 
for folder in [
    'cluster_metrics', 
    'layer_metrics', 
    'raw', 
    'ml_input/single_context', 
    'ml_input/cross_context',
    'ml_output/single_context',
    'ml_output/cross_context',
    'ml_output/cross_context_2stage',
    ]:
    Path(os.path.join(f'data/{dataset}/',folder)).mkdir(parents=True, exist_ok=True)

print("Created!")