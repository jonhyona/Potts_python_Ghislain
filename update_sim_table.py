import file_handling
import os

for root, keys, _ in os.walk('data_analysis'):
    for key in keys:
        file_handling.record_parameters(key)
