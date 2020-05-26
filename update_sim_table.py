import glob
import file_handling

param_files = glob.glob('data_analysis/parameters_*.pkl')

for ii in range(len(param_files)):
    file_handling.save_parameters(param_files[ii][25:])
