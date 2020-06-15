import glob
import os

dyn_files = glob.glob('data_analysis/*/dynamics*.txt')
for file in dyn_files:
    path = file[:47]
    name = file[47:61]
    extension = '.txt'
    os.rename(file, path+name+extension)
    
# os.rename(r'file path\OLD file name.file type',r'file path\NEW file name.file type')
