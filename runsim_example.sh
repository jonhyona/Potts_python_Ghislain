parallel 'python3 create_pkl_file.py 0 {1} {2}' :::  0. 0.5 1. ::: 10000
parallel --bar --shuf -j 3 'cd PottsModel ; python3 run.py {1} {2} {3}' ::: {0..10} ::: 0. 0.5 1. ::: 10000
python3 update_sim_table.py
