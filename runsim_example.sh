echo 'Create network'
parallel 'python3 create_pkl_file.py {1} {2} {3} {4} {5} {6} {7}' ::: 0 ::: 0. ::: 1000 ::: 2019 ::: 1.0 :::  0.0 ::: 0
echo 'Run simulations'
parallel --bar --shuf -j 3 'cd PottsModel ; python3 run.py {1} {2} {3} {4} {5} {6} {7}' ::: 0 ::: 0. ::: 1000 ::: 2019 ::: 1.0 :::  0.0 ::: 0
python3 update_sim_table.py
