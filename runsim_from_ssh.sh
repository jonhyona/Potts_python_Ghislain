rm *.pkl
rm *.txt
~/bin/parallel --sshloginfile nodefile_soft 'cd PottsModel ; python3 create_pkl_file.py 0 {1} {2}' :::  0. 0.5 1. ::: 100000
~/bin/parallel --progress --shuf -j 3 --sshloginfile nodefile 'cd PottsModel ; python3 run.py {1} {2} {3}' ::: $(seq 1 1 200) ::: 0. 0.5 1. ::: 100000
