rm *.pkl
rm *.txt
echo {1..199} > cue_list
~/bin/parallel --sshloginfile nodefile_soft 'cd PottsModel ; python3 create_pkl_file.py 0 {1} {2}' :::  0. 0.5 1. ::: 100000
cat cue_list | ~/bin/parallel --progress --shuf -j 3 --sshloginfile nodefile 'cd PottsModel ; python3 run.py {1} {2} {3}' :::: - ::: 0. 0.5 1. ::: 100000
