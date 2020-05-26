rsync -a "$(pwd -P)" gdelabbey@dromon.ens.fr: --exclude=".*" --exclude="*.pkl" --exclude "*.png"
ssh -t gdelabbey@dromon.ens.fr 'cd PottsModel/data_analysis ; rm *.pkl'
parallel --sshloginfile nodefile_soft 'cd PottsModel ; pwd ; hostname ; cat /proc/cpuinfo | grep "name" ; python3 create_pkl_file.py 0 {1} {2}' :::  0. 0.5 1. ::: 100000
parallel --bar --shuf -j 3 --sshloginfile nodefile 'cd PottsModel ; python3 run.py {1} {2} {3}' ::: {0..199} ::: 0. 0.5 1. ::: 100000
rsync -r gdelabbey@dromon.ens.fr:PottsModel/data_analysis/ data_analysis/
python3 update_sim_table.py
