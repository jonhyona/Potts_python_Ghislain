rsync -a "$(pwd -P)" gdelabbey@dromon.ens.fr: --exclude=".*" --exclude="*.pkl" --exclude "*.png" --exclude="*.txt"
ssh -t gdelabbey@dromon.ens.fr 'cd PottsModel/data_analysis ; rm -R -- */ ; rm *.txt ; rm *.pkl'
parallel --sshloginfile nodefile_soft 'cd PottsModel ; pwd ; hostname ; cat /proc/cpuinfo | grep "name" ; python3 generate_network_patterns_in_file.py 0 {1} 10000 2021 {2} {3}' :::  0. 0.5 1. :::+ 1. 1.2 1.4 ::: 0. 0.05 0.1 0.2 0.4
parallel --bar --shuf -j 3 --sshloginfile nodefile 'cd PottsModel ; python3 run.py {1} {2} 1000 2021 {3} {4}' ::: {0..0} ::: 0. 0.5 1. :::+ 1. 1.2 1.4 ::: 0. 0.05 0.1 0.2 0.4
rsync -v -r gdelabbey@dromon.ens.fr:PottsModel/data_analysis/ data_analysis/
python3 update_sim_table.py
