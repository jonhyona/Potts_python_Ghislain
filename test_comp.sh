parallel 'ssh {1} ; hostname ; echo {1} >> node_working' :::: nodefile
