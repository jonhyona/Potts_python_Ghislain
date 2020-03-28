with open('pattern_S7_a0-25.txt', 'r') as reader:
     # Read and print the entire file line by line
     line = reader.readline()
     cpt = 0
     while cpt < p:  # The EOF char is an empty string
         # print(line, end='')
         ksi_i_mu[:,cpt] = [int(s) for s in line.split(' ')[:N]]
         line = reader.readline()
         cpt+=1