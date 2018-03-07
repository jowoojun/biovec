import numpy as np
import simplejson as json

dataset = open("dataset.fasta" , "w" )

# load FG-NUPS db
with open('./FG-NUPS.json', 'r') as f:
    FG_NUPS_data = json.load(f)
f.close()

seq_len = 0

# parsing FG-NUPS json data
for FG_NUPS in FG_NUPS_data:
    sequence = FG_NUPS["sequence"]
    name = "FG_NUPS"
    data = '>%s\n%s\n'%(name,sequence)
    dataset.write(data) 
    seq_len += len(sequence)

seq_len = seq_len/2167

print seq_len

f = open("pdb_seqres.fasta" , "r")
i = 0
pdb_list = []
seq_len = 0
for line in f:
    if i%2 == 1 :
#and len(line) <= seq_len+1 and len(line) >= seq_len-1 :
        seq=line
        pdb_list.append(seq)
        seq_len += len(seq)
    i+=1
f.close()

print seq_len/len(pdb_list)
#pdb_list = np.random.choice(pdb_list , 2200)

print len(pdb_list)

for pdb_seq in pdb_list:
    name = "pdb"
    data = ">%s\n%s"%(name , pdb_seq)
    dataset.write(data)



dataset.close()
