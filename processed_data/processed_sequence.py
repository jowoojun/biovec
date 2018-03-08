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
    if 87<= len(sequence):
        name = "FG_NUPS"
        data = '>%s\n%s\n'%(name,sequence)
        dataset.write(data) 

f = open("pdb_seqres.fasta" , "r")

i = 0
for line in f:
    if i%2 == 1 :
        seq=line
        if 890 <= len(seq) and len(seq) <= 910 :
            name = 'pdb'
            data = ">%s\n%s"%(name , seq)
            dataset.write(data)
            
    i+=1
f.close()
dataset.close()
