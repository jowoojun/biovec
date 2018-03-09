import numpy as np
import simplejson as json
from Bio import SeqIO
import os
# load FG-NUPS db

directory="binary_svm"
if not os.path.exists(directory):
    os.makedirs(directory)

with open('./FG-NUPS.json', 'r') as f:
    FG_NUPS_data = json.load(f)
f.close()

if not os.path.exists("./binary_svm/dataset.fasta"):
    dataset = open("./binary_svm/dataset.fasta" , "w" )
    
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

    
# density map 
directory="density_map"
if not os.path.exists(directory):
    os.makedirs(directory)

if not os.path.exists("./density_mapdis-fg-nups.fasta"):
    with open("./density_map/dis-fg-nups.fasta", "w") as dis_fg_nups:

        with open("uniprot_sprot.fasta" , "r") as uniprot:
            uni_list = []
            for r in SeqIO.parse(uniprot, "fasta"):
                if 80 <= len(r) and len(r) <= 94:
                     uni_list.append(r.seq)
            
            uni_list = np.array(uni_list)
            uni_list = np.random.choice(uni_list , 129 , replace=False)
            for seq in uni_list:
               name = "dis-fg-nups"
               data = '>%s\n%s\n'%(name,seq)
               dis_fg_nups.write(data) 

        uniprot.close()

    dis_fg_nups.close()
        
    
if not os.path.exists("./density_map/fg-nups.fasta"):    
    with open("./density_map/fg-nups.fasta","w") as fg_nups:
        # parsing FG-NUPS json data
        for FG_NUPS in FG_NUPS_data:
            sequence = FG_NUPS["sequence"]
            if 80 <= len(sequence) and len(sequence) <= 94:
                name = "FG_NUPS"
                data = '>%s\n%s\n'%(name,sequence)
                fg_nups.write(data)

    fg_nups.close()

if not os.path.exists("./density_map/pdb1.fasta") and not os.path.exists("./density_map/pdb2.fasta"):   
    with open("pdb_seqres.fasta" , "r") as pdb:
        i = 0
        pdb_seq = []
        for line in pdb:
            if i%2 == 1 :
                seq=line
                if 890 <= len(seq) and len(seq) <= 910 :
                    pdb_seq.append(seq)    
            i+=1
        
        print (len(pdb_seq))
        
    p1 = np.random.choice(pdb_seq , 267 , replace=False)
    p2 = np.random.choice(pdb_seq , 267 , replace=False)
    
    pdb.close()

    with open("./density_map/pdb1.fasta" , "w") as pdb1:
        for seq in p1:
            name = "pdb1"
            data = ">%s\n%s"%(name , seq)
            pdb1.write(data)
    pdb1.close()

    with open("./density_map/pdb2.fasta" , "w") as pdb2:
        for seq in p2:
            name = "pdb2"
            data = ">%s\n%s"%(name , seq)
            pdb2.write(data)
    pdb2.close()

if not os.path.exists("./density_map/dis-disprot.fasta"):

    with open("uniprot_sprot.fasta" , "r") as uniprot:
        uni_list = []
        for r in SeqIO.parse(uniprot, "fasta"):
            uni_list.append(r.seq)
        uni_list = np.array(uni_list)
        uni_list = np.random.choice(uni_list , 803, replace=False)
    uniprot.close()

    with open("./density_map/dis-disprot.fasta" , "w") as dis_disprot:
        for seq in uni_list:
           name = "dis-disprot"
           data = '>%s\n%s\n'%(name,seq)
           dis_disprot.write(data) 
    dis_disprot.close()

if not os.path.exists("./density_map/disprot.fasta"):
    with open("disprot.json", "r") as f:
        disprot_data = json.load(f)
    f.close()

    with open("./density_map/disprot.fasta" , "w") as disprot:
        # parsing FG-NUPS json data
        for DP in disprot_data:
            sequence = DP["sequence"]
            name = "disprot"
            data = '>%s\n%s\n'%(name,sequence)
            disprot.write(data) 
    disprot.close()
