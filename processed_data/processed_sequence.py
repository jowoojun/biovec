import numpy as np
import simplejson as json
from Bio import SeqIO
import os
# load FG-NUPS db

directory="binary_svm"
if not os.path.exists(directory):
    os.makedirs(directory)

with open('./dis-disprot.json', 'r') as f:
    FG_NUPS_data = json.load(f)
f.close()

if not os.path.exists("./binary_svm/dataset.fasta"):
    dataset = open("./binary_svm/dataset.fasta" , "w" )

    with open ("fg-nups.fasta","r") as f:
        data_list = []
        for r in SeqIO.parse(f , "fasta"):
            data_list.append(r.seq)

        data_list = np.array(data_list)
        fg_len = len(data_list)
        for seq in data_list:
            name = "fg-nups"
            data = ">%s\n%s\n" %(name,seq)
            dataset.write(data)
    f.close()
        
    with open("disordered-pdb.fasta" , "r") as f:
        data_list = []
        for r in SeqIO.parse(f, "fasta"):
            data_list.append(r.seq)

        data_list = np.array(data_list)
# data_seq = []
#       for seq in data_list:
#           if 890 <= len(seq) and len(seq) <= 910 :
#               data_seq.append(seq)

#        data_seq = np.random.choice(np.array(data_seq) , fg_len)
        for seq in data_list:
                name = "pdb"
                data = ">%s\n%s\n" %(name,seq)
                dataset.write(data)
    f.close()
    dataset.close()

    
# density map 
directory="density_map"
if not os.path.exists(directory):
    os.makedirs(directory)

if not os.path.exists("./density_map/dis-fg-nups.fasta"):
    with open("./density_map/dis-fg-nups.fasta", "w") as dis_fg_nups:

        with open("dis-fg-nups.fasta" , "r") as f:
            data_list = []
            for r in SeqIO.parse(f, "fasta"):
                 data_list.append(r.seq)
            
            data_list = np.array(data_list)
            # uni_list = np.random.choice(uni_list , 100 , replace=False)
            for seq in data_list:
               name = "dis-fg-nups"
               data = '>%s\n%s\n'%(name,seq)
               dis_fg_nups.write(data) 

        f.close()

    dis_fg_nups.close()
        
    
if not os.path.exists("./density_map/fg-nups.fasta"):    
    with open("./density_map/fg-nups.fasta","w") as fg_nups:
        with open ("fg-nups.fasta","r") as f:
            data_list = []
            for r in SeqIO.parse(f , "fasta"):
                data_list.append(r.seq)

            data_list = np.array(data_list)
            for seq in data_list:
                name = "fg-nups"
                data = ">%s\n%s\n" %(name,seq)
                fg_nups.write(data)
        f.close()

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
    
    pdb_index = range(0,534)
    p1_index = np.random.choice(pdb_index , 267 , replace=False)
    p2_index = np.delete(pdb_index , p1_index)
    p1 = np.take(pdb_seq , p1_index)
    p2 = np.take(pdb_seq , p2_index)
    
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

    with open("dis-disprot.json", "r") as f:
        dis_disprot_data = json.load(f)
    f.close()

    with open("./density_map/dis-disprot.fasta" , "w") as dis_disprot:
        # parsing FG-NUPS json data
        for DP in dis_disprot_data:
            sequence = DP["sequence"]
            name = "dis-disprot"
            data = '>%s\n%s\n'%(name,sequence)
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
