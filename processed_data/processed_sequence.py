f = open("pdb_seqres.fasta" , "r")
pdb = open("pdb.fasta" , "w" )
i = 0
len_seq = 0
for line in f:
    if i%2 == 1 and len(line) >= 230 and len(line) <= 270:
        seq=line
        name = "pdb"
        data = ">%s\n%s"%(name , seq)
        pdb.write(data)
    i+=1

f.close()
pdb.close()

