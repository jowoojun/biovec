import simplejson as json

# load disprot db
with open('./disprot.json', 'r') as f:
    disprot_data = json.load(f)
f.close()
# create disprot.fasta
with open('./disprot.fasta','w') as disprot_fasta:
    # parsing disprot json data
    for disprot in disprot_data:
        name = disprot["uniprot_accession"]
        for pfam in disprot['pfam']:
            f_name = pfam['id']
        
        sequence = disprot['sequence']
        data = '>%s %s\n%s\n'%(name,f_name,sequence)
        disprot_fasta.write(data) 

disprot_fasta.close()
