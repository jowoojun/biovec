from Bio import SwissProt
handle = open("uniprot_sprot.dat")

descriptions = [record.description for record in SwissProt.parse(handle)]

num = len(descriptions)

print num
