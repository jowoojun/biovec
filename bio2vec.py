from Bio import SwissProt

handle = open("uniprot_sprot.dat") 

#descriptions = [record.description for record in SwissProt.parse(handle)]

def build_dataset(handle):
	sequences = list()
	for record in SwissProt.parse(handle):
		sequences.append(record.sequence[0:])
		sequences.append(record.sequence[1:])
		sequences.append(record.sequence[2:])

	return sequences

sequences = build_dataset(handle)

print sequences


