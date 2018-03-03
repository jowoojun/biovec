from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
structure = PDBParser().get_structure('2BEG', 'PDB/2BEG.pdb')
ppb=PPBuilder()
for pp in ppb.build_peptides(structure):
    print(pp.get_sequence())
