#!/usr/bin/env python

import numpy as np, os
import numpy.linalg as npl

"""
Short script to convert basis set data from BSE to a binary dictionary with entries for each atom.

"""
def sym2num(self,sym):
	"""Routine that converts atomic symbol to atomic number"""
	symbol = [
		"X","H","He",
		"Li","Be","B","C","N","O","F","Ne",
		"Na","Mg","Al","Si","P","S","Cl","Ar",
		"K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
		"Co", "Ni", "Cu", "Zn",
		"Ga", "Ge", "As", "Se", "Br", "Kr",
		"Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
		"Rh", "Pd", "Ag", "Cd",
		"In", "Sn", "Sb", "Te", "I", "Xe",
		"Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",  "Eu",
		"Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
		"Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
		"Tl","Pb","Bi","Po","At","Rn"]
	return symbol.index(str(sym))


#opens the file
with open('sto3glim.dat') as basis_file:
	basis = basis_file.read()

#splits the content into one array for each atom
basis = basis.split(r'****')
basis = basis[1:-1]

for i in range(0,len(basis)):

	basis[i] = basis[i].split('\n')
	basis[i] = basis[i] [1:-1]

	for idx, line in enumerate(basis[i]):

		if not line:
			pass

		elif(idx == 0):
			atom = sym2num(line[0])

		elif(idx > 0):
			





print(basis)



"""

basis = basis[1:]
np.asarray(basis)

print(basis)

for i in range(len(basis)):

	basis[i] = basis[i].replace('\n', '')
	basis[i] = basis[i][:-1]
	basis[i] = basis[i].split(' ')

	#basis[i].insert(2, '0')
	#basis[i].insert(len(basis[i]), '')
	np.asarray(basis[i])
	#basis[i] = np.reshape(basis[i], (-1,3))
	print(len(basis[i]))



#print(basis)
"""