import numpy as np
from mmd.molecule import *
from mmd.postscf import *

water = """
0 1
O    0.000000      -0.075791844    0.000000
H    0.866811829    0.601435779    0.000000
H   -0.866811829    0.601435779    0.000000
"""

# init molecule and build integrals
#mol = Molecule(geometry=water,basis='sto-3g')

HeH = """
1 1
H 0.0 0.0 0.0
He 0.0 0.0 0.9295
"""

mol = Molecule(geometry=HeH,basis='sto-3g')

# do the SCF
mol.RHF()

# do MP2
PostSCF(mol).MP2()
PostSCF(mol).CCSD()




