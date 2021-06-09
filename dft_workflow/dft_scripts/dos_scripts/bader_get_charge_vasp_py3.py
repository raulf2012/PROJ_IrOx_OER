#!/usr/bin/env python

# #!/usr/common/software/python/3.7-anaconda-2019.07/bin/python

"""TEMP.

Author(s): Raul A. Flores
"""


#| - Import Modules

import numpy as np
from ase import io
import os
import sys
import glob
import subprocess
from ase.io import write
import os.path

#__|

# prints out excess charge associated with each atom

def get_bader_charges(traj):
    #| - get_bader_charges
    if not (os.path.exists('CHGCAR')):
        quit('ERROR no CHGCAR present')
    if(os.path.exists('AECCAR0')):
        subprocess.call('/project/projectdirs/m2997/bin/chgsum.pl AECCAR0 AECCAR2', shell=True)
        subprocess.call('bader CHGCAR -ref CHGCAR_sum', shell=True)
    else:
        subprocess.call('bader CHGCAR', shell=True)

    outfilename = 'bader_charges.txt'
    outfile=open(outfilename, 'w+')
    file = open("ACF.dat","r")
    lines = file.readlines() # This is what you forgot
    file.close()
    for j in [1, 0, -4,-3, -2, -1]:
        del lines[j]


    newlines = []
    for line in lines:
        newline = map(float,line.split())
        newlines.append(list(newline))

    newlines = np.array(newlines)
    #print ((newlines[:,4]))
    charge = newlines[:,4]

    #atoms=io.read('OUTCAR')
    atoms=io.read(traj)
    write('qn.xyz',atoms)

    # get xyz file
    filelist = glob.glob('*.xyz')
    xyzfile = filelist[0]
    file = open(xyzfile,"r")
    lines = file.readlines() # This is what you forgot
    file.close()
    for j in [1, 0]:
        del lines[j]


    del newlines
    newlines = []
    for line in lines:
        newline =line.split() # map(float,line.split())
        newlines.append(newline)

    newarray = np.array(newlines)
    name = newarray[:,0]

    # charge on slab
    chargedict = {
    'Pt':10,
    'Ce':12,
    'Sm':11,
    'Ti':4,
    'V':5,
    'Cr':6,
    'Mn':7,
    'Fe':8,
    'Co':9,
    'Ni':10,
    'Cu':11,
    'Zn':12,
    'Ga':3,
    'Ge':4,
    'Zr' :  12.000 ,
    'Nb' :  11.000 ,
    'Mo':14,
    'Tc' :   7.000 ,
    'Ru' :   8.000 ,
    'Rh' :   9.000 ,
    'Pd' :  10.000 ,
    'Ag' :  11.000 ,
    'Cd' :  12.000 ,
    'In' :   3.000 ,
    'Sn' :   4.000 ,
    'Ir':9,
    'Al':3,
    'Au':11,
    'La':11,
    'S':6,
    'O':6,
    'N':5,
    'C':4,
    'P':5,
    'Na':1,
    'K':7,
    'Li':1,
    'Cl':7,
    'Y':11,
    'Bi':5,
    'H':1}


    write_charge=[]
    for i in range(0,len(charge)):
        name_i = name[i]
        index = i
        charge_i=charge[i]
        netcharge=-(charge_i-chargedict[name_i])
        netcharge_round=round(netcharge,2)
        print (netcharge_round)
        write_charge.append(netcharge_round)
        print ('index: '+str(index)+' name: '+name_i+' charge: '+str(netcharge), file=outfile)
    outfile.close()

    outputfile=open(outfilename, "r")
    printout=outputfile.readlines()
    for line in printout:
        print (line)
    return write_charge
    #__|


if __name__ == '__main__':
    print('Argument List:', str(sys.argv))
    traj = 'OUTCAR'
    Len = len(sys.argv)
    if Len > 1:
        for i in range(1, Len):
            if sys.argv[i] == "-t":
                traj = sys.argv[i+1]

    atoms_charge=get_bader_charges(traj)
    write_charge_traj=True
    if write_charge_traj:
        atoms=io.read(traj)
        #atoms_charge.set_initial_magnetic_moments(write_charge)
        atoms.set_initial_charges(atoms_charge)
        io.write('bader_charge.json',atoms)

#print ('DONE with BADER')
