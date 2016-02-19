
from numpy import array, reshape

chart = {'ALA': 1, 'IIE' : 2, 'LEU':3, 'VAL':4, "PHE":5, "TRP":6, "TYR":7, "ASN":8, "CYS":9,"GLU":10, "MET":11,"SER":12,"THR":13, "ASP":14, "GLU":15, "ARG":18, "HIS":20, "LYS":20}

chart = {'ALA' : 1, 'GLN':2, "LEU":3, "SER":4, "ARG":5,"GLU":6, "LYS":7, "THR":8,'ASN':9,"GLY":10, 'MET':11, "TRP":12, "ASP":13, "HIS":14, "PHE":15, "TYR":16, "CYS":17, "ILE":18, "PRO":19, "VAL":20, "UNK":21}
def read_pdb(fileName):
    residues = []
    coords = []
    prevcoords = []
    with open(fileName, "r") as f:
        for line in f:
            myline = line.split() #split the lines to a list
            if myline[0] == 'ATOM' and len(myline[3]) > 2: #check only for amino acids, not the RNA molecules
                #print myline[3]
                residues.append(chart[(myline[3][-3:])]) #get the last three letters. Ignore Alpha or Beta type
                #print myline[5],
                #print myline[6],
                #print myline[7]
                if (cmp(prevcoords,[]) == 0):
                    mycoords = [0 , 0, 0]
                else:
                    mycoords = [round(float(prevcoords[0]) - float(myline[5])),
                    round(float(prevcoords[1]) - float(myline[6])),
                    round(float(prevcoords[2]) - float(myline[7]))]
                prevcoords = [myline[5], myline[6], myline[7]]
                coords.append(mycoords[0])
                coords.append(mycoords[1])
                coords.append(mycoords[2])

                #for simplicity now, we will only compute till residues number equals multiple of 9
                scale_factor = len(residues) % 9
                #print scale_factor
                #for x in range(scale_factor):
                    #residues.pop()
                    #coords.pop()
                    #print len(coords)
    myres = residues[: (len(residues)- len(residues) % 9)]
    mycoords = coords[: (len(coords) - len(coords) % 27)]

    #print len(myres)%9
    #print len(mycoords)%27
    #reduce mycoords to a flat list
    #import operator
    #mycoords = reduce(operator.add, mycoords)

    # create numpy arrays from the lists
    myres = array(myres, dtype='float32')
    mycoords = array(mycoords, dtype='float32')

    myres = myres.reshape(len(myres)/9, 9)
    #print myres
    #print myres.shape

    mycoords = mycoords.reshape(len(mycoords)/27, 27)
    #print mycoords
    #print mycoords.shape
    #print mycoords
    return myres, mycoords

if __name__ == "__main__":
    import sys
    a = read_pdb(sys.argv[1])
    print a
    print len(a[0]) % 9
