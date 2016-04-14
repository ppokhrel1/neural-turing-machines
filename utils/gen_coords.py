
from numpy import array, reshape

#chart = {'ALA': '00001', 'IIE' : '00010', 'LEU':'00011', 'VAL':'00100', "PHE":'00101', "TRP": '00110', "TYR": '00111', "ASN": '01000', "CYS": '01001',"GLU": '01010', "MET": '01011',"SER": '01100',"THR": '01101', "ASP": '01110', "GLU": '01111', "ARG":'10000', "HIS":'10001', "LYS":'10010', 'UNK' : '10011'}

input_length = 9*20

from itertools import izip
def gen_training_data(fileName): #pdb filename from which we check both dssp and pdb files
    input_data = [0.0] * input_length
    for x, y in izip(read_pdb(fileName), read_dssp(fileName[:-4] + '.dssp')):
        for i in xrange(20):
            input_data.pop(i)
        for a in x:
            for num in a:
                input_data.append(float(num))
        #output_data = y
        print x
        output_data = []
        #print input_data
        for b in y:
            for mynum in b:
                output_data.append(float(mynum))
        #print output_data
        print len(input_data)
        #print len(output_data)
        #later on, we add features for storing the structure predicted to make it easier for later predictions
        yield array(input_data, dtype='float32').reshape(1, input_length), array(output_data, dtype='float32').reshape(1, 3)




chart = {'ALA' : '0'*19 + '1', 'GLN': '0'*18 + '1' + '0', "LEU": '0'*17 + '1' + '0'*2, "SER": '0'*16 + '1' + '0'*3, "ARG": '0'*15 + '1' + '0'*4,"GLU": '0'*14 + '1' + '0'*5, "LYS": '0'*13 + '1' + '0'*6, "THR": '0'*12 + '1' + '0'*7,'ASN': '0'*11 + '1' + '0'*8,"GLY": '0'*10 + '1' + '0'*9, 'MET': '0'*9 + '1' + '0'*10, "TRP": '0'*8 + '1' + '0'*11, "ASP":'0'*7 + '1' + '0'*12, "HIS": '0'*6 + '1' + '0'*13, "PHE": '0'*5 + '1' + '0'*14, "TYR":'0'*4 + '1' + '0'*15, "CYS": '0'*3 + '1' + '0'*16, "ILE": '0'*2 + '1' + '0'*17, "PRO": '0' + '1' + '0'*18, "VAL": '1' + '0'*19, "UNK": '0'*20}
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

                #for simplicity now, we will only compute till residues number equals multiple of 9
                #scale_factor = len(residues) % 9
                #print scale_factor
                #for x in range(scale_factor):
                    #residues.pop()
                    #coords.pop()
                    #print len(coords)
    returnVal = residues
    return returnVal
    #return myresval, mycoordsval


CODES = {'0': '000', 'H' : '001', 'B' : '010', 'E' : '011', 'G': '100', 'I' : '101', 'T': '110', 'S': '111'}

def read_dssp(fileName):
    residues = []
    structures = []
    with open(fileName, "r") as f:
        start = False
        for line in f:
            myline = line.split() #split the lines to a list
            if myline[1] == 'RESIDUE' and myline[2] == 'AA':
                start = True
            #if the above line finds start of structures from dssp file
            elif start == True:
                #check if fourth index is structure
                #structures if structre present else give '0'
                if myline[4].isalpha() and myline[4] != 'X':
                    structures.append(myline[4])
                else:
                    structures.append('0')
    #TODO:Now,compare it with dict and give values
    temp = []
    for structure in structures:
        temp.append(CODES[structure])
    #print temp
    return temp

if __name__ == "__main__":
    import sys
    a = read_pdb(sys.argv[1])
    print a
    print len(a[0]) % 9
