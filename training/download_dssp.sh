#!user/bin/bash
for f in *.pdb
do
  wget ftp://ftp.cmbi.ru.nl/pub/molbio/data/dssp/${f%%.*}.dssp
done
