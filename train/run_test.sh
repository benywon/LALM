#!bin/bash

for i in {0,1,2,3,5,7}; do
nohup python3.6 test.py --data=$i > log/$i.txt &
done