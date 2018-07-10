#!/bin/sh
#example for running pressed-BKZ
for i in $(seq 0 0)
do
    nohup ./latticegen -randseed $i r 150 1500 | ./fplll -a bkz -b 30 -bkzmaxloops 10 -bkzdumpgso test_bkzdumgso -s ../strategies/default.json -v -bkzitertours 0 > test_output 2>&1 &
done
