#!/bin/bash
# Usage parse_log.sh caffe.log
# It creates two files one caffe.log.test that contains the loss and test accuracy of the test and
# another one caffe.log.loss that contains the loss computed during the training

# revised by Hongyang Li, 2015.1.26
# mainly for googlenet

function pause(){
   read -p "$*"
}
# # breakpoint
# pause 'Press [Enter] key to continue...'


# get the dirname of the script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

if [ "$#" -lt 1 ]
then
echo "Usage parse_log.sh /path/to/your.log"
exit
fi


### ============== TEST ==============
# aux.txt: extracting all test results
LOG=`basename $1`
grep -B 1 'Test ' $1 > aux.txt

# NOTE: change the following based on your test net output format
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep 'Test net output #7:' aux.txt | awk '{print $11}' > aux1.txt # top1 of loss3
grep 'Test net output #0:' aux.txt | awk '{print $15}' > aux2.txt # loss1 (weighted)
grep 'Test net output #3:' aux.txt | awk '{print $15}' > aux5.txt # loss2 (weighted)
grep 'Test net output #6:' aux.txt | awk '{print $15}' > aux6.txt # loss3 (weighted)
grep 'Test net output #8:' aux.txt | awk '{print $11}' > top5.txt # top5 of loss3

# Extracting elapsed seconds
# For extraction of time since this line contains the start time
grep '] Solving ' $1 > aux3.txt
grep 'Testing net' $1 >> aux3.txt
$DIR/extract_seconds.py aux3.txt aux4.txt # aux4.txt records time in second unit


paste aux2.txt aux5.txt aux6.txt | column -t > temp1.txt
awk '{printf "%.2f\n", $1+$2+$3 }' temp1.txt > aux7.txt		# total loss

# Generating
echo 'Iters Seconds Top1 Top5 TestLoss1 TestLoss2 TestLoss3 TestLossTotal'> $LOG.test
paste aux0.txt aux4.txt 	aux1.txt top5.txt 	aux2.txt aux5.txt aux6.txt 	aux7.txt | column -t >> $LOG.test
rm aux.txt aux0.txt aux1.txt aux2.txt aux3.txt aux4.txt aux5.txt aux6.txt aux7.txt temp1.txt top5.txt


### ============== TRAIN ==============
# For extraction of time since this line constains the start time
grep '] Solving ' $1 > aux.txt
grep ', loss = ' $1 >> aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep ', loss = ' $1 | awk '{print $9}' > aux1.txt
grep ', lr = ' $1 | awk '{print $9}' > aux2.txt

# # breakpoint
# pause 'Press [Enter] key to continue...'

# Extracting elpased seconds
$DIR/extract_seconds.py aux.txt aux3.txt

# # breakpoint
# pause 'Press [Enter] key to continue...'

# Generating
echo 'Iters Seconds TrainLossTotal LearningRate'> $LOG.train
paste aux0.txt aux3.txt aux1.txt aux2.txt | column -t >> $LOG.train
rm aux.txt aux0.txt aux1.txt aux2.txt  aux3.txt
