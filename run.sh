if [[ $# -ne 4 ]]
then
echo "Enter Correct number of arguments"
exit 1
fi
QNO=$1
MNO=$2
IFN=$3
OFN=$4
if [ -e OFN ]
then
	rm $4
fi
if [ $QNO -eq 1 ]
then
	if [ $MNO -eq 1 ]
	then 
		python3 p11.py $IFN $OFN
	fi
	if [ $MNO -eq 2 ]
	then
		python3 p12.py $IFN $OFN
	fi
	if [ $MNO -eq 3 ]
	then
		python3 p13.py $IFN $OFN
	fi
fi
if [ $QNO -eq 2 ]
then
	if [ $MNO -eq 1 ]
	then
		python3 p21.py $IFN $OFN
	fi
	if [ $MNO -eq 2 ]
	then
		python3 p22.py $IFN test_out.txt
		svm-scale -l 0 -u 1 test_out.txt >test_out_scale
		svm-predict test_out_scale p22.model $OFN
	fi
	if [ $MNO -eq 3 ]
	then
		python3 p23.py $IFN best_out.txt
		svm-scale -l 0 -u 1 best_out.txt >best_out_scale
		svm-predict best_out_scale p23.model $OFN
	fi
fi