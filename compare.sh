#!/usr/bin/env bash

FILE=./dataset.csv

if ! [[ -f "$FILE" ]]; then
    echo "$FILE not exist."
    exit 1
fi

awk 'FNR>1' ${FILE} | shuf > ./data/train.csv
head -n 1   ${FILE} > ./tmp.csv
head -n 1   ${FILE} > ./data/test.csv
cat  ./data/train.csv >> ./tmp.csv
tail -n 50 tmp.csv >> ./data/test.csv


for step in {51..501..50}
do
	echo "step=${step}"
	tail -n ${step} tmp.csv > ./data/train.csv
	algorithm=lr   python3 rf.py
	algorithm=rf   python3 rf.py
	algorithm=tree python3 rf.py
	algorithm=ada  python3 rf.py
	algorithm=gbdt python3 rf.py
done

rm ./data/train.csv
rm ./data/test.csv
rm tmp.csv