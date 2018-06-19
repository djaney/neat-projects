#!/usr/bin/env bash

kill $(ps -eaf | grep pole.py | awk '{print $2}') > /dev/null 2>&1

source venv/bin/activate

python3 pole.py train &

sleep 3

echo "${PID}"
GENOMES_STR=$(echo "pop" | nc -uq1 localhost 9999)
echo "GENOMES: ${GENOMES_STR}"
IFS=' ' read -r -a GENOME_ARRAY <<< "${GENOMES_STR}"

for var in "${GENOME_ARRAY[@]}"
do
  OUT=$(echo "act ${var} 1,1,1,1" | nc -uq1 localhost 9999)
  echo "#${var}: ${OUT}"
  echo "fit ${var} ${RANDOM}" | nc -uq1 localhost 9999
done

WINNER=$(echo "evo" | nc -uq1 localhost 9999)

echo "WINNER IS: #${WINNER}"

echo "DONE"

kill $(ps -eaf | grep pole.py | awk '{print $2}') > /dev/null 2>&1
