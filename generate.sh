#!/usr/bin/env bash

set -e

SAMPLES=${1:-30000}
FQS=${2:-10000}
CLASSES=${3:-3}
CORPUS_DIR=${CORPUS_DIR:-./corpus}
echo "Writing to ${CORPUS_DIR}. To change this, set the CORPUS_DIR environment variable."

SRC=${CORPUS_DIR}/sources.txt
DST=${CORPUS_DIR}/sources.txt

mkdir -p ${CORPUS_DIR}

# Generate data
./generate.py ${SAMPLES} ${FQS} ${CLASSES} ./fq.txt ${CORPUS_DIR}

# Shuffle
mkfifo rnd1 rnd2
tee rnd1 rnd2 < /dev/urandom > /dev/null &
shuf --random-source=rnd1 ${CORPUS_DIR}/sources.txt > ${CORPUS_DIR}/sources.txt.shuf &
shuf --random-source=rnd2 ${CORPUS_DIR}/targets.txt > ${CORPUS_DIR}/targets.txt.shuf &
wait

rm -f rnd1 rnd2
mv ${CORPUS_DIR}/sources.txt.shuf ${CORPUS_DIR}/sources.txt
mv ${CORPUS_DIR}/targets.txt.shuf ${CORPUS_DIR}/targets.txt
