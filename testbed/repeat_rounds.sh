#!/bin/bash

for i in {1..10}
do
  python3 start_round.py \
  --contract 0x41880B93713A0037357FF16C97F80b7561ada1af \
  --abi ../build/contracts/Different.json \
  --rounds 50

  sleep 3600
done