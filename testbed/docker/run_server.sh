#!/bin/sh

IP=$(ifconfig eth0 | grep 'inet' | awk '{print $2}' | sed 's/addr://')
INDEX=$(dig -x $IP +short | sed 's/[^0-9]*//g')
ACCOUNT=$(jq -r '.trainers' accounts.json | jq 'keys_unsorted' | jq -r "nth($((INDEX-1)))")
PASSWORD=$(jq -r '.trainers' accounts.json | jq -r ".[\"$ACCOUNT\"]")

PRVINDEX=$((INDEX % MINERS))
if [ "$PRVINDEX" -eq "0" ]; then
  PRVINDEX=$MINERS
fi
PROVIDER=$(dig bfl_geth-miner_$PRVINDEX +short)

if [ ! -e ${HOME}/.ipfs/api ]
then 
    ipfs daemon &
else
   echo "IPFS already started"
fi

seed=$(date +%s)
sleep_time=$(( $seed % 721 + 180 ))

# Sleep for the randomly generated amount of time
sleep $sleep_time

while [ ! -e ${HOME}/.ipfs/api ]
do 
   echo "Waiting for IPFS to start";
   sleep 300
done

python run_server.py \
  --provider "http://$PROVIDER:8545" \
  --abi /root/abi.json \
  --ipfs $IPFS_API \
  --account $ACCOUNT \
  --passphrase $PASSWORD \
  --contract $CONTRACT \
  --log /root/log.log \
  --train /root/dataset/owner_val.tfrecord \
  --test /root/dataset/owner_val.tfrecord


# Use the following flag for a private dataset
# --val /root/dataset/test/$((INDEX-1)).npz 

# Use the following flag for a common validation dataset:
# --val /root/dataset/owner_val.npz
