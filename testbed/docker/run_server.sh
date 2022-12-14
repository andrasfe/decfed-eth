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
nohup bash -c "ipfs daemon &" && sleep 15
python run_server.py \
  --provider "http://$PROVIDER:8545" \
  --abi /root/abi.json \
  --ipfs $IPFS_API \
  --account $ACCOUNT \
  --passphrase $PASSWORD \
  --contract $CONTRACT \
  --log /root/log.log \
  --val /root/dataset/owner_val.tfrecord \
  --scoring $SCORING


# Use the following flag for a private dataset
# --val /root/dataset/test/$((INDEX-1)).npz 

# Use the following flag for a common validation dataset:
# --val /root/dataset/owner_val.npz
