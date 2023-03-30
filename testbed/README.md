# Experiment Testbed 🛌

## Other prerequisites

When re-starting the initiation, just run: `
```sh
docker stop $(docker ps --filter status=running -q) && \
docker rm $(docker ps --filter status=exited -q) && \
docker system prune -a
``` 


(careful, it will remove ALL containers AND images).

```sh
sudo apt-get update && sudo apt-get -y upgrade
sudo npm install -g truffle
pip3 install rlp
pip3 install eth_account
pip3 install web3
sudo apt install fuse libfuse2
sudo modprobe fuse
sudo groupadd fuse
sudo apt-get install pcscd
sudo service pcscd start
## create ethereum/.env

```

You need a `testbed/ethereum/.env` that needs at minimum one entry. Here is mine:

`NETWORK_ID=12345`. Without this miner dockers will exit as they will give a parsing error due to null `--networkid`.

This directory contains all the necessary code to run experiments using the [`blocklearning`](../blocklearning/) library via Docker.

- [Setup](#setup)
  - [Generate Ethereum Accounts](#generate-ethereum-accounts)
  - [Update Genesis with Accounts](#update-genesis-with-accounts)
  - [Build Docker Images](#build-docker-images)
  - [Create Docker Network](#create-docker-network)
  - [Launch Blockchain Containers](#launch-blockchain-containers)
  - [Connect Peers](#connect-peers)
  - [Deploy the Contract](#deploy-the-contract)
  - [Launch ML Containers](#launch-ml-containers)
  - [Collect Statistics](#collect-statistics)
  - [Run Rounds](#run-rounds)
  - [Collect Logs](#collect-logs)
- [Contract](#contract)
- [IPFS](#ipfs)
- [How to Run Different Experiments](#how-to-run-different-experiments)
  - [Consensus Algorithms](#consensus-algorithms)
  - [Selection Mechanisms](#selection-mechanisms)
  - [Without Score Mechanisms](#without-scoring-mechanisms)
  - [With Score Mechanisms](#with-scoring-mechanisms)

## Setup

Install python packages: `python3 setup.py develop` (important, so that you can debug)

### Generate Ethereum Accounts

Generate the accounts that will be used on the network. To generate 5 accounts for miners (or validators) and 10 for trainers, run:

```bash
python3 toolkit.py generate-accounts 10 25
```

The files in `ethereum/datadir` are as follows:

- `keystore` includes the accounts information that were generated and this is necessary to start the node.
- `geth/nodekey_owner` is the node key that will be given to the "owner" RPC endpoint for our Ethereum network.
- `geth/nodekey_{i}` is the node key that will be given to the i-th Ethereum miner.
- `geth/static-nodes.json` includes the addresses that will be added as static peers to our nodes.
- `accounts.json` is the account information and password.
- `miners.json` has the public addresses generated from the private key of the miners `nodekey`. This is important for some consensus protocol genesis.

### Update Genesis with Accounts

After generating the accounts, the genesis files needs to be generated with the new accounts accounts in order to boot the network with 100 ETH in each of the accounts. To do so, run:

```bash
python3 toolkit.py update-genesis
```

### Build Docker Images

Several Docker images are used to run the Ethereum blockchain. To build then, run:

```bash
python3 toolkit.py build-images
```

### Create Docker Network

```bash
docker network create \
  --driver=bridge \
  --subnet=172.16.254.0/20 \
  bflnet
```

### Launch Blockchain Containers

For Docker compose, use:

```bash
CONSENSUS=poa MINERS=10 docker-compose -f blockchain.yml -p bfl up
```

Where `CONSENSUS` = `poa|qbft|pow`.

### Connect Peers

Unfortunately, peer discovery [doesn't work with private networks](https://ethereum.stackexchange.com/questions/121380/private-network-nodes-cant-find-peers). Not even if we use a bootstrap node. Thus, we need to connect the peers to each other manually.

```bash
python3 toolkit.py connect-peers bflnet
```

Where `<network>` is the ID of the Docker network where the containers are running. You can check that by running `docker network ls` and looking for `bflnet` or `priv-eth-net`. If no network is passed, the script will try to infer the correct network.

### Deploy the Contract

The contract deployment script fetches the account to use from `ethereum/datadir/accounts.json` and uses the first account (index 0).

```bash
python3 toolkit.py deploy-contract
```

copy the contract address. E.g., for NoScore. You will need it when you launch the ML containers (below)

### Datasets

File `trainingSet.tar` needs to be placed under `testbed/datasets/$DATASET/`, then expanded. Then run `python3 data_loader.py`.
This will populate the client and owner datasets.

### Generate .h5 file for Model

Run `python3 model_persister.py` to generate the model and weights in `testbed/datasets`.

### Launch ML Containers

```bash
CONTRACT=0x41880B93713A0037357FF16C97F80b7561ada1af \
PEDERSEN_CONTRACT=0x2B8d5C0B445aF5C0059766512bd33E71f0073af0 \
DATASET=mnist MINERS=5 SERVERS=1 CLIENTS=10 \
docker-compose -f ml.yml -p bfl-ml up
```

### Collect Statistics

Start collecting statistics before running the rounds (on the results repository):

```bash
DIR=./results/CURRENT/stats
mkdir -p $DIR

while true; do
  echo "fetching"
  docker stats --no-stream --format "{{.ID}}, {{.Name}}, {{.CPUPerc}}, {{.MemUsage}}, {{.MemPerc}}, {{.NetIO}}, {{.BlockIO}}" > $DIR/$(date '+%s').log
  sleep 2
done
```

### Run Rounds

```bash
python3 start_round.py \
  --contract 0x8058697F84Ecc48d1E130C6B22a945BaFB84758C \
  --abi ../build/contracts/Different.json \
  --rounds 50
```

### Collect Logs

To collect the logs of the trainers and validators afterwards, run:

```bash
python3 toolkit.py collect-logs
```

## Contract

Some *required* base information for the contract can be found in [../contracts.json](../contracts.json). This file includes two fields that must be filled before deploying the contract:

- `model`: the IPFS CID of the exported model (head model for Split-CNN contract) in `.h5` format.
- `bottomModel`: the IPFS CID of the exported bottom model for the Split-CNN contract in `.h5` format.
- `weights` (optional): the IPFS CID with the initial weights.

## IPFS

First, install on your desktop:
```sh
wget https://dist.ipfs.io/go-ipfs/v0.12.2/go-ipfs_v0.12.2_linux-amd64.tar.gz && \
  tar -xvzf go-ipfs_v0.12.2_linux-amd64.tar.gz && \
  cd go-ipfs && \
  sudo bash install.sh && \
  ipfs --version
```

Run this in the home directory: `ipfs init`

To add any file to IPFS, run:

```
ipfs add [-r] path
```
example: `ipfs add ./datasets/model.h5` and `ipfs add ./datasets/weights.h5`.

## How to Run Different Experiments

### Consensus Algorithms

```bash
CONSENSUS=poa|pow|qbft MINERS=10 docker compose -f blockchain.yml -p bfl up

CONTRACT=0x8C3CBC8C31e5171C19d8a26af55E0db284Ae9b4B \
  DATASET=mnist MINERS=3 SERVERS=1 CLIENTS=5 \
  SCORING="none" ABI=NoScore \
  docker compose -f ml.yml -p bfl-ml up

python3 start_round.py \
  --contract 0x8C3CBC8C31e5171C19d8a26af55E0db284Ae9b4B \
  --abi ../build/contracts/NoScore.json \
  --rounds 50
```

### Selection Mechanisms

```bash
CONSENSUS=poa MINERS=10 docker compose -f blockchain.yml -p bfl up

CONTRACT=0xA63052D2C43CD996731937AAD7986bF8f88C2511 \
  DATASET=mnist MINERS=10 SERVERS=10 CLIENTS=25 \
  SCORING="none" ABI=FirstComeFirstServed \
  docker compose -f ml.yml -p bfl-ml up

python3 start_round.py \
  --contract 0xA63052D2C43CD996731937AAD7986bF8f88C2511 \
  --trainers fcfs \
  --abi ../build/contracts/FirstComeFirstServed.json \
  --rounds 50
```

### Without Score Mechanisms

```bash
CONSENSUS=poa MINERS=10 docker compose -f blockchain.yml -p bfl up

CONTRACT=0x8C3CBC8C31e5171C19d8a26af55E0db284Ae9b4B \
  DATASET=mnist MINERS=10 SERVERS=10 CLIENTS=5|10|25|50 \
  SCORING="none" ABI=NoScore \
  docker compose -f ml.yml -p bfl-ml up

python3 start_round.py \
  --contract 0x8C3CBC8C31e5171C19d8a26af55E0db284Ae9b4B \
  --abi ../build/contracts/NoScore.json \
  --rounds 50
```

## Debugging

Launch the test images. These will stay alive until you kill all 6 of them. Good for validating scripts:

```bash
DATASET=mnist docker-compose -f ml-test.yml -p bfl-init up
```


Run a shell in selected image, as follows:

```sh
docker exec -it bfl-init_server_1 /bin/sh
ipfs add ./dataset/model.h5
ipfs add ./dataset/weights.pkl
```
