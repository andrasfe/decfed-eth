
docker stop $(docker ps --filter status=running -q)
docker rm $(docker ps --filter status=exited -q) && \
docker system prune -a -f

python3 toolkit.py build-images

docker network create \
  --driver=bridge \
  --subnet=172.16.254.0/20 \
  bflnet

CONSENSUS=poa MINERS=10 docker-compose -f blockchain.yml -p bfl up &

sleep 300

python3 toolkit.py connect-peers `docker network ls | awk '$2 == "bflnet" {print $1}'`

python3 toolkit.py deploy-contract

CONTRACT=0x41880B93713A0037357FF16C97F80b7561ada1af \
PEDERSEN_CONTRACT=0x2B8d5C0B445aF5C0059766512bd33E71f0073af0 \
DATASET=mnist MINERS=5 SERVERS=1 CLIENTS=10 \
docker-compose -f ml.yml -p bfl-ml up &

./stats.sh &

./repeat_rounds.sh
  













