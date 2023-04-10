
docker stop $(docker ps --filter status=running -q)
docker rm $(docker ps --filter status=exited -q) && \
docker system prune -a -f

python3 toolkit.py build-images

docker network create \
  --driver=bridge \
  --subnet=172.16.254.0/20 \
  bflnet

CONSENSUS=poa MINERS=10 docker-compose -f blockchain.yml -p bfl up








