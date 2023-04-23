DIR=./results/CURRENT/stats
mkdir -p $DIR

while true; do
  echo "fetching"
#   docker stats --no-stream --format "{{.ID}}, {{.Name}}, {{.CPUPerc}}, {{.MemUsage}}, {{.MemPerc}}, {{.NetIO}}, {{.BlockIO}}" > $DIR/$(date '+%s').log
  docker stats --no-stream --format "{{.ID}}, {{.Name}}, {{.CPUPerc}}, {{.MemUsage}}, {{.MemPerc}}, {{.NetIO}}, {{.BlockIO}}" >> $DIR/$(date '+%Y-%m-%d_%H').log

  sleep 10
done