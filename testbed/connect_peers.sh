output="None"

while [ "$output" == "None" ]
do
  output=$(python3 toolkit.py connect-peers `docker network ls | awk '$2 == "bflnet" {print $1}'`)
  echo $output

  if [ "$output" == "None" ]
  then
    sleep 30
  fi
done