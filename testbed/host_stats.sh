#!/bin/bash

while true; do
    date=$(date +%F" "%T)
    load=$(uptime | awk '{print $(NF-2)}')
    cpu=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')
    memory=$(free -m | awk 'NR==2{printf "%.2f%%", $3*100/$2}')
    disk=$(df -h / | awk '{print $5}' | tail -n 1)
    network=$(ifstat | awk 'NR==3{printf "%s in, %s out\n", $2, $4}')
    echo "$date, Load Avg: $load, CPU Usage: $cpu, Memory Usage: $memory, Disk Usage: $disk, Network Usage: $network"
    sleep 5
done
