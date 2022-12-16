
#!/bin/bash
cat ../../blocklearning-results/results/CURRENT/logs/manager.log  | awk '/^INFO:manager:[1-9]/' | sed 's/^\(INFO:manager:\)*//' 