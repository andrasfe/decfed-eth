
#!/bin/bash

contract_address="$1"
pedersen_address="$2"

rm ../../blocklearning-results/results/CURRENT/logs/report.csv
rm ../../blocklearning-results/results/CURRENT/logs/manager.log

for i in {1..50}
do
   echo "Iteration $i"
   python3 e2e_test_deployed_contract.py --contract_address $contract_address --pedersen_address $pedersen_address
done

./extract_report.sh >> ../../blocklearning-results/results/CURRENT/logs/report.csv
