
#!/bin/bash

contract_address="$1"
pedersen_address="$2"

for i in {1..10}
do
   echo "Iteration $i"
   python3 e2e_test_deployed_contract.py --contract_address $contract_address --pedersen_address $pedersen_address
done
