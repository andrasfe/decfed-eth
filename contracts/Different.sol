// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

contract Different {
    struct Update {
        // the next 4 elements are onyl relevant starting round 2
        uint256 trainingAccuracy;
        uint256 testingAccuracy;
        uint256 trainingDataPoints;
        string weights;
        // Pedersen ccommitments only relevant in first round
        uint256 firstCommit;
        uint256 secondCommit;
    }

    enum RoundPhase {
        Stopped,
        WaitingForUpdates,
        WaitingForScores,
        WaitingForAggregations,
        WaitingForTermination,
        WaitingForBackpropagation,
        WaitingForFirstUpdate,
        WaitingForProofPresentment
    }

    // Initialization Details
    address public owner;
    uint public startBlock;
    uint public maxDurationPerCycle;
    string public model; // IPFS CID for model encoded as h5.
    RoundPhase afterUpdate; // Which phase is executed after WaitingForUpdates.

    // Registration Details
    address[] public aggregators;
    mapping(address => bool) public registeredAggregators;
    address[] public trainers;
    mapping(address => bool) public registeredTrainers;

    // Round Details
    uint256 public round = 0;
    RoundPhase public roundPhase = RoundPhase.Stopped;
    mapping(uint256 => string) public weights; // Round => Weights ID
    mapping(uint256 => address[]) public selectedTrainers; // Round => Trainers for the round
    mapping(uint256 => address[]) public selectedAggregators; // Round => Aggregators for the round

    // Updates Details
    mapping(uint256 => uint256) updatesCount; // Round => Submited Updates
    mapping(uint256 => mapping(address => bool)) updatesSubmitted; // Round => Address => Bool
    mapping(uint256 => mapping(address => Update)) public updates; // Round => Address => Update

    // Aggregations Details
    mapping(uint256 => uint256) aggregationsCount; // Round => Submited Aggregations
    mapping(uint256 => mapping(address => bool)) aggregationsSubmitted; // Round => Address => Bool
    mapping(uint256 => mapping(address => string)) public aggregations; // Round => Address => Weights ID
    mapping(uint256 => mapping(string => uint256)) aggregationsResultsCount; // Round => Weights ID => Count

    constructor(string memory _model, string memory _weights) {
        owner = msg.sender;
        model = _model;
        weights[0] = _weights;
    }

    function startRound(
        address[] memory roundTrainers,
        address[] memory roundAggregators,
        uint timeDelta
    ) public {
        require(msg.sender == owner, "NOWN");
        require(roundPhase == RoundPhase.Stopped, "NS");
        require(roundTrainers.length > 0, "TR");
        require(roundAggregators.length > 0, "VR");
        require(
            aggregators.length > 0 && trainers.length > 0,
            "NO_REGISTRATIONS"
        );

        round++;
        startBlock = block.number;
        maxDurationPerCycle = timeDelta;
        selectedTrainers[round] = roundTrainers;
        selectedAggregators[round] = roundAggregators;
        roundPhase = RoundPhase.WaitingForFirstUpdate;
    }

    function registerAggregator() public {
        if (registeredAggregators[msg.sender] == false) {
            aggregators.push(msg.sender);
            registeredAggregators[msg.sender] = true;
        }
    }

    function registerTrainer() public {
        if (registeredTrainers[msg.sender] == false) {
            trainers.push(msg.sender);
            registeredTrainers[msg.sender] = true;
        }
    }

    function getTrainers() public view returns (address[] memory) {
        return trainers;
    }

    function getAggregators() public view returns (address[] memory) {
        return aggregators;
    }

    function isInAddressArray(address[] memory arr, address look)
        internal
        pure
        returns (bool)
    {
        bool found = false;
        for (uint256 i = 0; i < arr.length; i++) {
            if (arr[i] == look) {
                found = true;
                break;
            }
        }
        return found;
    }

    function isSelectedAggregator() internal view returns (bool) {
        return isInAddressArray(selectedAggregators[round], msg.sender);
    }

    function isSelectedTrainer() internal view returns (bool) {
        return isInAddressArray(selectedTrainers[round], msg.sender);
    }

    function getRoundForTraining()
        public
        view
        virtual
        returns (uint256, string memory)
    {
        // require(roundPhase == RoundPhase.WaitingForUpdates, "NWFS");
        require(isSelectedTrainer(), "TNP");
        return (round, weights[round - 1]);
    }

    function submitFirstUpdate(Update memory submission) public virtual {
        require(roundPhase == RoundPhase.WaitingForFirstUpdate, "NWFFS");
        require(updatesSubmitted[round][msg.sender] == false, "AS");
        require(isSelectedTrainer(), "TNP");

        updates[round][msg.sender] = submission;
        updatesSubmitted[round][msg.sender] = true;
        updatesCount[round]++;

        if (updatesCount[round] == selectedTrainers[round].length) {
            roundPhase = RoundPhase.WaitingForProofPresentment;
        }
    }

    function validatePedersen(
        uint256 r,
        uint256 v,
        string memory hiddenWeights
    ) public {
        require(roundPhase == RoundPhase.WaitingForProofPresentment, "NWFPP");
        require(round == 1 && updatesSubmitted[round][msg.sender] == true, "AS");

        //   PedersenContract pedersen = PedersenContract(pedersenAddr);
        //   submissions[addr].validated = pedersen.verify(r, v, submissions[addr].firstCommit, submissions[addr].secondCommit);

        if (block.number - startBlock > maxDurationPerCycle  || updatesCount[round] == selectedTrainers[round].length) {
            roundPhase = RoundPhase.WaitingForUpdates;
        }

    }

    function submitUpdate(Update memory submission) public virtual {
        require(roundPhase == RoundPhase.WaitingForUpdates, "NWFS");
        require(updatesSubmitted[round][msg.sender] == false, "AS");
        require(isSelectedTrainer(), "TNP");

        updates[round][msg.sender] = submission;
        updatesSubmitted[round][msg.sender] = true;
        updatesCount[round]++;

        if (updatesCount[round] == selectedTrainers[round].length) {
            roundPhase = afterUpdate;
        }
    }

    function getUpdatesForPriorRound()
        public
        view
        returns (
            uint256,
            address[] memory,
            Update[] memory
        )
    {
        require(round > 1, "NGT1");

        Update[] memory roundUpdates = new Update[](
            selectedTrainers[round - 1].length
        );
        address[] memory roundTrainers = new address[](
            selectedTrainers[round - 1].length
        );
        for (uint256 i = 0; i < selectedTrainers[round - 1].length; i++) {
            address trainer = selectedTrainers[round - 1][i];
            roundTrainers[i] = trainer;
            roundUpdates[i] = updates[round - 1][trainer];
        }
        return (round, roundTrainers, roundUpdates);
    }

    function getUpdatesForAggregation()
        public
        view
        returns (
            uint256,
            address[] memory,
            Update[] memory
        )
    {
        require(roundPhase == RoundPhase.WaitingForAggregations, "NWFA");
        require(isSelectedAggregator() == true, "CSNS");

        Update[] memory roundUpdates = new Update[](
            selectedTrainers[round].length
        );
        address[] memory roundTrainers = new address[](
            selectedTrainers[round].length
        );
        for (uint256 i = 0; i < selectedTrainers[round].length; i++) {
            address trainer = selectedTrainers[round][i];
            roundTrainers[i] = trainer;
            roundUpdates[i] = updates[round][trainer];
        }
        return (round, roundTrainers, roundUpdates);
    }

    function _submitAggregation(string memory aggregation) internal virtual {
        require(roundPhase == RoundPhase.WaitingForAggregations, "NWFA");
        require(aggregationsSubmitted[round][msg.sender] == false, "AS");
        require(isSelectedAggregator() == true, "CSNS");

        aggregations[round][msg.sender] = aggregation;
        aggregationsSubmitted[round][msg.sender] = true;
        aggregationsCount[round]++;
        aggregationsResultsCount[round][aggregation]++;

        if (aggregationsCount[round] == selectedAggregators[round].length) {
            roundPhase = RoundPhase.WaitingForTermination;
        }
    }

    function submitAggregation(string memory _weights) public virtual {
        _submitAggregation(_weights);
    }

    function terminateRound() public {
        require(roundPhase == RoundPhase.WaitingForTermination, "NWFT");

        uint256 minQuorum = (selectedAggregators[round].length * 50) / 100 + 1;
        uint256 count;
        string memory roundWeights;

        for (uint256 i = 0; i < selectedAggregators[round].length; i++) {
            address aggregator = selectedAggregators[round][i];
            string memory w = aggregations[round][aggregator];
            uint256 c = aggregationsResultsCount[round][w];
            if (c >= minQuorum) {
                if (c > count) {
                    roundWeights = w;
                    count = c;
                }
            }
        }

        require(count != 0, "CNT");
        weights[round] = roundWeights;
        roundPhase = RoundPhase.Stopped;
    }
}
