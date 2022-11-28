const Different = artifacts.require("Different");
const PC = artifacts.require("PedersenContract");
const bootstrap = require('../contracts.json');

module.exports = async function (deployer) {
    await deployer.deploy(PC);
    await deployer.deploy(Different, PC.address, bootstrap.model, bootstrap.weights);
};
