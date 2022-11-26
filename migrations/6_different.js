const Different = artifacts.require("Different");
const bootstrap = require('../contracts.json');

module.exports = function (deployer) {
  deployer.deploy(Different, bootstrap.model, bootstrap.weights);
};
