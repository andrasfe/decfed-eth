import blocklearning.aggregators as agg
import unittest
import pickle
import tensorflow as tf
from blocklearning.models.simple_model import SimpleMLP
from sklearn.metrics import accuracy_score


submissions = [(100000000000000000000, 100000000000000000000, 100, 'QmTEmmaxV5c8YFeWqJC9YxAaVPavCSA8rnErCqtx9hn8js', 0, 0), 
                (99715584516525260800, 90625000000000229376, 100, 'QmaeT2zwgMzcZJ2ifBt9ioM3KfoGrnDqf9gAqh2MBFm4s5', 0, 0), 
                (98738360404968259584, 81621621621621768192, 100, 'QmcbDYPzskvRrqYH9ir841U4rS2zH45J33uQSbj7gKXjQ9', 0, 0), 
                (98099452257156366336, 86321243523316072448, 100, 'Qma8aP26X7kMo69CNfhvpUypquH5uNrYjYLGVryy6DV5rx', 0, 0), 
                (98695343732833861632, 86197183098591674368, 100, 'QmR6kpEU67qQ4R7nxeaDsizdV37yv84nmtQSUnm6JfGuZJ', 0, 0), 
                (99754101037979123712, 93008130081300742144, 100, 'QmYhpYYbUca3UGv7h5vUmjb4QTm1DVrKgeeYC9AKHRoch6', 0, 0), 
                (100000000000000000000, 100000000000000000000, 100, 'QmPp67T4sbg9ANLRgJTm4b6sYSTzVxsJHJbYYayGkMAj6U', 0, 0), 
                (99958115816116338688, 94476987447698817024, 100, 'QmZZMASpMxn3muBoeAj1SS8NUStJTra6WjfGjyWjPUPk3d', 0, 0), 
                (99923312664031985664, 96946564885496152064, 100, 'QmUxWpmh4m8qEN2kt8M3YvgKqHL5oqookyXaUm9wZjTmUx', 0, 0), 
                (99643492698669432832, 92355555555555639296, 100, 'QmSsRsMGP29KbBjfU7NpVQQGTHG3r9GeoH3c9S4ZE7yXAh', 0, 0)
]

mock_weight_mapping = {}

for i in range(len(submissions)):
    mock_weight_mapping[submissions[i][3]] = './files/weights_cifar_{}.pkl'.format(i)

class MockLoader():
    def __init__(self, mock_weight_mapping) -> None:
         self.mock_weight_mapping = mock_weight_mapping

    def load(self, cid):
            weights_path = self.mock_weight_mapping[cid]
            with open(weights_path, 'rb') as fp:
                weights = pickle.load(fp)
                return weights

weights_loader = MockLoader(mock_weight_mapping=mock_weight_mapping)

class Testing(unittest.TestCase):  

    @classmethod
    def setUpClass(self):
        self.test_ds = tf.data.experimental.load('./files/data_tf')


    def perf(self, model):
        acc = []
        loss = []
        for slice in self.test_ds:
            cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            logits = model.predict(slice[0])
            loss.append(cce(slice[1], logits).numpy())
            acc.append(accuracy_score(tf.argmax(logits, axis=1), tf.argmax(slice[1], axis=1)))
        return sum(acc)/len(acc), sum(loss)/len(loss)

    def test_local(self):
        model = SimpleMLP.build('cifar')
        for submission in submissions:
            weights = weights_loader.load(submission[3])
            model.set_weights(weights)
            acc, loss = self.perf(model)
            print('submission acc, loss: {}, {}'.format(acc,loss))


    def test_aggregation(self):
        fedavg = agg.FedAvgAggregator(weights_loader)
        new_weights = fedavg.aggregate(submissions)
        model = SimpleMLP.build('cifar')
        model.set_weights(new_weights)
        acc, loss = self.perf(model)
        print(acc, loss)

if __name__ == '__main__':
    unittest.main()

