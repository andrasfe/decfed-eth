from blocklearning.aggregators import BasilAggregator
import tensorflow as tf
import unittest
import pickle
from sklearn.metrics import f1_score

submissions = [(97343748807907098624, 99642857142857138176, 100, 
    'QmdBSQrjacTAQeQ1DoT3Yz2DyJgr9DB8odGroMkx4MEjXn', 13941463177824009744775924363945039550267837549997785618729079601690940308853, 60745476210894129251440937083306136362089371153726980127651439742728162462717), (92544645071029665792, 97857142857142845440, 100, 
    'QmdNMQbSRy1396ZRh96WdG22Duqh4C2Cm3HfwhmgbjFgg4', 13117254613777649519764556490870460094501054607969640018918288065741507682481, 8025169340393864142431052643283124464424719763226950954626244878413597875243), (95089286565780635648, 99017857142857154560, 100, 
    'QmV7gnkrsoojUmP9LeCfiVD7nwEVnT8nTALE9wbXe9AcX3', 49186043320078386844138223472698218083377733226816964101651613384319217117955, 43382979128478897712614539033583593615322472359538252925604833165736303300681), (91785717010498043904, 96071428571428569088, 100, 
    'QmTQUKHvsqrSVp1Xz3t1bV1jRksQDRVfbYkSHf7Tq5yysj', 8033842591309285727357717934346284846295402432213973702288432908358143249613, 1706199990729591610993116554035041961116827540595720520927823377565322213549), (91540175676345827328, 98214285714285707264, 100, 
    'QmQWrMukiAWT2sk58xt4Na7xqRcbgjTEiqpAqrGNw9Gxed', 21711111349245905145608547719605965681715323435914913335404602226418157401875, 108380235189173991546482556629672715191859534292455285092036778680291363358)
]

my_layer = [-0.07157102,  0.21285036,  0.21127528,  0.06954922, -0.08739046, -0.06497964, -0.07436084, -0.04972843, -0.0772506,  -0.0683938 ]
data_tf = tf.data.experimental.load('./files/data_tf')
model = tf.keras.models.load_model('./files/model.h5')

mock_weight_mapping = {
    "QmdBSQrjacTAQeQ1DoT3Yz2DyJgr9DB8odGroMkx4MEjXn": "./files/weights1.pkl",
    "QmdNMQbSRy1396ZRh96WdG22Duqh4C2Cm3HfwhmgbjFgg4": "./files/weights2.pkl",
    "QmV7gnkrsoojUmP9LeCfiVD7nwEVnT8nTALE9wbXe9AcX3": "./files/weights3.pkl",
    "QmTQUKHvsqrSVp1Xz3t1bV1jRksQDRVfbYkSHf7Tq5yysj": "./files/weights4.pkl",
    "QmQWrMukiAWT2sk58xt4Na7xqRcbgjTEiqpAqrGNw9Gxed": "./files/weights5.pkl"
}

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
    def test_calc_F1(self):
        data_tf = tf.data.experimental.load('./files/data_tf')
        model = tf.keras.models.load_model('./files/model.h5')        
        basil_aggregator = BasilAggregator(weights_loader)
        f1, f1_detailed = basil_aggregator._BasilAggregator__calc_F1(data_tf, model)
        self.assertGreater(f1, 0)
    
    def test_aggregation(self):
        basil_aggregator = BasilAggregator(weights_loader)
        my_new_model = basil_aggregator.aggregate(model, submissions, data_tf)
        self.assertIsNotNone(my_new_model)

        # validate that the new model is actually working
        x, y = tuple(zip(*data_tf))
        logits = model.predict(x[1])
        detailed_f1 = f1_score(tf.argmax(logits, axis=1), tf.argmax(y[1], axis=1), average=None, labels=[0,1,2,3,4,5,6,7,8,9], zero_division=1)
        print(detailed_f1)


if __name__ == '__main__':
    unittest.main()
