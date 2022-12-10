from blocklearning.aggregators import Prioritizer
import unittest

my_layer = [-0.07157102,  0.21285036,  0.21127528,  0.06954922, -0.08739046, -0.06497964, -0.07436084, -0.04972843, -0.0772506,  -0.0683938 ]

layer_map = {
    '1': [ 0.23483534, 0.30847335, -0.05984734, -0.07607213, -0.09220622, -0.04460782, -0.07110225, -0.04606494, -0.09109719, -0.06231092],
    '1a': [-0.07157102,  0.21285036,  0.21127528,  0.06954922, -0.08739046, -0.06497964, -0.07436084, -0.04972843, -0.0772506,  -0.0683938 ],
    '2': [-0.07048314, 0.24734423,  0.10572382,  0.08403454, -0.0591095,  -0.08908992,-0.05695954, -0.04222631, -0.05276531, -0.06652968],
    '3': [-0.05767729, -0.0622144,  -0.07899111,  0.18980998,  0.32437977, -0.08031974, -0.05667223, -0.04275361, -0.06246652, -0.07081176],
    '4': [-0.05449647, -0.08283762, -0.08644688, -0.07563742,  0.07355765,  0.27018163, 0.12546556, -0.03898764, -0.0576025,  -0.07615358],
    '5': [-0.06347934, -0.08120234, -0.09793913, -0.08356779, -0.07387421, -0.08950191, 0.11284477,  0.3702575,   0.08826583, -0.08926054]
}

class Testing(unittest.TestCase):
    def test_prioritizer(self):
        prioritizer = Prioritizer(my_weights=my_layer, alpha=.63)
        
        for i in range(100):
            print(i, prioritizer.get_parms(i/100))

        sorted_distance_map = prioritizer.get_sorted_distance_map(layer_map)
        print(sorted_distance_map)

        out = prioritizer.group_weights_by_dist(sorted_distance_map)

        prioritized = prioritizer.get_prioritized_weights(layer_map)
        print(prioritized)


if __name__ == '__main__':
    unittest.main()
