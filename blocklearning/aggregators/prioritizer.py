
import numpy as np

class Prioritizer():
    def __init__(self, alpha):
         self.fl = (1 - alpha)**2
         self.fm = -2*alpha**2 + 2*alpha
         self.fh = alpha**2

    def __euclidian(self, weights1, weights2):
        return np.linalg.norm(np.array(weights1) - np.array(weights2))


    def select_weights(self, my_weights, weight_map):
        distance_map = {}

        for address in weight_map.keys():
            dist = self.__euclidian(my_weights, weight_map[address])

            if dist > min(self.fl, self.fh) and dist < max(self.fl, self.fh):
                distance_map[address] = dist

        return distance_map

        
