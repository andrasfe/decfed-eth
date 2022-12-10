
from math import ceil
import numpy as np
from itertools import islice

class Prioritizer():
    def __init__(self, my_weights, alpha):
        self.fl, self.fm, self.fh = self.get_parms(alpha=alpha)
        self.my_weights = my_weights

    def get_parms(self, alpha):
        return ((1 - alpha)**2, -2*alpha**2 + 2*alpha, alpha**2)

    def __euclidian(self, weights1, weights2):
        return np.linalg.norm(np.array(weights1) - np.array(weights2))

    def get_sorted_distance_map(self, weight_map):
        distance_map = {}

        for address in weight_map.keys():
            dist = self.__euclidian(self.my_weights, weight_map[address])
            distance_map[address] = dist

        sorted_distance_map = {k: v for k, v in sorted(distance_map.items(), key=lambda item: item[1])}
        return sorted_distance_map

    def group_weights_by_dist(self, distance_map):
        total = len(distance_map)
        cnt = total//3
        keys = list(distance_map.keys())
        values =  list(distance_map.values())
        lo = {keys[k]:values[k] for k in range(0, cnt)}
        mid = {keys[k]:values[k] for k in range(cnt, 2*cnt + total % 3)}
        hi = {keys[k]:values[k] for k in range(2*cnt + total % 3, total)}

        return (lo, mid, hi)

    def select_qualified(self, weight_group, perc):
        cnt = ceil(perc*len(weight_group))
        keys = list(weight_group.keys())
        values =  list(weight_group.values())
        return {keys[k]:values[k] for k in range(0, cnt)}

    def get_prioritized_weights(self, weights):
        sorted_map = self.get_sorted_distance_map(weights)
        weight_groups = self.group_weights_by_dist(sorted_map)
        qual_lo = self.select_qualified(weight_groups[0], self.fl)
        qual_mid = self.select_qualified(weight_groups[1], self.fm)
        qual_hi = self.select_qualified(weight_groups[2], self.fh)
        qual_lo.update(qual_mid)
        qual_lo.update(qual_hi)
        return qual_lo


        
