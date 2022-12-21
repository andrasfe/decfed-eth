from statistics import stdev
from  blocklearning.models import SimpleMLP
import tensorflow as tf
import math
import copy
from sklearn.metrics import f1_score
from .prioritizer import Prioritizer

class BasilAggregator():
    def __init__(self, weights_loader):
        self.weights_loader = weights_loader

    def __calc_F1(self, data_tf, model, labels = [0,1,2,3,4,5,6,7,8,9]):
        x, y = tuple(zip(*data_tf))
        logits = model.predict(x[1])

        detailed_f1 = f1_score(tf.argmax(logits, axis=1), tf.argmax(y[1], axis=1), average=None, labels=labels, zero_division=1)
        f1 = f1_score(tf.argmax(logits, axis=1), tf.argmax(y[1], axis=1), average='macro', labels=labels, zero_division=1)

        return f1, detailed_f1

    def __get_last_dense_layer_weights(self, model):
        return model.layers[-2].get_weights()[1]

    def __set_last_dense_layer_weights(self, model, last_dense_output):
        last_dense_weights = model.layers[-2].get_weights()
        last_dense_weights[1] = last_dense_output
        model.layers[-2].set_weights(last_dense_weights)

    def __weight_sigmoid(self, omega1, omega2, disc, certainty):
        return max(0, omega1/(1 + math.exp(-disc/100)) - omega2)*certainty

    def __compute_fa_fo_wg(self, my_detailed_f1, detailed_f1_list, fi=3, mu=10, omega_fa1=10, omega_fa2=4, omega_fo1=10, omega_fo2=4, kappa=10):
        disc = [[0 for _ in range(len(my_detailed_f1))] for _ in range(len(detailed_f1_list))]
        fa_wg = disc.copy()        
        fo_columns = disc.copy()
        fo_wg = [0]*len(my_detailed_f1)

        for m in range(len(detailed_f1_list)):
            best = sorted(detailed_f1_list[m], reverse=True)[:fi]
            certainty = max(sum(best)/len(best) - stdev(best), 0)
                    
            for c in range(len(my_detailed_f1)):
                if detailed_f1_list[m][c] > 0 and my_detailed_f1[c] == 0:
                    fo_columns[m][c] == 1
                    
                weighed_diff = (detailed_f1_list[m][c] - my_detailed_f1[c])*mu
                if detailed_f1_list[m][c] > my_detailed_f1[c]:
                    disc[m][c] = weighed_diff**(3 + my_detailed_f1[c])
                else:
                    disc[m][c] = -math.inf #-1*pow(weighed_diff, 4 + int(my_detailed_f1[c]))

                fa_wg[m][c] = self.__weight_sigmoid(omega_fa1, omega_fa2, disc[m][c], certainty)  

        fo_wg[m] = self.__weight_sigmoid(omega_fo1, omega_fo2, sum(disc[m]), certainty)
        return fa_wg, fo_wg, fo_columns

    def __layers_weighted_average(self, my_layer, layer_list, fa_wg_list, fo_columns_list):
        my_new_layer = copy.deepcopy(my_layer)
        total_weights = [1]*len(my_layer)
        for i in range(len(layer_list)):
            for c in range(len(my_layer)):
                my_new_layer[c] = my_layer[c] + layer_list[i][c]*fa_wg_list[i][c]
                total_weights[c] += fa_wg_list[i][c]
                
        for c in range(len(my_layer)):
            my_new_layer[c] = my_new_layer[c]/total_weights[c]
        
        return my_new_layer

    def aggregate(self, my_model, submissions, data_tf, image_lib = 'cifar'):
        _, my_detailed_f1 = self.__calc_F1(data_tf, my_model)
        my_last_dense_layer = self.__get_last_dense_layer_weights(my_model)
        prioritizer = Prioritizer(my_last_dense_layer, .4)

        weight_list = []
        last_dense_layer_list = []
        f1_list = []

        model = SimpleMLP.build(784 if image_lib == 'mnist' else 1024, 10)

        for submission in submissions:
            weights_cid = submission[3]
            weights = self.weights_loader.load(weights_cid)
            weight_list.append(weights)
            model.set_weights(weights)
            last_dense_layer_list.append(self.__get_last_dense_layer_weights(model))

        last_dense_layer_map = {k:last_dense_layer_list[k] for k in range(len(last_dense_layer_list))}
        prioritized_map = prioritizer.get_prioritized_weights(last_dense_layer_map)
        
        prioritized_last_dense_list = []
        for key in prioritized_map.keys():
            model.set_weights(weight_list[key])
            _, f1 = self.__calc_F1(data_tf, model)
            f1_list.append(f1)
            prioritized_last_dense_list.append(last_dense_layer_list[key])

        fa, fo, fo_columns = self.__compute_fa_fo_wg(my_detailed_f1, f1_list, fi=15, mu=0.9, omega_fa1=1.0, omega_fa2=0.0001, omega_fo1=0.99, omega_fo2=0.0001)
        my_new_last_dense_layer = self.__layers_weighted_average(my_last_dense_layer, prioritized_last_dense_list, fa, fo_columns)
        self.__set_last_dense_layer_weights(my_model, my_new_last_dense_layer)

        return my_model
