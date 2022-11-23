from statistics import stdev
from  blocklearning.models import SimpleMLP
import tensorflow as tf
import math
import copy
from sklearn.metrics import f1_score

class BasilAggregator():
    def __init__(self, weights_loader):
        self.weights_loader = weights_loader

    def calc_F1(self, test_data_tf,  y_local_test, model, labels = [0,1,2,3,4,5,6,7,8,9], logits = None, fi=10):
        if logits == None:
            logits = model.predict(test_data_tf)

        detailed_f1 = f1_score(tf.argmax(logits, axis=1), tf.argmax(y_local_test, axis=1), average=None, labels=labels, zero_division=1)
        f1 = f1_score(tf.argmax(logits, axis=1), tf.argmax(y_local_test, axis=1), average='macro', labels=labels, zero_division=1)

        return f1, detailed_f1

    def get_last_dense_layer_weights(self, model):
        return model.layers[-2].get_weights()

    def weight_sigmoid(self, omega1, omega2, disc, certainty):
        return max(0, omega1/(1 + math.exp(-disc/100)) - omega2)*certainty

    def compute_fa_fo_wg(self, my_detailed_f1, detailed_f1, fi=10, mu=0.9, omega_fa1=0.99, omega_fa2=0.1, omega_fo1=0.5, omega_fo2=0.5):
        disc = [0]*len(my_detailed_f1)
        fa_wg = [0]*len(my_detailed_f1)
        fo_columns = [0]*len(my_detailed_f1)
        
        best = sorted(detailed_f1, reverse=True)[:fi]
        certainty = max(sum(best)/len(best) - stdev(best), 0) if len(best) > 0 else 0
                
        for c in range(len(my_detailed_f1)):
            if detailed_f1[c] > 0 and my_detailed_f1[c] == 0:
                fo_columns[c] == 1
                
            weighed_diff = (detailed_f1[c] - my_detailed_f1[c])*mu
            if detailed_f1[c] > my_detailed_f1[c]:
                disc[c] = weighed_diff**(3 + my_detailed_f1[c])
            else:
                disc[c] = -math.inf #-1*pow(weighed_diff, 4 + int(my_detailed_f1[c]))

            fa_wg[c] = self.weight_sigmoid(omega_fa1, omega_fa2, disc[c], certainty)  

        fo_wg = self.weight_sigmoid(omega_fo1, omega_fo2, sum(disc), certainty)
        return fa_wg, fo_wg, fo_columns

    def layers_weighted_average(self, my_layer, layer_list, fa_wg_list, fo_columns_list):
        my_new_layer = copy.deepcopy(my_layer)
        total_weights = [1]*len(my_layer[1])
        for i in range(len(layer_list)):
            for c in range(len(my_layer[1])):
                my_new_layer[1][c] = my_layer[1][c] + layer_list[i][1][c]*fa_wg_list[i][c]
                total_weights[c] += fa_wg_list[i][c]
                
        for c in range(len(my_layer)):
            my_new_layer[1][c] = my_new_layer[1][c]/total_weights[c]
        
        return my_new_layer

    def scale_model_weights(self, weight, scalar):
        '''function for scaling a models weights'''
        weight_final = []
        steps = len(weight)
        for i in range(steps):
            weight_final.append(scalar * weight[i])
        return weight_final

    def sum_scaled_weights(self, scaled_weight_list):
        '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
        avg_grad = list()
        #get the average grad accross all client gradients
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean.numpy())
            
        return avg_grad

    def weighed_aggretate(self, weight_list, factors = []):
        if factors == []:
            factors = [1]*len(weight_list)
        
        for i in range(len(weight_list)):
            weight_list[i] = self.scale_model_weights(weight_list[i], factors[i]/sum(factors))
            
        return self.sum_scaled_weights(weight_list)

    def reconstruct_model(self, base_model_weights, last_dense_layer):
        model = SimpleMLP.build_model()
        model.set_weights(base_model_weights)
        model.layers[-2].set_weights(last_dense_layer)
        return model

    def local_aggregate(self, my_model, last_dense_layer_list, batched_data):        
        my_f1, my_detailed_f1 = self.calc_F1(test_data_tf, y_local_test, my_model)
        top_f1_values, top_ranked_weights, layer_list = self.local_rank(my_model, last_dense_layer_list, test_data_tf, y_local_test, my_f1, my_detailed_f1) 
        top_f1_values.append(my_f1)
        my_dense_layer = self.get_last_dense_layer_weights(my_model)
        top_ranked_weights.append(my_dense_layer)
        if len(top_ranked_weights) > 1:
            my_weights = self.weighed_aggretate(top_ranked_weights, top_f1_values)
            my_model.set_weights(my_weights)

        return my_model

    def get_last_dense_layer_list(self, submissions):
        weight_list = []
        model = SimpleMLP.build_model()

        for i, submission in enumerate(submissions):
            weights_cid = submission[3]
            weights = self.weights_loader.load(weights_cid)
            model.set_weights(weights)
            weight_list.append(self.get_last_dense_layer_weights(model))

        return weight_list


    def aggregate(self, my_model, submissions, batched_data):
        # warning - still includes own model. needs to be excluded
        last_dense_layer_list = self.get_last_dense_layer_list(submissions)