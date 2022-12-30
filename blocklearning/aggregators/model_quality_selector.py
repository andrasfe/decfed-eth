from .utils import *

class ModelSelector():
    def __init__(self, weights_loader):
        self.weights_loader = weights_loader

    def __select_top(self, trainers, scores):
        medians = []
        for t, trainer in enumerate(trainers):
            medians.append(np.median(scores[t]))

        R = len(trainers)
        f = R // 3 - 1

        sorted_idxs = np.argsort(medians)
        lowest_idxs = sorted_idxs[:R-f]
        return [trainers[i] for i in lowest_idxs], [scores[i] for i in lowest_idxs]



    def bestScores(self, trainers, submissions):
        trainers, scores, _ = score(self.weights_loader, trainers, submissions)
        return self.__select_top(trainers, scores)

    def closestToLocal(self, my_weights, other_trainers, other_submissions):
        return local_score(self.weights_loader, my_weights, other_trainers, other_submissions)

 