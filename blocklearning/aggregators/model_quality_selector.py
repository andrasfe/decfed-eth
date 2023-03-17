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

    def __weights_from_submissions(self, submissions):
        weights_cids = [cid for (_, _, _, cid, _, _) in submissions]
        return [self.weights_loader.load(cid) for cid in weights_cids]

    def bestScores(self, trainers, submissions):
        weights = self.__weights_from_submissions(submissions)
        trainers, scores, _ = score(weights, trainers)

        return self.__select_top(trainers, scores)

    def closestToLocal(self, my_weights, other_trainers, submissions):
        weights = self.__weights_from_submissions(submissions)
        return local_score(weights, my_weights, other_trainers)

 