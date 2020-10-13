import numpy as np


class BaseMetric(object):

  def __init__(self, rel_threshold, k):
    self.rel_threshold = rel_threshold
    if np.isscalar(k):
      k = np.array([k])
    self.k = k

  def __len__(self):
    return len(self.k)

  def __call__(self, *args, **kwargs):
    raise NotImplementedError

  def _compute(self, *args, **kwargs):
    raise NotImplementedError


class PrecisionRecall(BaseMetric):

  def __init__(self, rel_threshold=0, k=10):
    super(PrecisionRecall, self).__init__(rel_threshold, k)

  def __len__(self):
    return 2 * len(self.k)

  def __str__(self):
    str_precision = [('Precision@%1.f' % x) for x in self.k]
    str_recall = [('Recall@%1.f' % x) for x in self.k]
    return (','.join(str_precision)) + ',' + (','.join(str_recall))

  def __call__(self, targets, predictions):
    precision, recall = zip(*[
      self._compute(targets, predictions, x)
      for x in self.k
    ])
    result = np.concatenate((precision, recall), axis=0)
    return result

  def _compute(self, targets, predictions, k):
    predictions = predictions[:k]
    num_hit = len(set(predictions).intersection(set(targets)))

    return float(num_hit) / len(predictions), float(num_hit) / len(targets)


class MeanAP(BaseMetric):

  def __init__(self, rel_threshold=0, k=np.inf):
    super(MeanAP, self).__init__(rel_threshold, k)

  def __call__(self, targets, predictions):
    result = [self._compute(targets, predictions, x)
              for x in self.k]
    return np.array(result)

  def __str__(self):
    return ','.join([('MeanAP@%1.f' % x) for x in self.k])

  def _compute(self, targets, predictions, k):
    if len(predictions) > k:
      predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
      if p in targets and p not in predictions[:i]:
        num_hits += 1.0
        score += num_hits / (i + 1.0)

    if not list(targets):
      return 0.0

    return score / min(len(targets), k)


class NormalizedDCG(BaseMetric):

  def __init__(self, rel_threshold=0, k=10):
    super(NormalizedDCG, self).__init__(rel_threshold, k)

  def __call__(self, targets, predictions):
    result = [self._compute(targets, predictions, x)
              for x in self.k]
    return np.array(result)

  def __str__(self):
    return ','.join([('NDCG@%1.f' % x) for x in self.k])

  def _compute(self, targets, predictions, k):
    k = min(len(targets), k)

    if len(predictions) > k:
      predictions = predictions[:k]

    # compute idcg
    idcg = np.sum(1 / np.log2(np.arange(2, k + 2)))
    dcg = 0.0
    for i, p in enumerate(predictions):
      if p in targets:
        dcg += 1 / np.log2(i + 2)
    ndcg = dcg / idcg

    return ndcg


if __name__ == "__main__":
  labels = [[0], [0], [0]]
  preds = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.3], [0.3, 0.2, 0.1]]
  metric = MeanAP(k=2)
  for i in range(3):
    label = labels[i]
    pred = np.argsort(preds[i])[::-1]
    print(metric(label, pred))
