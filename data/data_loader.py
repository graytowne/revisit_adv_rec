import os
import pandas as pd
from scipy import sparse
import numpy as np


class DataLoader(object):

  def __init__(self, path):
    self.path = path

    self.user2id, self.item2id = self._load_mappings()
    self.n_users = len(self.user2id)
    self.n_items = len(self.item2id)

    self.train_data = None
    self.test_data = None

  def load_train_data(self):
    if self.train_data is None:
      path = os.path.join(self.path, 'train.csv')

      tp = pd.read_csv(path)
      rows, cols = tp['uid'], tp['sid']

      self.train_data = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype='float64',
        shape=(self.n_users, self.n_items))
    return self.train_data

  def load_test_data(self):
    if self.test_data is None:
      path = os.path.join(self.path, 'test.csv')

      tp = pd.read_csv(path)
      rows, cols = tp['uid'], tp['sid']

      self.test_data = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype='float64',
        shape=(self.n_users, self.n_items))
    return self.test_data

  def _load_mappings(self):
    path = os.path.join(self.path, 'user2id.txt')
    user2id = dict()
    id = 0
    with open(path, 'r') as f:
      for line in f:
        user2id[line] = id
        id += 1

    path = os.path.join(self.path, 'item2id.txt')
    item2id = dict()
    id = 0
    with open(path, 'r') as f:
      for line in f:
        item2id[line] = id
        id += 1

    return user2id, item2id
