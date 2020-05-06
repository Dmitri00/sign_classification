from collections import Counter

import torch.nn as nn

from .datasets import TrafficSign, TrafficSignSubset

class SubsetDatasetsBase(nn.Module):
  def forward(self, dataset):
    raise NotImplementedError
    
class SubsetDatasetsByClass(SubsetDatasetsBase):
  def __init__(self, class_labels):
    super().__init__()
    self.class_labels = set(class_labels)
  def split_func(self, dataset):
    base_labels = dataset.ground_truth
    subset_label_indexes = []
    for idx, label in enumerate(base_labels):
      if label in self.class_labels:
        subset_label_indexes.append(idx)
    return subset_label_indexes
  def forward(self, datasets):
    for ds_type, dataset in datasets.items():
      split_indices = self.split_func(dataset)
      datasets[ds_type] = dataset.subset(split_indices)
    return datasets

class SubsetDatasetsByTopFreq(SubsetDatasetsBase):
  def __init__(self, topk=5):
    super().__init__()
    self.topk = topk
  def get_topk_labels(self, dataset):
    label_count_pairs = Counter(dataset.ground_truth).items()
    label_col = 0
    count_col = 1
    label_count_pairs_sorted = sorted(label_count_pairs, 
                                      key=lambda x: x[count_col], reverse=True)
    topk_label_count_pairs = label_count_pairs_sorted[:self.topk]
    topk_labels = set((label_count[label_col] for label_count in topk_label_count_pairs))
    return topk_labels
  def split_func(self, dataset, topk_labels):
    topk_label_indices = []
    for idx, label in enumerate(dataset.ground_truth):
      if label in topk_labels:
        topk_label_indices.append(idx)
    return topk_label_indices
  def forward(self, datasets):
    topk_labels = self.get_topk_labels(datasets['train'])
    for ds_type, dataset in datasets.items():
      split_indices = self.split_func(dataset, topk_labels)
      datasets[ds_type] = dataset.subset(split_indices)
    return datasets



class DatasetSubsetPipeline(nn.Module):
  def __init__(self, *dataset_splitters):
    super().__init__()
    for dataset_splitter in dataset_splitters:
      if not isinstance(dataset_splitter, SubsetDatasetsBase):
        raise ValueError('Dataset splitter must be a child class of SubsetDatasetsBase')
    self.dataset_splitters = dataset_splitters
  def forward(self, datasets):
    subsets = datasets
    for dataset_splitter in self.dataset_splitters:
      subsets = dataset_splitter(subsets)
    return subsets
