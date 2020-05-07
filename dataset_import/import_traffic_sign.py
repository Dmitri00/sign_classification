from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets.import_data_pipeline import data_transforms
from datasets.import_data_pipeline import dataset_split_pipeline
from datasets.import_data_pipeline import prepare_data
from datasets.traffic_sign import TrafficSign


def import_data(root, batch_size, num_workers):
    default_transforms = data_transforms
    default_splitter = dataset_split_pipeline
    sampler_classes = {'train': RandomSampler, 'val': SequentialSampler}
    dataset_class = TrafficSign
    return prepare_data(dataset_class, root, default_transforms, default_splitter,
                        sampler_classes, batch_size, num_workers)
