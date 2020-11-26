from unittest import TestCase
import unittest
import os
from datasets.import_data_pipeline import data_transforms
from datasets.traffic_sign import TrafficSign


class TestTrafficSign(TestCase):
    def test_ds_length(self):
        ds_path = os.path.join(os.environ['PROJECT_DIR'], 'signs_toydataset')
        image_set = 'val'
        ds_len_expected = 50
        default_data_transforms = data_transforms
        ds = TrafficSign(ds_path, image_set, default_data_transforms)
        self.assertTrue(True)

        ds_len_real = len(ds)

        self.assertEqual(ds_len_expected, ds_len_real)


if __name__ == '__main__':
    unittest.main()
