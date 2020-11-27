# sign_classification
This project contains configurable pipeline for training sign classification convolutional network with help of pytorch ignite.
# Installation:
1. Download classification part of dataset from http://graphics.cs.msu.ru/en/research/projects/rtsd and unpack it. 
  Dataset directory must have following structure:
  ```
   data_root
    |
    |- gt_train.csv
    |- gt_test.csv
    |- train/
          |
          |- 000001.png
          |- 000002.png
          |- ...
          |- xxxxxx.png
    |- test/
          |
          |- 000001.png
          |- 000002.png
          |- ...
          |- xxxxxx.png
  ```
2. Configure IDE to use virtual environment 'venv' or activate it manually

3. Study command line args. Atleast path to dataset should be set up.
# Usage:
python ./train.py --data-path <path to dataset root folder>

