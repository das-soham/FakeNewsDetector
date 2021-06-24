import os
import config
import sys
import shutil
import zipfile
from transformers import BertPreTrainedModel, BertTokenizer
import json
TOKENIZER_DOWNLOADER = False

def data_load():
    if not os.path.isfile(os.path.join(config.DATA_PATH, config.DATASET_TITLE)):
        os.system('kaggle datasets download -d' + config.DATASET)
        if not os.path.isdir(config.DATA_PATH):
            os.makedirs(config.DATA_PATH)
        shutil.move(config.DATASET_TITLE, os.path.join(config.DATA_PATH, config.DATASET_TITLE))
        if config.DATASET_TITLE.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(config.DATA_PATH, config.DATASET_TITLE)) as zip_data:
                zip_data.extractall(config.DATA_PATH)
    else:
        sys.stdout.write('Skipping downloading as data already exists')

'''
def model_load():
    if os.path.isfile('model/pytorch_model.bin'):
        model = BertPreTrainedModel.from_pretrained('model/')
    else:
        model = BertPreTrainedModel.from_pretrained(config.MODEL_NAME)
        model.save_pretrained(config.MODEL_PATH)
    if os.path.isfile('model/tokenizer_config.json'):
        config_file = open('model/tokenizer_config.json')
        config_data = config_file.readline()
        config_dict = json.loads(config_data)
        if os.path.isfile(config_dict['tokenizer_file']):
            tokenizer = BertTokenizer.from_pretrained('model/')
    else:
        tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
        tokenizer.save_pretrained(config.MODEL_PATH)
'''