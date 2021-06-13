from transformers import BertPreTrainedModel, BertTokenizer
import config
import os
import json
TOKENIZER_DOWNLOADER = False

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

inputs = tokenizer('We love you!')
print(inputs)

