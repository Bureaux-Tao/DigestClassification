import os

# event_type = "pulmonary"
event_type = "digest"

#################################################################################

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前地址
proj_path = current_dir

weights_path = proj_path + "/weights"

##################################################################h

# Classification
train_file_path = proj_path + "/data/train_no_dump.txt"
val_file_path = proj_path + "/data/val_no_dump.txt"
test_file_path = proj_path + "/data/train_no_dump.txt"

#################################################################################

# Model Config
MODEL_TYPE = 'albert'

BASE_MODEL_DIR = proj_path + "/albert_tiny_google_zh"
BASE_CONFIG_NAME = proj_path + "/albert_tiny_google_zh/albert_config.json"
BASE_CKPT_NAME = proj_path + "/albert_tiny_google_zh/albert_model.ckpt"
