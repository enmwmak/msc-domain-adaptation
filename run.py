from .models import Xvector as Model
from .trainer import SemiTrainer, mmd
import os
import yaml


PROJECT_NAME = "my_project"
WORK_DIR = "/home/"
train_para = {
    # basic training para
    "sample_per_epoch": int(12e4), # The number of samles in each epoch.
    "sample_length_range": [400, 401], # Sampling speech frame range.
    "model": Model,
    "n_epoch": 1000, # Total epoch
    "lr": 0.001,
    "batch_size": 64,
    "world_size": 2, # The number of GPU
    "project_dir": f"{WORK_DIR}/{PROJECT_NAME}", # The folder that contains all things relevent to the experiments
    "train_file": f"{WORK_DIR}/h5/train_data.h5", # Training data
    "adapt_augment_file": f"{WORK_DIR}/h5/adaptation_data_aug.h5", # Training data
    # adapt para
    "target_sample_length_range": [400, 401], # Sampling speech frame range for target-domain data.
    "adapt_file": f"{WORK_DIR}/h5/adaptation_data.h5", # adaptation data
    "domain_loss_weight": 1,
    "unsup_loss": mmd,
    # model para
    "model_config": {
        "mfcc_dim": 23,
        "embedding_layer": "last",
    },
}

if not os.path.exists(train_para["project_dir"]):
    os.mkdir(train_para["project_dir"])
with open(train_para["project_dir"] + "/train_config.yaml", "a") as f:
    yaml.dump(train_para, f, default_flow_style=False)

SemiTrainer(**train_para).dist_train()


