from trainer import SemiTrainer, mmd, Trainer
import os
import yaml
from scorer import Scorer
from extract import sequential_extract
import torch

PROJECT_NAME = "T-ASLP20"
WORK_DIR = "/home2a/mwmak/so/spkver/sre18-dev"
PRJ_DIR = WORK_DIR + '/' + PROJECT_NAME
run_mode = "scoring"                             # "training" or "scoring"
net_type = 'XvectorNet'                        # 'XvectorNet' or DenseNet
mfcc_dim = 41

if net_type == 'XvectorNet':
    from xvectornet import XvectorNet as Model
elif net_type == 'DenseNet':
    from densenet import DenseNet121 as Model
    
train_para = {
    # basic training para
    "sample_per_epoch": int(12e4), # The number of samles in each epoch.
    "sample_length_range": [400, 401], # Sampling speech frame range.
    "model": Model,
    "n_epoch": 30, # No. of epoch (normally 1000, set to 10 for test)
    "lr": 0.001,
    "batch_size": 64,
    "min_utts_per_spk": 3,
    "world_size": 2, # The number of GPU
    "project_dir": f"{PRJ_DIR}", # The folder that contains all things relevent to the experiments
    "train_file": f"{PRJ_DIR}/h5/train_data.h5",  # Training data from source domain
    "adapt_augment_file": f"{PRJ_DIR}/h5/adaptation_data_aug.h5", # Augmentation data from target domain

    # adapt para
    "target_sample_length_range": [400, 401], # Sampling speech frame range for target-domain data.
    "adapt_file": f"{PRJ_DIR}/h5/adaptation_data.h5", # adaptation data
    "domain_loss_weight": 1,
    "unsup_loss": mmd,
}
if net_type == 'XvectorNet':
    train_para["model_config"] = {
        "mfcc_dim": mfcc_dim,
        "embedding_layer": "last",
    }    
elif net_type == 'DenseNet':
    train_para["model_config"] = {
        "mfcc_dim": mfcc_dim,
    }

score_para = {
    "enroll": f"{PRJ_DIR}/h5/xvector_enroll.h5",
    "test": f"{PRJ_DIR}/h5/xvector_test.h5",
    "ndx_file": f"/corpus/sre18-eval/docs/sre18_eval_trial_key.tsv",
    "average_by": "spk_ids",
    "score_using_spk_ids": True,
    "data_source": "cmn2",              # 'cmn2' or 'vast'        
}

if run_mode == "training":
    print("Training...")
    if not os.path.exists(train_para["project_dir"]):
        os.mkdir(train_para["project_dir"])
    with open(train_para["project_dir"] + "/models/train_config.yaml", "a") as f:
        yaml.dump(train_para, f, default_flow_style=False)
        
    SemiTrainer(**train_para).dist_train()
    # Calling sequence: SemiTrainer.dist_train() --> Trainer.dist_train() --> 
    # Trainer._dist_train() --> SemiTrainer.train() --> SemiTrainer._semi_train()
    

if run_mode == "scoring":
    print('Scoring...')
    nclasses = Trainer.get_n_classes(train_para['train_file'], train_para['min_utts_per_spk'], 
                                     train_para['sample_length_range'][1])
    if net_type == 'DenseNet':
        densenet_para = {'n_classes': nclasses, 'mfcc_dim': mfcc_dim}
        model = Model(**densenet_para)
    else:
        model = Model(nclasses, mfcc_dim)
    model.load_state_dict(torch.load(f"{PRJ_DIR}/models/model_final")['model_state_dict'])
    model.to(torch.device(torch.cuda.current_device()))

    sequential_extract(model, f"{PRJ_DIR}/h5/mfcc_enroll.h5", score_para['enroll'])
    sequential_extract(model, f"{PRJ_DIR}/h5/mfcc_test.h5", score_para['test'])
    scorer = Scorer(**score_para)
    scorer.batch_cosine_score()