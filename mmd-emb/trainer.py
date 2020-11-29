import os
import random
import time
import warnings
from random import shuffle

import GPUtil
import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.decomposition import PCA
from torch.multiprocessing import Process

from torch.utils.data import Dataset
import scipy.io as sio

from scorer import Scorer
from datasets import RandomSampleDataset
from datasets import InnerDataset
from datasets import ExtractDataset
from extract import extract_collate
import sys


class Trainer:
    def __init__(
        self,
        n_epoch,
        lr,
        batch_size,
        project_dir,
        model_config,
        min_utts_per_spk,
        model,
        gpu_ids=None,
        init_file=None,
        only_keep_two_model=False,
        checkpoint_warmrestart=False,
        cooling_epoch=None,
        optimizer_choice='adam',
        scheduler=None,
        scheduler_config=None,
        sample_per_epoch=None,
        sample_length_range=None,
        n_blocks_in_samples=1,
        train_file=None,
        train_egs_dir=None,
        training_metadata=None,
        checkpoint_interval=20,
        valid_enroll=None,
        valid_test=None,
        valid_target=None,
        valid_trial_list=None,
        world_size=None,
        weight_decay=0,
        checkpoint=None,
        score_paras=None,
    ):

        if not os.path.exists(f"{project_dir}"):
            os.mkdir(project_dir)
        if not os.path.exists(f"{project_dir}/tmp"):
            os.mkdir(project_dir + "/tmp")
        if not os.path.exists(f"{project_dir}/models"):
            os.mkdir(project_dir + "/models")
        if not os.path.exists(f"{project_dir}/loggers"):
            os.mkdir(project_dir + "/loggers")
        if not os.path.exists(f"{project_dir}/share_file"):
            os.mkdir(project_dir + "/share_file")
        if not os.path.exists(f"{project_dir}/h5"):
            os.mkdir(project_dir + "/h5")
        if not os.path.exists(f"{project_dir}/h5/first_layer"):
            os.mkdir(project_dir + "/h5/first_layer")
        if not os.path.exists(f"{project_dir}/h5/last_layer"):
            os.mkdir(project_dir + "/h5/last_layer")

        # training paras
        if not cooling_epoch:
            self.cooling_epoch = n_epoch
        else:
            print('with cooling')
            self.cooling_epoch = cooling_epoch
        self.score_paras = score_paras
        self.checkpoint_warmrestart = checkpoint_warmrestart

        self.scheduler_choice = scheduler
        self.scheduler_config = scheduler_config

        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.optimizer_choice = optimizer_choice
        self.lr = lr
        self.weight_decay = weight_decay
        self.world_size = world_size
        self.only_keep_two_model = only_keep_two_model
        self.checkpoint_interval = checkpoint_interval

        # loader paras
        self.sample_length_range = sample_length_range
        self.min_utts_per_spk = min_utts_per_spk
        self.n_blocks_in_samples = n_blocks_in_samples
        self.sample_per_epoch = sample_per_epoch

        # files
        self.train_file = train_file
        self.train_egs_dir = train_egs_dir
        self.training_metadata = training_metadata
        self.project_dir = project_dir
        self.logger_dir = f"{project_dir}/loggers"
        self.checkpoint = checkpoint
        self.save_checkpoint_to = f"{project_dir}/models/model"
        if init_file is None:
            self.init_file = f"{project_dir}/share_file/{random.random()}"
        else:
            self.init_file = init_file
        self.valid_enroll, self.valid_test = valid_enroll, valid_test
        self.valid_target = valid_target
        self.valid_trial_list = valid_trial_list

        if train_file:
            read_class_info_from = train_file if not training_metadata else training_metadata
            model_config["n_classes"] = self.get_n_classes(read_class_info_from,
                                                           min_utts_per_spk=min_utts_per_spk,
                                                           min_frames_per_utt=sample_length_range[1])
            print(f"Total no. of speakers is {model_config['n_classes']}")

        self.model_config = model_config
        self.model = model(**model_config)

        if checkpoint:
            print(f"loading from {checkpoint}")
        else:
            # breakpoint()
            model_config_copy = model_config.copy()
            with open(f"{project_dir}/models/model_config.yaml", "w") as f:
                yaml.dump(model_config_copy, f, default_flow_style=False)

        # self.model_config = model_config
        # self.model = model

        with open(f"{project_dir}/models/model.log", "w") as f:
            print(self.model, file=f)

        self.info = {}
        self.gpu_ids = gpu_ids

    def general_train(self):
        if self.world_size == 1:
            self.single_train()
        else:
            self.dist_train()

    def dist_train(self):
        gpu_ids_avail = GPUtil.getAvailable(maxMemory=0.02, limit=8)
        shuffle(gpu_ids_avail)
        gpu_ids = gpu_ids_avail[: self.world_size]
        assert len(gpu_ids) == self.world_size, "not enough GPUs"
        processes = []
        for rank, gpu_id in enumerate(gpu_ids):
            p = Process(target=self._dist_train, args=(rank, gpu_id))
            p.start()
            print(f"process {rank} has started")
            processes.append(p)

        for p in processes:
            p.join()

    def init_model(self, gpu_id=0):
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(torch.cuda.current_device())
        self.model = self.model.to(self.device)
        if self.checkpoint:
            self.load_checkpoint(self.checkpoint)
        else:
            self.epoch = 0
            self.model = self.model.cuda()
            self.init_optimizer()


    def _dist_train(self, rank, gpu_id):
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(torch.cuda.current_device())

        if self.checkpoint:
            self.load_checkpoint(self.checkpoint)
        else:
            self.epoch = 0
            self.model = self.model.cuda()
            self.init_optimizer()

        self.gpu_id = gpu_id
        # self.init_model(gpu_id)
        self.init_dist(rank)
        self.batch_size = self.batch_size // self.world_size
        self.rank = rank
        t0 = time.time()
        if self.train_egs_dir:
            local_file_lst = np.array_split(self.file_lst, self.world_size)[rank].tolist()
            self.egs_train(local_file_lst)
        else:
            self.train()    # Will call SemiTrainer.train() if self is an object of SemiTrainer
        if rank == 0:
            self.save_model(f"{self.save_checkpoint_to}_final")
        print(f"train total time is {time.time() - t0}")

    def single_train(self):
        if self.gpu_ids is None:
            self.gpu_ids = GPUtil.getAvailable(maxMemory=0.02, limit=8)
        gpu_ids = self.gpu_ids[0]
        self.init_model(gpu_ids)
        self.rank = 0
        self.train()        # Will call SemiTrainer.train() if self is an object of SemiTrainer
        self.save_model(f"{self.save_checkpoint_to}_final")

    def egs_train(self, file_list):
        outer = OuterDataset(file_list, logger=f'{self.logger_dir}/data_{self.rank}.log',
                             inner_batch_size=self.batch_size // self.world_size)
        warnings.warn('outer_loader is not shuffled')
        outer_loader = torch.utils.data.DataLoader(outer, batch_size=1, shuffle=False,
                                                   collate_fn=unpack_list, num_workers=1)
        for epoch in range(self.epoch, self.epoch + self.n_epoch):
            for step, inner_load in enumerate(outer_loader):
                self.epoch = epoch
                self.info['tot_loss'] = AverageMeter()
                self.info['acc'] = AverageMeter()
                self.end = time.time()
                self.model.train()
                self._train(inner_load)
                if self.rank == 0:
                    print(f'step {step+1}')
                    if bool(step % self.checkpoint_interval == 0) & bool(self.valid_trial_list):
                        self.eval_openset()
                        self.save_model(f"{self.save_checkpoint_to}_{self.epoch}_{step}.tar")

    def train(self):
        #print("Calling Trainer.train()")
        with h5py.File(self.train_file, "r") as f:
            dset = RandomSampleDataset(
                f,
                n_blocks=self.n_blocks_in_samples,
                sample_per_epoch=self.sample_per_epoch // self.world_size,
                balance_class=True,
                meta_data_file=self.training_metadata,
                min_frames_per_utt=self.sample_length_range[1],
                min_utts_per_spk=self.min_utts_per_spk,
                sample_length_range=self.sample_length_range,
            )
            loader = torch.utils.data.DataLoader(
                dset, batch_size=self.batch_size, shuffle=False, num_workers=1
            )
            for epoch in range(self.epoch, self.epoch + self.n_epoch):
                loader.dataset.sample()

                self.epoch = epoch
                self.info['tot_loss'] = AverageMeter()
                self.info['acc'] = AverageMeter()
                self.end = time.time()
                self.model.train()
                self._train(loader)

                self.info['time'] = time.time() - self.end
                self.end = time.time()
                self.scheduler.step()
                if self.rank == 0:
                    message = (f"epoch {self.epoch} "
                               f"time {self.info['time']:.0f} "
                               f"loss {self.info['tot_loss'].avg:.3f} "
                               f"acc {self.info['acc'].avg:.3f}\n")
                    with open(f"{self.logger_dir}/training.log", "a") as f:
                        f.write(message)
                    print(message, end="")

                if self.rank == 0:
                    # if ((self.epoch + 1) in self.checkpoint_interval) & bool(self.valid_trial_list):
                    if ((self.epoch + 1) % self.checkpoint_interval) == 0 & bool(self.valid_trial_list):
                        self.eval_openset()
                        # if self.epoch >= 100:
                        # if self.epoch >= 30:
                        self.save_model(f"{self.save_checkpoint_to}_{self.epoch}.tar")
                        print('save model')
                        if self.only_keep_two_model:
                            if os.path.exists(f"{self.save_checkpoint_to}_{self.epoch-2*self.checkpoint_interval}.tar"):
                                os.remove(f"{self.save_checkpoint_to}_{self.epoch-2*self.checkpoint_interval}.tar")

                    if self.epoch == 0:
                        self.eval_openset()

    def _train(self, loader):
        #print("Calling Trainer._train()")
        for step, (mfcc, spk_ids, _) in enumerate(loader):
            self.optimizer.zero_grad()

            mfcc, spk_ids = mfcc.cuda(), spk_ids.cuda()
            # if self.model_config is not None:
            #     if self.model_config['pooling_layer'] == 'mask_pooling':
            #         spk_ids = spk_ids.repeat(self.model_config['pooling_config']['n_mask'])
            logit, logit_nomargin = self.model(mfcc, spk_ids)
            loss = F.cross_entropy(logit, spk_ids)

            loss.backward()
            self.optimizer.step()

            acc = logit_nomargin.max(-1)[1].eq(spk_ids).sum().item() / len(spk_ids) * 100
            self.info['tot_loss'].update(loss.item(), spk_ids.shape[0])
            self.info['acc'].update(acc, spk_ids.shape[0])


    def eval_openset(self):
        self.sequential_extract(self.valid_test, f"{self.project_dir}/tmp/test.h5")
        if self.valid_enroll:
            self.sequential_extract(self.valid_enroll, f"{self.project_dir}/tmp/enroll.h5")
            enroll_embedding = f"{self.project_dir}/tmp/enroll.h5"
        else:
            enroll_embedding = f"{self.project_dir}/tmp/test.h5"

        if self.valid_target:
            self.sequential_extract(
                self.valid_target, f"{self.project_dir}/tmp/target.h5"
            )
            data_target = h52dict(f"{self.project_dir}/tmp/target.h5")
            transform_lst = [PCA(whiten=True)]
            for transform in transform_lst:
                transform.fit_transform(data_target["X"])
        else:
            transform_lst = None

        if self.score_paras is None:
            self.score_paras = {}
        scorer = Scorer(
            comp_minDCF=False,
            enroll=enroll_embedding,
            test=f"{self.project_dir}/tmp/test.h5",
            ndx_file=self.valid_trial_list,
            transforms=transform_lst,
            **self.score_paras,
        )
        eer = scorer.batch_cosine_score()

        with open(f"{self.logger_dir}/validation.log", "a") as f:
            f.write(f"{self.epoch} EER is {eer}\n")

    def sequential_extract(self, mfcc_file, save_xvec_to):
        print(f"reading mfcc from {mfcc_file}")
        with h5py.File(mfcc_file, "r") as fr:
            dset = ExtractDataset(fr)
            loader = torch.utils.data.DataLoader(
                dset, batch_size=1, shuffle=False, collate_fn=extract_collate
            )
            with h5py.File(save_xvec_to, "w") as fw:
                fw["X"], fw["spk_ids"], fw["spk_path"] = self._extraction(loader)

    def _extraction(self, loader):
        self.model.eval()
        X, spk_ids, utt_ids = [], [], []
        with torch.no_grad():
            for batch_idx, (mfcc, spk_id, utt_id) in enumerate(loader):
                # if self.model_choice.endswith('2d'):
                #     mfcc = mfcc[:, None, ...]
                mfcc = mfcc.to(self.device)
                if self.world_size == 1:
                    x = self.model.extract(mfcc)
                else:
                    x = self.model.module.extract(mfcc)
                spk_ids.append(spk_id)
                utt_ids.append(utt_id)
                X.append(x)

            X = torch.cat(X).to("cpu").numpy()
            spk_ids = np.stack(spk_ids).astype(unicode)
            utt_ids = np.stack(utt_ids).astype(unicode)
        return X, spk_ids, utt_ids

    def init_dist(self, rank):
        print("Initialize Process Group...")
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{self.init_file}",
            rank=rank,
            world_size=self.world_size,
        )
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            # find_unused_parameters=True,
        )

    def load_checkpoint(self, checkpoint):
        # checkpoint = torch.load(
        #     checkpoint, map_location=torch.device(torch.cuda.current_device())
        # )
        checkpoint = torch.load(
            checkpoint, map_location=torch.device('cpu')
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model = self.model.cuda()
        self.init_optimizer()

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"] + 1


    def save_model(self, save_checkpoint_to):
        if not os.path.exists(os.path.dirname(save_checkpoint_to)):
            os.makedirs(os.path.dirname(save_checkpoint_to))
        print('Saving model to %s' % save_checkpoint_to)    
        torch.save(
            {
#                "model_state_dict": self.model.state_dict() if self.world_size == 1 else self.model.module.state_dict(),
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "logger": self.logger_dir,
            },
            save_checkpoint_to,
        )

    @staticmethod
    def get_n_classes(file, min_utts_per_spk, min_frames_per_utt):
        with h5py.File(file, 'r') as f:
            df = pd.DataFrame(
                {
                    "spk_ids": f["spk_ids"][:],
                    "utt_ids": f["utt_ids"][:],
                    "starts": f["positions"][:, 0],
                    "ends": f["positions"][:, 1],
                }
            )
            df = df[(df.ends - df.starts) > min_frames_per_utt]
            utt_counts = df.spk_ids.value_counts()
            df["n_utts"] = df.spk_ids.map(utt_counts)
            df = df[df.n_utts > min_utts_per_spk]
        return df.spk_ids.nunique()

    def init_optimizer(self):
        if self.optimizer_choice == 'adam':
            print('using adam')
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_choice == 'sgd':
            print('using sgd')
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                momentum=0.9,
            )
        elif self.optimizer_choice == 'adamW':
            print('using adamW')
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            )

        if self.scheduler_choice == 'MultiStepLR':
            print('using MultiStepLR')
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, **self.scheduler_config,
            )
        elif self.scheduler_choice == 'CosineAnnealingLR':
            print('CosineAnnealingLR')
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **self.scheduler_config,
            )
        elif self.scheduler_choice == 'ReduceLROnPlateau':
            print('ReduceLROnPlateau')
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **self.scheduler_config,
            )
        elif self.scheduler_choice == 'CosineAnnealingWarmRestarts':
            print('CosineAnnealingWarmRestarts')
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, **self.scheduler_config,
            )
        else:
            print('No scheduler')
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=1
            )



def unpack_list(x):
    return x[0]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class OuterDataset(Dataset):
    def __init__(self, file_lst, logger, inner_batch_size, padding=False):
        self.batch_size = inner_batch_size
        self.logger = logger
        if not padding:
            self.file_lst = file_lst
        else:
            self.file_lst = np.tile(file_lst, 30).tolist()
        # self.file_lst = np.tile(file_lst, 20).tolist()[:total_step]

    def __getitem__(self, idx):
        t0 = time.time()
        with h5py.File(self.file_lst[idx], 'r') as f:
            data = {
                'positions': f['positions'][:],
                'spk_ids': f['spk_ids'][:],
                'utt_ids': f['utt_ids'][:],
                'mfcc': f['mfcc'][:],
            }
        with open(self.logger, 'a') as f:
            f.write(f'disk time is {time.time()-t0}\n')
        inner = InnerDataset(data)
        inner_loader = torch.utils.data.DataLoader(inner,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   )
        return inner_loader

    def __len__(self):
        return len(self.file_lst)





unicode = h5py.special_dtype(vlen=str)


def h5list2dict(files):
    with h5py.File(files[0], 'r') as f:
        attributs = list(f.keys())

    data = {attribut: [] for attribut in attributs}

    for file in files:
        with h5py.File(file, 'r') as f:
            assert set(attributs) == set(list(f.keys()))
            for attribut in attributs:
                data[attribut].append(f[attribut][...])

    data = {attribut: np.concatenate(data[attribut]) for attribut in attributs}
    return data


def h52dict(file):
    if type(file) is str:
        return h5single2dict(file)
    elif type(file) is list:
        return h5list2dict(file)
    else:
        raise NotImplementedError


def h5single2dict(file):
    with h5py.File(file, 'r') as f:
        return {name: f[name][...] for name in f}


def dict2h5(in_dict, file, dataset='', mode='a'):
    with h5py.File(file, mode) as f:
        for key, val in in_dict.items():
            if val.dtype == np.object:
                f[dataset + key] = val.astype(unicode)
            else:
                f[dataset + key] = val


def mat2h5(mat_file, h5_file,  n_dims_keeped=None):
    with h5py.File(h5_file, 'a') as f:
        mat_conts = sio.loadmat(mat_file, squeeze_me=True)
        if n_dims_keeped is not None:
            mat_conts['w'] = mat_conts['w'][:, :n_dims_keeped]
        f['spk_ids'] = mat_conts['spk_logical'].astype(unicode)
        f['X'] = mat_conts['w']
        f['n_frames'] = mat_conts['num_frames']
        f['spk_path'] = mat_conts['spk_physical'].astype(unicode)


# median  62.42532
SIGMAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
          1e-1, 1, 5, 10, 15, 20, 25,
          30, 35, 100, 1e3, 1e4, 1e5, 1e6)



def l2_dist(x, y):
    return (x - y).pow(2).sum()


def cosine_dist(x, y):
    x = x / x.norm()
    y = y / y.norm()
    return x @ y
# SIGMAS = [62.42532*x for x in np.exp2(np.arange(-8, 8+0.5, step=0.5))]

#
# def linear_mmd(x, y):
#     cost = linear_kernel(x, x) \
#            + linear_kernel(y, y) \
#            - 2 * linear_kernel(x, y)
#     return cost


# def linear_kernel(x, y):
#     dist = my_cdist(x, y)[:, None, None]
#     # Todo do not know if it's ok
#     # dist = dist / x.shape[-1]  # averge dim
#     return s.exp().mean()  # probably not mean # average sample
#

def mmd(x, y, sigmas=SIGMAS):
    cost = gaussian_kernel(x, x, sigmas) \
          + gaussian_kernel(y, y, sigmas) \
          - 2 * gaussian_kernel(x, y, sigmas)
    return cost
    # return cost.sqrt()


def gaussian_kernel(x, y, sigmas):
    sigmas = torch.tensor(sigmas, device=x.get_device())
    beta = 1. / (2. * sigmas[:, None, None])
    dist = my_cdist(x, y)[:, None, None]
    # Todo do not know if it's ok
    # dist = dist / x.shape[-1]  # averge dim
    s = -beta * dist
    return s.exp().mean()  # probably not mean # average sample


def my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res


def moment2_match(X_src, X_tgt):
    mom2_src = (X_src.t() @ X_src) / X_src.shape[0]
    mom2_tgt = (X_tgt.t() @ X_tgt) / X_tgt.shape[0]

    mom2_diff = (mom2_src - mom2_tgt).pow(2).mean()
    # mom1_diff = (X_src.mean(0) - X_tgt.mean(0)).pow(2).mean()
    # return mom2_diff + mom1_diff
    return mom2_diff


class SemiTrainer(Trainer):
    def __init__(
        self,
        target_sample_length_range,
        adapt_file,
        domain_loss_weight,
        unsup_loss,
        adapt_augment_file,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.adapt_file = adapt_file
        self.target_sample_range = target_sample_length_range
        self.domain_loss_weight = domain_loss_weight
        self.unsup_loss = unsup_loss
        self.adapt_augment_file = adapt_augment_file


    def train(self):
        # Source-domain training dataset for cross-entropy (classification) loss
        f_train = h5py.File(self.train_file, "r")
        train_dset = RandomSampleDataset(
            f_train,
            n_blocks=self.n_blocks_in_samples,
            sample_per_epoch=self.sample_per_epoch // self.world_size,
            balance_class=True,
            meta_data_file=self.training_metadata,
            min_frames_per_utt=self.sample_length_range[1],
            min_utts_per_spk=self.min_utts_per_spk,
            sample_length_range=self.sample_length_range,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=False, num_workers=1
        )

        # Unlabeled target-domain augmented dataset for consistency regularization
        f_target_aug = h5py.File(self.adapt_augment_file, "r")
        target_dset_aug = RandomSampleDataset(
            f_target_aug,
            sample_length_range=self.target_sample_range,
            n_blocks=1,
            sample_per_epoch=self.sample_per_epoch // self.world_size,
            balance_class=False,
            min_frames_per_utt=self.target_sample_range[1],
            min_utts_per_spk=0,
        )
        target_loader_aug = torch.utils.data.DataLoader(
            target_dset_aug, batch_size=self.batch_size, shuffle=False, num_workers=1
        )

        # Unlabeled target-domain dataset for MMD loss and consistency regularization
        f_target = h5py.File(self.adapt_file, "r")
        target_dset = RandomSampleDataset(
            f_target,
            sample_length_range=self.target_sample_range,
            n_blocks=1,
            sample_per_epoch=self.sample_per_epoch // self.world_size,
            balance_class=False,
            min_frames_per_utt=self.target_sample_range[1],
            min_utts_per_spk=0,
        )
        target_loader = torch.utils.data.DataLoader(
            target_dset, batch_size=self.batch_size, shuffle=False, num_workers=1
        )

        for epoch in range(self.epoch, self.epoch + self.n_epoch):
            train_loader.dataset.sample()
            target_loader.dataset.sample()
            target_loader_aug.dataset.sample()

            self.epoch = epoch
            self.end = time.time()
            self.info['tot_loss'] = AverageMeter()
            self.info['sup/acc'] = AverageMeter()
            self.info['unsup/loss'] = AverageMeter()

            self.model.train()
            self._semi_train(train_loader, target_loader, target_loader_aug)

            # self.losses = AverageMeter()
            # self.accs = AverageMeter()
            # self.total_time = AverageMeter()
            # self.stat_domain_loss = AverageMeter()
            # self.unsup_accs = AverageMeter()

        f_train.close()
        f_target.close()


    def _semi_train(self, train_loader, target_loader, target_loader_aug):
        #print("Calling SemiTrainer._semi_train()")
        for step, (data_src, data_tgt, data_tgt_aug) in enumerate(zip(train_loader, target_loader, target_loader_aug)):
            self.optimizer.zero_grad()
            mfcc_src, spk_ids_src, _ = data_src
            mfcc_tgt, spk_ids_tgt, _ = data_tgt
            mfcc_tgt_aug, spk_ids_tgt_aug, _ = data_tgt_aug

            mfcc_src, spk_ids_src = mfcc_src.cuda(), spk_ids_src.cuda()
            mfcc_tgt = mfcc_tgt.cuda()
            mfcc_tgt_aug = mfcc_tgt_aug.cuda()
            
            # For source-domain
            frames_src = self.model.module.frame_layers(mfcc_src)  # 128X1500X400
            stats_src = self.model.module.stat_pooling(frames_src)
            embed_src = self.model.module.utt_layers(stats_src)
            if self.model.module.__class__.__name__ == 'DenseNet':
                logit_src, logit_nomargin_src = self.model.module.am_linear(embed_src, spk_ids_src)
            elif self.model.module.__class__.__name__ == 'XvectorNet':
                logit_src = self.model.module.am_linear(embed_src)
            else:
                sys.exit("Wrong network type, can only be DenseNet or XvectorNet")

            # For target-domain
            frames_tgt = self.model.module.frame_layers(mfcc_tgt)

            stats_tgt = self.model.module.stat_pooling(frames_tgt)
            embed_tgt = self.model.module.utt_layers(stats_tgt)

            # For target-domain augmented data
            frames_tgt_aug = self.model.module.frame_layers(mfcc_tgt_aug)
            stats_tgt_aug = self.model.module.stat_pooling(frames_tgt_aug)
            embed_tgt_aug = self.model.module.utt_layers(stats_tgt_aug)

            cls_loss = F.cross_entropy(logit_src, spk_ids_src)
            if self.model.module.__class__.__name__ == 'DenseNet':
                loss2 = self.unsup_loss(frames_src.permute(0, 2, 1).reshape(-1, 1280)[:],
                                        frames_tgt.permute(0, 2, 1).reshape(-1, 1280)[:])                
            else:    
                sample_idx = np.random.randint(frames_tgt.shape[0] * frames_tgt.shape[2] - 3000)
                loss2 = self.unsup_loss(frames_src.permute(0, 2, 1).reshape(-1, 1500)[sample_idx:sample_idx + 3000],
                                        frames_tgt.permute(0, 2, 1).reshape(-1, 1500)[sample_idx:sample_idx + 3000])

            domain_loss = self.domain_loss_weight * self.unsup_loss(embed_tgt, embed_src) + loss2 + \
                            self.unsup_loss(embed_tgt, embed_tgt_aug)
            loss = cls_loss + domain_loss

            loss.backward()
            self.optimizer.step()

            if self.model.module.__class__.__name__ == 'DenseNet':
                acc = logit_nomargin_src.max(-1)[1].eq(spk_ids_src).sum().item() / len(spk_ids_src) * 100
            else:    
                acc = logit_src.max(-1)[1].eq(spk_ids_src).sum().item() / len(spk_ids_src) * 100
            self.info['sup/acc'].update(acc, spk_ids_src.shape[0])
            self.info['unsup/loss'].update(domain_loss.item(), spk_ids_src.shape[0])
            self.info['tot_loss'].update(loss.item(), spk_ids_src.shape[0])
            self.info['time'] = time.time() - self.end
            self.end = time.time()

        if self.rank == 0:
            message = (f"epoch {self.epoch} " 
                       f"time {self.info['time']:.0f} " 
                       f"loss {self.info['tot_loss'].avg:.3f} " 
                       f"acc {self.info['sup/acc'].avg:.3f} "
                       f"unsup loss {self.info['unsup/loss'].avg:.3f}\n")
            with open(f"{self.logger_dir}/training.log", "a") as f:
                f.write(message)
            print(message, end="")
            if self.epoch % self.checkpoint_interval == 0:
                if self.epoch > 80:
                    self.save_model(f"{self.save_checkpoint_to}_{self.epoch}")
                if self.valid_trial_list:
                    self.eval_openset()


