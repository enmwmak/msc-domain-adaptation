import glob
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
from torch.utils.data import DataLoader

# from .score import Score


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import h5py
import warnings


class Score:
    def __init__(self, enroll, test, ndx_file,
                 comp_minDCF=True,
                 score_using_spk_ids=False,
                 average_by=None,
                 top_scores=200,
                 preserve_trial_order=False,
                 cohort=None, transforms=None, group_infos=None,
                 blind_trial=False, save_scores_to=None):
        # print('loading cohort ivc from {}'.format(cohort))
        # Todo implement score snorm
        # Todo xvector average
        self.top_scores = top_scores
        self.comp_minDCF = comp_minDCF
        self.preserve_trial_order = preserve_trial_order
        if (type(test) is str) or (type(test) is list):
            self.test = h52dict(test)
            print('loading test ivc from {}'.format(test))
        else:
            self.test = test

        if type(enroll) is str or (type(enroll) is list):
            self.enroll = h52dict(enroll)
            print('loading enroll ivc from {}'.format(enroll))
        else:
            self.enroll = enroll

        if average_by == 'spk_ids' or score_using_spk_ids:
            self.enroll = average_xvec(self.enroll, 'spk_ids')
            # self.test = average_xvec(self.test, 'spk_ids')
        elif average_by == 'utt_ids':
            self.enroll = average_xvec(self.enroll, 'spk_path')
            self.test = average_xvec(self.test, 'spk_path')
        else:
            print('no average')


        #Todo temp need fix
        if score_using_spk_ids:
            # self.test['spk_ids'] = self.test['spk_path']
            self.enroll['spk_path'] = self.enroll['spk_ids']
        else:
            self.enroll['spk_ids'] = self.enroll['spk_path']
        # self.test['spk_ids'] = self.test['spk_path']
        self.test['spk_ids'] = self.test['spk_path']
        # breakpoint()
        if cohort:
            # self.cohort = h52dict(cohort_file)

            if type(cohort) is str or (type(cohort) is list):
                self.cohort = h52dict(cohort)
                print('loading cohort_file ivc from {}'.format(cohort))
            else:
                self.cohort = cohort

        #Todo temp need fix
        if cohort:
            self.cohort['spk_ids'] = self.cohort['spk_path']


        if transforms:
            for transform in transforms:
                self.enroll['X'] = transform.transform(self.enroll['X'])
                self.test['X'] = transform.transform(self.test['X'])
                if cohort:
                    self.cohort['X'] = transform.transform(self.cohort['X'])
        if group_infos:
            for kind, file in group_infos.items():
                data = getattr(self, kind)
                group_info = pd.read_csv(file, sep='\t', header=None, names=['utt_id', 'group'])
                group_info = group_info.set_index('utt_id').group
                data['group'] = pd.Series(data['spk_path']).map(group_info)
                mask = (data['group']).isna()
                data['group'] = data['group'].values
                print(f'total {mask.sum()} NAN in found in {kind}')
                setattr(self, kind, mask_dict(data, ~mask))

        self.enroll['X_origin'] = self.enroll['X'].copy()
        self.test['X_origin'] = self.test['X'].copy()
        if cohort:
            self.cohort['X_origin'] = self.cohort['X'].copy()
        # Todo add format checking
        self.blind_trial = blind_trial
        if not blind_trial:
            # self.ndx = pd.read_csv(ndx_file, sep='\t', usecols=['enroll', 'test', 'label'])
            # self.ndx['label'] = self.ndx.label.map({'target': 1, 'nontarget': 0})
            self.ndx = (
                pd.read_csv(ndx_file, sep='\t', dtype=str,
                            usecols=['modelid', 'segmentid', 'targettype'])
                  .rename(columns={'modelid': 'enroll', 'segmentid': 'test', 'targettype': 'label'})
            )
            self.ndx['label'] = self.ndx.label.map({'target': 1, 'nontarget': 0})
            if preserve_trial_order:
                self.ndx = self.ndx.groupby(['enroll', 'test']).apply(lambda x: x.index.values).reset_index().rename(
                    {0: 'dup_index'}, axis=1)
            else:
                self.ndx = self.ndx.sort_values(by=['enroll', 'test'])
        else:
            # self.ndx = pd.read_csv(ndx_file, sep='\t', usecols=[0, 1], names=['enroll', 'test'])
            self.ndx = (
                pd.read_csv(ndx_file, sep='\t',   dtype=str,
                usecols=['modelid', 'segmentid'])
                  .rename(columns={'modelid': 'enroll', 'segmentid': 'test'})
            )
            if preserve_trial_order:
                self.ndx = self.ndx.groupby(['enroll', 'test']).apply(lambda x: x.index.values).reset_index().rename(
                    {0: 'dup_index'}, axis=1)
            else:
                self.ndx = self.ndx.sort_values(by=['enroll', 'test'])
        self.save_scores_to = save_scores_to

    def batch_plda_score(self, pq):
        score = BatchPLDAScore(self.enroll, self.test, pq=pq)
        self.ndx['scores'] = score.score(self.ndx)
        if hasattr(self, 'cohort'):
            normalizer = ScoreNormalizer(self.enroll, self.test, self.cohort, pq, top_scores=self.top_scores)
            self.ndx['scores'] = normalizer.normalize_scores(self.ndx)
        if not self.blind_trial:
            eer = comp_eer(self.ndx.scores, self.ndx.label)
            print(f'EER by PLDA scoreis {eer:.3f}')
            self.reset_X()

        if self.comp_minDCF:
            minDCF = compute_c_norm(self.ndx.scores, self.ndx.label)
            print(f'minDCF is  {minDCF:.3f}')

        if self.save_scores_to:
            print(self.save_scores_to)
            self.save_scores()

        # return self
        return eer

    def batch_cosine_score(self):
        score = BatchCosineScore(self.enroll, self.test)
        self.ndx['scores'] = score.score(self.ndx)
        if hasattr(self, 'cohort'):
            normalizer = ScoreNormalizer(self.enroll, self.test, self.cohort, top_scores=self.top_scores)
            self.ndx['scores'] = normalizer.normalize_scores(self.ndx)

        if self.preserve_trial_order:
            self.ndx = self.ndx.explode('dup_index').set_index('dup_index').sort_index()


        if self.save_scores_to:
            self.save_scores()
        if not self.blind_trial:
            # breakpoint()
            eer = comp_eer(self.ndx.scores, self.ndx.label)
            print(f'EER by cosine score is {eer:.3f}')
            if self.comp_minDCF:
                minDCF = compute_c_norm(self.ndx.scores, self.ndx.label)
                print(f'minDCF is  {minDCF:.3f}')
            return eer

    def plda_score(self, pq):
        scores = np.zeros(len(self.ndx))
        tgt_dict = build_id_dict(self.enroll['X'], self.enroll['spk_ids'])
        tst_dict = build_id_dict(self.test['X'], self.test['spk_ids'])

        for i, (tgt_id, tst_name) in enumerate(self.ndx[['enroll', 'test']].values):
            scores[i] = _plda_score_scores_averge(
                X=tgt_dict[tgt_id],
                y=tst_dict[tst_name].squeeze(),
                P=pq['P'],
                Q=pq['Q'],
                const=pq['const']
            )

            if i % 100000 == 0:
                print(f'{i}/{scores.shape[0]}, {tgt_id}, {tst_name}, {scores[i]}')

        self.ndx['scores'] = scores
        eer = comp_eer(scores, self.ndx.label)
        print(f'EER by PLDA scoreis {eer:.3f}')
        if self.comp_minDCF:
            minDCF = compute_c_norm(scores, self.ndx.label)
            print(f'minDCF is  {minDCF:.3f}')
        self.reset_X()
        return self

    def cosine_score(self):
        self.enroll['X'] = lennorm(self.enroll['X'])
        self.test['X'] = lennorm(self.test['X'])

        scores = np.zeros(len(self.ndx))
        tgt_dict = build_id_dict(self.enroll['X'], self.enroll['spk_ids'])
        tst_dict = build_id_dict(self.test['X'], self.test['spk_ids'])
        for i, (enroll_id, test_id) in enumerate(self.ndx[['enroll', 'test']].values):
            # breakpoint()
            scores[i] = tgt_dict[enroll_id].mean(0) @ tst_dict[test_id].squeeze().T
            if i % 100000 == 0:
                print(f'{i}/{scores.shape[0]}, {enroll_id}, {test_id}, {scores[i]}')

        self.ndx['scores'] = scores
        if not self.blind_trial:
            eer = comp_eer(scores, self.ndx.label)
            print(f'EER by cosine scoreis {eer:.3f}')

            if self.comp_minDCF:
                minDCF = compute_c_norm(scores, self.ndx.label)
                print(f'minDCF is  {minDCF:.3f}')
        # self.reset_X()
            return eer

    def save_scores(self):
        if not self.blind_trial:
            self.ndx_copy = self.ndx.copy()
            self.ndx_copy['label'] = self.ndx.label.map({1: 'target', 0: 'nontarget'})
            self.ndx_copy[['enroll', 'test', 'scores', 'label']].to_csv(self.save_scores_to, sep='\t', index=None)
            print(self.save_scores_to)
        elif self.blind_trial & self.preserve_trial_order:
            self.ndx['scores'] = MinMaxScaler().fit_transform(self.ndx['scores'].values.reshape(-1, 1))[:, 0]
            self.ndx[['enroll', 'test', 'scores']].to_csv(self.save_scores_to, sep='\t', index=None)
            print(self.save_scores_to)
        else:
            self.ndx[['enroll', 'test', 'scores']].to_csv(self.save_scores_to, sep='\t', index=None)
            print(self.save_scores_to)
        # self.ndx[['scores', 'label']].to_csv(scores_file, sep='\t', index=None)

    def reset_X(self):
        self.enroll['X'] = self.enroll['X_origin'].copy()
        self.test['X'] = self.test['X_origin'].copy()
        if hasattr(self, 'cohort'):
            self.cohort['X'] = self.cohort['X_origin'].copy()
        return self


def average_xvec(data, by):
    X, spk_ids, spk_path = [], [], []
    for uni in np.unique(data[by]):
        mask = data[by] == uni
        X.append(data['X'][mask].mean(0))
        spk_ids.append(data['spk_ids'][mask][0])
        spk_path.append(data['spk_path'][mask][0])
    return {
        'X': np.stack(X),
        'spk_ids': np.stack(spk_ids),
        'spk_path': np.stack(spk_path)
    }



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
        raise NotImplemented


def h5single2dict(file):
    with h5py.File(file, 'r') as f:
        return {name: f[name][...] for name in f}


from scipy import linalg as la
def lennorm(X):
    return X / la.norm(X, axis=1)[:, None]



class BatchCosineScore:
    def __init__(self, enroll, test, debug=False):
        # Donot do any sorting inside this class
        enroll_stat = {'x_avg': {}}
        enroll['X'] = lennorm(enroll['X'])
        test['X'] = lennorm(test['X'])

        # enroll_stat will be retrieved by key
        for uni_id in np.unique(enroll['spk_ids']):
            mask = enroll['spk_ids'] == uni_id
            enroll_stat['x_avg'][uni_id] = enroll['X'][mask].mean(0)
        # test_stat will be retrieved by mask
        # test['xtQx'] = row_wise_dot(test['X'], test['X'].dot(pq['Q']))
        # test['Px'] = test['X'].dot(pq['P'])

        self.enroll_stat = enroll_stat
        self.test_stat = sort_dict(test, 'spk_ids')

        self.debug = debug

    def score(self, ndx):
        # warnings.warn('ndx has to be sorted other wise batch score will fail')
        # Todo check whether ndx is sorted
        # ndx = ndx.sort_values(['enroll', 'test'])
        scores = []
        for enroll_id, test_ids in ndx.groupby('enroll').test:
            x_avg = self.enroll_stat['x_avg'][enroll_id]
            mask = pd.Series(self.test_stat['spk_ids']).isin(test_ids)

            # ndx need to be sort enroll first and test second
            # so the test_ids is in order the test file will aligh with ndx
            if self.debug:
                assert np.all(self.test_stat['spk_ids'][mask] == test_ids)

            score = self.test_stat['X'][mask] @ x_avg.T

            scores.append(score)
        scores = np.concatenate(scores)
        return scores


class BatchPLDAScore:
    def __init__(self, enroll, test, pq, debug=False):
        # Donot do any sorting inside this class
        enroll['xtQx'] = row_wise_dot(enroll['X'], enroll['X'].dot(pq['Q']))
        enroll_stat = {'x_avg': {}, 'xtQx_avg': {}}

        # enroll_stat will be retrieved by key
        for uni_id in np.unique(enroll['spk_ids']):
            mask = enroll['spk_ids'] == uni_id
            enroll_stat['x_avg'][uni_id] = enroll['X'][mask].mean(0)
            enroll_stat['xtQx_avg'][uni_id] = enroll['xtQx'][mask].mean(0)

        # test_stat will be retrieved by mask
        test['xtQx'] = row_wise_dot(test['X'], test['X'].dot(pq['Q']))
        test['Px'] = test['X'].dot(pq['P'])

        self.enroll_stat = enroll_stat
        self.test_stat = sort_dict(test, 'spk_ids')

        self.const = pq['const']
        self.debug = debug

    def score(self, ndx):
        warnings.warn('ndx has to be sorted other wise batch score will fail')
        # Todo check whether ndx is sorted
        # ndx = ndx.sort_values(['enroll', 'test'])
        scores = []
        for enroll_id, test_ids in ndx.groupby('enroll').test:
            x_avg = self.enroll_stat['x_avg'][enroll_id]
            mask = pd.Series(self.test_stat['spk_ids']).isin(test_ids)

            # ndx need to be sort enroll first and test second
            # so the test_ids is in order the test file will aligh with ndx
            if self.debug:
                assert np.all(self.test_stat['spk_ids'][mask] == test_ids)

            xtPx = self.test_stat['Px'][mask] @ x_avg.T
            score = 0.5 * self.enroll_stat['xtQx_avg'][enroll_id] \
                    + xtPx + 0.5 * self.test_stat['xtQx'][mask] + self.const

            scores.append(score)
        scores = np.concatenate(scores)
        return scores


def row_wise_dot(x, y):
    return np.einsum('ij,ij->i', x, y)



def sort_dict(my_dict, field):
    # Warning this is totally wrong
    idx = np.argsort(my_dict[field])
    for key, val in my_dict.items():
        my_dict[key] = val[idx]
    return my_dict


def build_id_dict(X, spk_ids):
    return {spk_id: X[spk_ids == spk_id] for spk_id in np.unique(spk_ids)}


def _plda_score_scores_averge(X, y, P, Q, const):
    return (
        0.5 * ravel_dot(X.T @ X, Q) / X.shape[0]
        + kernel_dot(y, P, X.sum(0).T) / X.shape[0]
        + 0.5 * kernel_dot(y, Q, y.T)
        + const
    )


def ravel_dot(X, Y):
    return X.ravel() @ Y.ravel()


def kernel_dot(x, kernel, y):
    return x @ kernel @ y


class ScoreNormalizer:
    def __init__(self, enroll, test, cohort, pq=None, top_scores=200):
        self.enroll = enroll
        self.test = test
        self.cohort = cohort
        self.pq = pq
        self.top_scores = top_scores

    def normalize_scores(self, ndx):
        znorm = self._normalize_scores('enroll', ndx)
        snorm = self._normalize_scores('test', ndx)
        return (znorm + snorm) / 2

    def _normalize_scores(self, enroll_or_test, ndx):
        enroll = getattr(self, enroll_or_test)
        if 'group' in self.cohort.keys():
            print('using group info')
            stats = []
            for group in np.unique(self.cohort['group']):
                mask = enroll['group'] == group
                enroll_masked = mask_dict(enroll, mask)
                mask = self.cohort['group'] == group
                cohort_masked = mask_dict(self.cohort, mask)
                stats.append(self.get_norm_stat(enroll_masked, cohort_masked))
            stats = pd.concat(stats)
        else:
            stats = self.get_norm_stat(enroll, self.cohort)

        scores_normed = []
        for enroll_id, scores in ndx.groupby(enroll_or_test).scores:
            temp = (scores - stats.loc[enroll_id]['mean']) / stats.loc[enroll_id]['std']
            scores_normed.append(temp)
        # Todo this score may not aligh with ndx
        scores_normed = pd.concat(scores_normed).sort_index()
        return scores_normed

    def get_norm_stat(self, enroll, cohort):
        trial = array_product(np.unique(enroll['spk_ids']),
                              np.unique(cohort['spk_ids']))
        ndx = pd.DataFrame(data=trial, columns=['enroll', 'test'])
        ndx = ndx.sort_values(['enroll', 'test'])
        # Todo call batch PLDA Score need ndx sorted
        if self.pq:
            ndx['scores'] = BatchPLDAScore(enroll, cohort, self.pq).score(ndx)
        else:
            ndx['scores'] = BatchCosineScore(enroll, cohort).score(ndx)
        stat = ndx.groupby('enroll').scores.apply(get_first_n, self.top_scores)
        return stat


def get_first_n(group, n):
    df = group.sort_values(ascending=False).iloc[:n]
    return df.agg([np.mean, np.std])


def array_product(x, y):
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


def mask_dict(dict_, mask):
    return {key: val[mask] for key, val in dict_.items()}


def comp_eer(scores, labels):
    fnr, fpr = _compute_pmiss_pfa(scores, labels)
    eer = _comp_eer(fnr, fpr)
    return eer * 100


def compute_c_norm(scores, labels, p_target=0.01, c_miss=1, c_fa=1):
    fnr, fpr = _compute_pmiss_pfa(scores, labels)
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_det, c_det_ind = min(dcf), np.argmin(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    return c_det/c_def


def _comp_eer(fnr, fpr):
    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = ( fnr[x1] - fpr[x1] ) / ( fpr[x2] - fpr[x1] - ( fnr[x2] - fnr[x1] ) )

    return fnr[x1] + a * ( fnr[x2] - fnr[x1] )


def _compute_pmiss_pfa(scores, labels):
    tgt_scores = scores[labels == 1] # target trial scores
    imp_scores = scores[labels == 0] # impostor trial scores

    resol = max([np.count_nonzero(labels == 0), np.count_nonzero(labels == 1), 1.e6])
    edges = np.linspace(np.min(scores), np.max(scores), int(resol))

    fnr = _compute_norm_counts(tgt_scores, edges, )
    fpr = 1 - _compute_norm_counts(imp_scores, edges, )

    return fnr, fpr


def _compute_norm_counts(scores, edges):
    score_counts = np.histogram(scores, bins=edges)[0].astype('f')
    norm_counts = np.cumsum(score_counts)/score_counts.sum()
    return norm_counts



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
        elif train_egs_dir:
            self.file_lst = glob.glob(f'{train_egs_dir}/*.h5')
            with open(f"{train_egs_dir}/egs_maker.yaml", 'r') as f:
                model_config['n_classes'] = yaml.safe_load(f)['n_classes']
        else:
            raise NotImplementedError

        self.model_config = model_config
        print(f"total num of spk is {model_config['n_classes']}")
        # self.model = model(**model_config)

        if checkpoint:
            print(f"loading from {checkpoint}")
        else:
            # breakpoint()
            model_config_copy = model_config.copy()
            if issubclass(type(model_config_copy ['model']), torch.nn.Module):
                del model_config_copy ['model']
            with open(f"{project_dir}/models/model_config.yaml", "w") as f:
                yaml.dump(model_config_copy, f, default_flow_style=False)

        # self.model_config = model_config
        self.model = model

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

        #
        # self.model = self.model.to(self.device)

        self.gpu_id = gpu_id
        # self.init_model(gpu_id)
        self.init_dist(rank)
        self.batch_size = self.batch_size // self.world_size
        self.rank = rank
        t0 = time.time()
        # if self.train_file:
        #     self.train()
        # elif self.train_egs_dir:
        #     local_file_lst = np.array_split(self.file_lst, self.world_size)[rank].tolist()
        #     self.egs_train(local_file_lst)
        # else:
        #     raise NotImplementedError
        # if self.train_file:
        #     self.train()
        if self.train_egs_dir:
            local_file_lst = np.array_split(self.file_lst, self.world_size)[rank].tolist()
            self.egs_train(local_file_lst)
        else:
            self.train()
        if rank == 0:
            self.save_model(f"{self.save_checkpoint_to}_final")
        print(f"train total time is {time.time() - t0}")

    def single_train(self):
        if self.gpu_ids is None:
            self.gpu_ids = GPUtil.getAvailable(maxMemory=0.02, limit=8)
        gpu_ids = self.gpu_ids[0]
        self.init_model(gpu_ids)
        self.rank = 0
        self.train()
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
        score = Score(
            comp_minDCF=False,
            enroll=enroll_embedding,
            test=f"{self.project_dir}/tmp/test.h5",
            ndx_file=self.valid_trial_list,
            transforms=transform_lst,
            **self.score_paras,
        )
        eer = score.batch_cosine_score()

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
        # self.model = self.model.cuda()
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            # find_unused_parameters=True,
        )

    # def load_checkpoint(self, checkpoint):
    #     checkpoint = torch.load(
    #         checkpoint,
    #         map_location=torch.device('cpu')
    #     )
    #     self.model.load_state_dict(checkpoint["model_state_dict"])
    #     # self.model = self.model.cuda()
    #     self.init_optimizer()
    #
    #     self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #     self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    #     self.epoch = checkpoint["epoch"] + 1

    # def load_checkpoint(self, checkpoint):
    #     checkpoint = torch.load(
    #         checkpoint, map_location=torch.device(torch.cuda.current_device())
    #     )
    #     self.model.load_state_dict(checkpoint["model_state_dict"])
    #     self.model = self.model.cuda()
    #     self.init_optimizer()
    #
    #     self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #     self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    #     self.epoch = checkpoint["epoch"] + 1

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
        # save_checkpoint_to = f"{save_checkpoint_to}.tar"
        torch.save(
            {
                "model_state_dict": self.model.state_dict() if self.world_size == 1 else self.model.module.state_dict(),
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


import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
import h5py
import time
import pandas as pd
import random


# class OuterDataset(Dataset):
#     def __init__(self, file_lst, logger, inner_batch_size, padding=False):
#         self.batch_size = inner_batch_size
#         self.logger = logger
#         if not padding:
#             self.file_lst = file_lst
#         else:
#             self.file_lst = np.tile(file_lst, 30).tolist()
#         # self.file_lst = np.tile(file_lst, 20).tolist()[:total_step]
#
#     def __getitem__(self, idx):
#         t0 = time.time()
#         with h5py.File(self.file_lst[idx], 'r') as f:
#             data = {
#                 'spk_ids': f['spk_ids'][:],
#                 'utt_ids': f['utt_ids'][:],
#                 'mfcc': f['mfcc'][:],
#             }
#         with open(self.logger, 'a') as f:
#             f.write(f'disk time is {time.time()-t0}\n')
#         inner = NewInnerDataset(data, self.batch_size)
#         inner_loader = torch.etc.data.DataLoader(inner,
#                                                    batch_size=1,
#                                                    collate_fn=unpack_list,
#                                                    num_workers=1)
#         return inner_loader
#
#     def __len__(self):
#         return len(self.file_lst)
#
#
# class NewInnerDataset(Dataset):
#     def __init__(self, data, batch_size):
#         assert type(data) in [dict, h5py._hl.files.File]
#         n_subset = len(data['mfcc']) // batch_size + 1
#         self.mfccs = np.array_split(data['mfcc'], n_subset)
#         self.spk_ids = np.array_split(data['spk_ids'], n_subset)
#         self.utt_ids = np.array_split(data['utt_ids'], n_subset)
#
#     def __getitem__(self, idx):
#         spk_id = self.spk_ids[idx]
#         utt_id = self.utt_ids[idx]
#         # 16X400X23
#         mfcc = self.mfccs[idx].transpose(0, 2, 1)
#         # mfcc, spk_ids = mfcc.cuda(), spk_ids.cuda()
#         # mfcc, spk_ids = mfcc.to(self.device), spk_ids.cuda(self.device)
#         return torch.tensor(mfcc), torch.tensor(spk_id), utt_id
#
#     def __len__(self):
#         return len(self.utt_ids)


def unpack_list(x):
    return x[0]

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


class InnerDataset(Dataset):
    def __init__(self, data):
        assert type(data) in [dict, h5py._hl.files.File]
        self.mfccs = data['mfcc']
        self.positions = data['positions'][:]
        self.spk_ids = data['spk_ids'][:]
        self.utt_ids = data['utt_ids'][:]

    def __getitem__(self, idx):
        start, end = self.positions[idx]
        spk_id = self.spk_ids[idx]
        utt_id = self.utt_ids[idx]
        mfcc = self.mfccs[start:end].T
        return torch.tensor(mfcc), torch.tensor(spk_id), utt_id

    def __len__(self):
        return len(self.positions)


import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class RandomSampleDataset(Dataset):
    def __init__(self, f, sample_per_epoch,
                 min_frames_per_utt, balance_class, min_utts_per_spk=8,):
        self.df = self.get_df(f, min_utts_per_spk, min_frames_per_utt)
        self.smp_per_epo = sample_per_epoch
        self.mfccs = f['mfcc']
        self. balance_class = balance_class

    def sample(self, sample_length):
        np.random.seed()
        if self.balance_class:
            prob = (1 / self.df['spk_ids'].nunique() / self.df['n_utts']).values
            idxes = np.random.choice(len(self.df), p=prob, size=self.smp_per_epo)
            self.spk_ids, self.utt_ids, self.positions = \
                self._sample(self.df.iloc[idxes], sample_length)
        else:
            idxes = np.random.randint(len(self.df), size=self.smp_per_epo)
            self.spk_ids, self.utt_ids, self.positions = \
                self._sample(self.df.iloc[idxes], sample_length)

    def __getitem__(self, i):
        spk_id = self.spk_ids[i]
        utt_id = self.utt_ids[i]
        start, end = self.positions[i]

        mfcc = self.mfccs[start:end].T
        return torch.tensor(mfcc), torch.tensor(spk_id),  torch.tensor(utt_id)

    def __len__(self):
        return self.smp_per_epo

    @staticmethod
    def _sample(df_smp, sample_length):
        spk_ids = df_smp.spk_ids.values
        utt_ids = df_smp.utt_ids.values
        positions = []
        for start, end in df_smp[['starts', 'ends']].values:
            smp_start = np.random.randint(low=start, high=end-sample_length)
            smp_end = smp_start + sample_length
            positions.append([smp_start, smp_end])
        positions = np.array(positions)
        return spk_ids, utt_ids, positions

    @staticmethod
    def get_df(f, min_utts_per_spk, min_frames_per_utt):
        df = pd.DataFrame({'spk_ids': f['spk_ids'][:],
                           'utt_ids': f['utt_ids'][:],
                           'starts': f['positions'][:, 0],
                           'ends': f['positions'][:, 1]})
        df = df[(df.ends - df.starts) > min_frames_per_utt]
        utt_counts = df.spk_ids.value_counts()
        df['n_utts'] = df.spk_ids.map(utt_counts)
        df = df[df.n_utts > min_utts_per_spk]
        df['spk_ids'] = LabelEncoder().fit_transform(df['spk_ids'])
        df['utt_ids'] = LabelEncoder().fit_transform(df['utt_ids'])
        return df


class ExtractDataset(Dataset):
    def __init__(self, f, meta_data=None):
        self.mfccs = f['mfcc']
        if meta_data:
            with h5py.File(meta_data, 'r') as f_meta:
                self.positions = f_meta['positions'][:]
                self.spk_ids = f_meta['spk_ids'][:]
                self.utt_ids = f_meta['utt_ids'][:]
        else:
            self.positions = f['positions'][:]
            self.spk_ids = f['spk_ids'][:]
            self.utt_ids = f['utt_ids'][:]

    def __getitem__(self, idx):
        start, end = self.positions[idx]
        spk_id = self.spk_ids[idx]
        utt_id = self.utt_ids[idx]
        mfcc = self.mfccs[start:end].T
        return torch.tensor(mfcc), spk_id, utt_id

    def __len__(self):
        return len(self.positions)


class PartialExtractDataset(Dataset):
    def __init__(self, df, mfcc):
        self.mfccs = mfcc
        self.positions = df[['starts', 'ends']].values
        self.spk_ids = df['spk_ids'].values
        self.utt_ids = df['utt_ids'].values

    def __getitem__(self, idx):
        start, end = self.positions[idx]
        spk_id = self.spk_ids[idx]
        utt_id = self.utt_ids[idx]
        mfcc = self.mfccs[start:end].T
        return torch.tensor(mfcc), spk_id, utt_id

    def __len__(self):
        return len(self.positions)


class NestedPartialExtractDataset(Dataset):
    def __init__(self, df, mfcc, chunk=50):
        self.mfccs = mfcc
        positions = df[['starts', 'ends']].values
        n_subset = len(positions) // chunk + 1
        self.positions = np.array_split(positions, n_subset)

        self.spk_ids = np.array_split(df['spk_ids'].values, n_subset)
        self.utt_ids = np.array_split(df['utt_ids'].values, n_subset)
        self.chunk = chunk

    def __getitem__(self, idx):
        position = self.positions[idx]
        start, end = position.min(), position.max()
        mfcc = self.mfccs[start:end]

        data = {
            'positions': position-position.min(),
            'spk_ids': self.spk_ids[idx],
            'utt_ids': self.utt_ids[idx],
            'mfcc': mfcc,
        }

        inner = InnerDataset(data)
        inner_loader = torch.utils.data.DataLoader(inner,
                                                   batch_size=1)
        return inner_loader

    def __len__(self):
        return len(self.positions)



def extract_collate(batch):
    assert len(batch) == 1
    x = batch[0][0]
    spk_id = batch[0][1]
    utt_id = batch[0][2]
    return [x[None, ...], spk_id, utt_id]


import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class RandomSampleDataset(Dataset):
    def __init__(
        self,
        f,
        sample_length_range,
        n_blocks,
        sample_per_epoch,
        min_frames_per_utt,
        meta_data_file=None,
        balance_class=True,
        min_utts_per_spk=8,
    ):
        self.n_blocks = n_blocks
        if meta_data_file:
            with h5py.File(meta_data_file, 'r') as f_meta:
                self.df = self.get_df(f_meta, min_utts_per_spk, min_frames_per_utt)
        else:
            self.df = self.get_df(f, min_utts_per_spk, min_frames_per_utt)

        self.smp_per_epo = sample_per_epoch
        self.mfccs = f["mfcc"]
        self.balance_class = balance_class
        self.smp_len_min, self.smp_len_max = sample_length_range

    def sample(self):
        np.random.seed()
        sample_length = np.random.randint(low=self.smp_len_min, high=self.smp_len_max)
        if self.balance_class:
            prob = (1 / self.df["spk_ids"].nunique() / self.df["n_utts"]).values
            idxes = np.random.choice(len(self.df), p=prob, size=self.smp_per_epo)
            self.spk_ids, self.utt_ids, self.positions = self.sample_segments(
                self.df.iloc[idxes], sample_length
            )
        else:
            idxes = np.random.randint(len(self.df), size=self.smp_per_epo)
            self.spk_ids, self.utt_ids, self.positions = self.sample_segments(
                self.df.iloc[idxes], sample_length
            )

    def sample_original(self):
        np.random.seed()
        sample_length = np.random.randint(low=self.smp_len_min, high=self.smp_len_max)
        if self.balance_class:
            # remove augment data
            mask = ~self.df.utt_ids.str.endswith(('babble', 'music', 'noise', 'reverb'))
            df = self.df[mask]
            utt_counts = df.spk_ids.value_counts()
            df["n_utts"] = df.spk_ids.map(utt_counts)

            prob = (1 / df["spk_ids"].nunique() / df["n_utts"]).values
            idxes = np.random.choice(len(df), p=prob, size=self.smp_per_epo)
            self.spk_ids, self.utt_ids, self.positions = self.sample_segments(
                df.iloc[idxes], sample_length
            )
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        spk_id = self.spk_ids[i]
        utt_id = self.utt_ids[i]
        sub_positions = self.positions[i]
        mfccs = []
        for start, end in sub_positions:
            mfccs.append(self.mfccs[start:end])
        mfccs = np.concatenate(mfccs).T
        return torch.tensor(mfccs), torch.tensor(spk_id), utt_id

    def __len__(self):
        return self.smp_per_epo

    def sample_segments(self, df_smp, sample_length):
        spk_ids = df_smp.spk_ids.values
        utt_ids = df_smp.utt_ids.values
        positions = []
        for start, end in df_smp[["starts", "ends"]].values:
            sub_positions = []
            for _ in range(self.n_blocks):
                smp_start = np.random.randint(low=start, high=end - sample_length)
                smp_end = smp_start + sample_length
                sub_positions.append([smp_start, smp_end])
            positions.append(sub_positions)
        positions = np.array(positions)
        return spk_ids, utt_ids, positions

    @staticmethod
    def get_df(f, min_utts_per_spk, min_frames_per_utt):
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
        df["spk_ids"] = LabelEncoder().fit_transform(df["spk_ids"])
        # df['utt_ids'] = LabelEncoder().fit_transform(df['utt_ids'])
        return df


import h5py
import numpy as np
import scipy.io as sio


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
        raise NotImplemented


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


import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import scipy.linalg as la


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
        for step, (data_src, data_tgt, data_tgt_aug) in enumerate(zip(train_loader, target_loader, target_loader_aug)):
            self.optimizer.zero_grad()
            mfcc_src, spk_ids_src, _ = data_src
            mfcc_tgt, spk_ids_tgt, _ = data_tgt
            mfcc_tgt_aug, spk_ids_tgt_aug, _ = data_tgt_aug

            mfcc_src, spk_ids_src = mfcc_src.cuda(), spk_ids_src.cuda()
            mfcc_tgt = mfcc_tgt.cuda()
            mfcc_tgt_aug = mfcc_tgt_aug.cuda()
            # for source
            frames_src = self.model.module.frame_layers(mfcc_src)  # 128X1500X400
            stats_src = self.model.module.stat_pooling(frames_src)
            embed_src = self.model.module.utt_layers(stats_src)
            logit_src, logit_nomargin_src = self.model.module.am_linear(embed_src, spk_ids_src)

            # for target
            frames_tgt = self.model.module.frame_layers(mfcc_tgt)
            stats_tgt = self.model.module.stat_pooling(frames_tgt)
            embed_tgt = self.model.module.utt_layers(stats_tgt)

            frames_tgt_aug = self.model.module.frame_layers(mfcc_tgt_aug)
            stats_tgt_aug = self.model.module.stat_pooling(frames_tgt_aug)
            embed_tgt_aug = self.model.module.utt_layers(stats_tgt_aug)

            cls_loss = F.cross_entropy(logit_src, spk_ids_src)
            # domain_loss = self.domain_loss_weight * self.unsup_loss(embed_tgt, embed_src)
            # X 64X1500X400
            sample_idx = np.random.randint(frames_tgt.shape[0] * frames_tgt.shape[2] - 3000)
            loss2 = self.unsup_loss(frames_src.permute(0, 2, 1).reshape(-1, 1500)[sample_idx:sample_idx + 3000],
                                    frames_tgt.permute(0, 2, 1).reshape(-1, 1500)[sample_idx:sample_idx + 3000])

            domain_loss = self.domain_loss_weight * self.unsup_loss(embed_tgt, embed_src) + loss2 + self.unsup_loss(embed_tgt, embed_tgt_aug)
            loss = cls_loss + domain_loss

            loss.backward()
            self.optimizer.step()

            acc = logit_nomargin_src.max(-1)[1].eq(spk_ids_src).sum().item() / len(spk_ids_src) * 100
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


