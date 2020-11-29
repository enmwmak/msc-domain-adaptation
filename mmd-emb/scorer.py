import warnings
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Scorer:
    def __init__(self, enroll, test, ndx_file,
                 comp_minDCF=True,
                 score_using_spk_ids=False,
                 average_by=None,
                 top_scores=200,
                 preserve_trial_order=False,
                 data_source="cmn2",
                 cohort=None, transforms=None, group_infos=None,
                 blind_trial=False, save_scores_to=None):

        self.data_source = data_source              # Either cmn2 or vast for SRE16-18
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
            print('Average enrollment xvectors by spk_ids')
        elif average_by == 'utt_ids':
            self.enroll = average_xvec(self.enroll, 'spk_path')
            self.test = average_xvec(self.test, 'spk_path')
            print('Average enrollment and test xvectors by utt_ids')
        else:
            print('No xvector averaging')

        # Only works if enroll.h5 and test.h5 have 'spk_path' field
        if score_using_spk_ids:
            # self.test['spk_ids'] = self.test['spk_path']
            self.enroll['spk_path'] = self.enroll['spk_ids']
        else:
            self.enroll['spk_ids'] = self.enroll['spk_path']
        # self.test['spk_ids'] = self.test['spk_path']
        self.test['spk_ids'] = self.test['spk_path']
        # breakpoint()
        if cohort:
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
                            usecols=['modelid', 'segmentid', 'targettype', 'data_source'])
                  .rename(columns={'modelid': 'enroll', 'segmentid': 'test', 'targettype': 'label'})
            )
            self.ndx['label'] = self.ndx.label.map({'target': 1, 'nontarget': 0})
            self.ndx = self.ndx[self.ndx['data_source'] == self.data_source]
            if preserve_trial_order:
                self.ndx = self.ndx.groupby(['enroll', 'test']).apply(lambda x: x.index.values).reset_index().rename(
                    {0: 'dup_index'}, axis=1)
            else:
                self.ndx = self.ndx.sort_values(by=['enroll', 'test'])
        else:
            # self.ndx = pd.read_csv(ndx_file, sep='\t', usecols=[0, 1], names=['enroll', 'test'])
            self.ndx = (
                pd.read_csv(ndx_file, sep='\t',   dtype=str,
                            usecols=['modelid', 'segmentid', 'data_source'])
                .rename(columns={'modelid': 'enroll', 'segmentid': 'test'})
            )
            self.ndx = self.ndx[self.ndx['data_source'] == self.data_source]
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
        raise NotImplementedError


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
            #print(f"enroll_id is {enroll_id}")
            #print(list(self.enroll_stat['x_avg']))
            
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
    c_det, _ = min(dcf), np.argmin(dcf)
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




