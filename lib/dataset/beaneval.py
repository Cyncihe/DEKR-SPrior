__author__ = 'tsungyi'

import json
import logging
import os
import sys

import numpy as np
import datetime
import time
from collections import defaultdict
import copy

class BEANeval:
    def __init__(self, beanGt=None, beanDt=None, mode='area'):

        self.beanGt = beanGt  # ground truth COCO API
        self.beanDt = beanDt  # detections COCO API
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params()  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        self.bean_ae = []
        self.pod_ae = []
        self.area = []
        self.gt_production = []
        self.dt_production = []
        self.gt_production_pod = []
        self.dt_production_pod = []
        self.acc_production = 0

        assert mode == 'area' or mode == 'nbBean', "invalid mode"
        self.mode = mode
        print('******* current mode:', self.mode)

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """
        gts = self.beanGt  # a list of ann paths
        dts = self.beanDt  # detection json
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            f = open(gt)
            anns = json.load(f)
            anns = normalize_annotations(anns)  # reformat annotations
            for pod in anns['annotations']:
                pod['keypoints'] = list(pod['keypoints'].flatten())
                pod['ignore'] = False
                pod['iscrowd'] = False

            from lib.dataset.COCODataset import add_bbox_info
            anns['annotations'] = add_bbox_info(anns['annotations'])
            self._gts[anns['image_name']] = anns['annotations']
            self.params.img_names.append(anns['image_name'])

        for dt in dts:
            img_name = dt.pop('image_name')
            self._dts[img_name].append(dt)

        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        self.ious = {img_name: self.computeOks(img_name) for img_name in p.img_names}

        for img_name in p.img_names:
            mae_pod, mae_bean, gts_bean_cpt, dts_bean_cpt, gts_pod_cpt, dts_pod_cpt = self.computeAe(img_name)
            self.pod_ae.append(mae_pod)
            self.bean_ae.append(mae_bean)
            self.gt_production.append(gts_bean_cpt)
            self.dt_production.append(dts_bean_cpt)
            self.gt_production_pod.append(gts_pod_cpt)
            self.dt_production_pod.append(dts_pod_cpt)
        self.acc_production = 1 - abs(sum(self.dt_production) - sum(self.gt_production)) / (sum(self.gt_production) + 1e-10)
        self.area = [self.getAvgArea(img_name) for img_name in p.img_names]
        print('avg area:', np.mean(self.area))

        print('gt_bean:', self.gt_production)
        print('gt_pod:', self.gt_production_pod)
        print('dt_bean:', self.dt_production)
        print('dt_pod:', self.dt_production_pod)

        # self.pod_ae = [ae[0] for ae in aes]
        # self.bean_ae = [ae[1] for ae in aes]
        # self.gt_production = [pro[0] for pro in production]
        # self.dt_production = [pro[1] for pro in production]
        # self.acc_production = 1 - abs(sum(self.dt_production) - sum(self.gt_production)) / sum(self.gt_production)

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        # self.evalImgs = [evaluateImg(img_name, areaRng, maxDet, None)
        #                  for areaRng in p.areaRng
        #                  for img_name in p.img_names]

        self.evalImgs = [evaluateImg(img_name, None, maxDet, nb_bean)
                         for nb_bean in p.nbBeans
                         for img_name in p.img_names]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def getAvgArea(self, img_name):
        gts = self._gts[img_name]
        pod_cpt = 0
        area = 0
        for pod in gts:
            pod_cpt += 1
            area += pod['area']
        return area / pod_cpt

    def computeAe(self, img_name, thre=0.10):
        gts = self._gts[img_name]
        dts = self._dts[img_name]
        gts_pod_cpt, gts_bean_cpt = 0, 0
        dts_pod_cpt, dts_bean_cpt = 0, 0
        for pod in gts:
            gts_pod_cpt += 1
            gts_bean_cpt += pod['keypoints'][2::3].count(1.0)

        for pod in dts:
            if pod['score'] >= thre:
                dts_pod_cpt += 1
                dts_bean_cpt += (5 - pod['keypoints'][2::3].count(0.0))

        mae_pod = abs(gts_pod_cpt - dts_pod_cpt)
        mae_bean = abs(gts_bean_cpt - dts_bean_cpt)

        if mae_pod > 4:
            print(f'mae_pod: {mae_pod}, gt:{gts_pod_cpt}, dt:{dts_pod_cpt}', img_name)
        if mae_bean > 4:
            print(f'mae_bean: {mae_bean}, gt:{gts_bean_cpt}, dt:{dts_bean_cpt}', img_name)

        return mae_pod, mae_bean, gts_bean_cpt, dts_bean_cpt, gts_pod_cpt, dts_pod_cpt

    def computeOks(self, img_name):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[img_name]
        dts = self._dts[img_name]

        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xgo = g[0::3]
            ygo = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)  # the number of visible keypoints
            idx = np.nonzero(vg)[0][0]
            if k1 <= 0:
                raise Exception('No visible keypoints')
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                best_oks = -np.inf
                vd = d[2::3]
                xdo = d[0::3]
                ydo = d[1::3]
                k2 = np.count_nonzero(vd > 0)
                idx = np.nonzero(vd)[0][0]
                xd_nonzero = xdo[idx:idx + k2]
                yd_nonzero = ydo[idx:idx + k2]

                # # if k1 >= k2:
                # for v in range(k - k2 + 1):
                #     xd = np.concatenate((np.zeros(v), xd_nonzero, np.zeros(k - v - k2)), axis=None)
                #     yd = np.concatenate((np.zeros(v), yd_nonzero, np.zeros(k - v - k2)), axis=None)
                #
                #     # measure the per-keypoint distance if keypoints visible
                #     dx = xd - xgo
                #     dy = yd - ygo
                #
                #     e = (dx ** 2 + dy ** 2) / vars / (gt['area'] + np.spacing(1)) / 2
                #     if k1 > 0:
                #         e = e[vg > 0]
                #     oks = np.sum(np.exp(-e)) / e.shape[0]
                #     if oks > best_oks:
                #         best_oks = oks

                for v in range(k):
                    xd = np.zeros_like(xdo)
                    yd = np.zeros_like(ydo)
                    for m in range(k2):
                        xd[(v + m) % k] = xd_nonzero[m]
                        yd[(v + m) % k] = yd_nonzero[m]
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xgo
                    dy = yd - ygo

                    e = (dx ** 2 + dy ** 2) / vars / (
                            2 * p.lambdas[k1 - 1] * (gt['area'] + np.spacing(1)))
                    if k1 > 0:
                        e = e[vg > 0]
                    oks = np.sum(np.exp(-e)) / e.shape[0]
                    if oks > best_oks:
                        best_oks = oks

                # compute the oks between the ith pod of gt and the jth pod of dt
                ious[i, j] = best_oks
                print('best oks:', best_oks)
        return ious

    # 改进前的oks计算
    def computeOks2(self, img_name):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[img_name]
        dts = self._dts[img_name]

        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xgo = g[0::3]
            ygo = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)  # the number of visible keypoints
            if k1 <= 0:
                raise Exception('No visible keypoints')
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                best_oks = -np.inf
                vd = d[2::3]
                xdo = d[0::3]
                ydo = d[1::3]
                k2 = np.count_nonzero(vd > 0)
                idx = np.nonzero(vd)[0][0]

                # measure the per-keypoint distance if keypoints visible
                dx = xdo - xgo
                dy = ydo - ygo

                e = (dx ** 2 + dy ** 2) / vars / (
                        2 * p.lambdas[k1 - 1] * (gt['area'] + np.spacing(1)))
                if k1 > 0:
                    e = e[vg > 0]
                oks = np.sum(np.exp(-e)) / e.shape[0]
                if oks > best_oks:
                    best_oks = oks

                # compute the oks between the ith pod of gt and the jth pod of dt
                ious[i, j] = best_oks
                print('best oks:', best_oks)
        return ious

    def evaluateImg(self, img_name, aRng, maxDet, nbBean=None):
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
        if bool(aRng) == bool(nbBean):
            raise Exception('Please specify 1 and only 1 of the 2 arguments: aRng and nbBean')
        assert (bool(aRng) != bool(nbBean))
        p = self.params
        if p.useCats:
            gt = self._gts[img_name]
            dt = self._dts[img_name]
        else:
            gt = [_ for _ in self._gts[img_name]]
            dt = [_ for _ in self._dts[img_name]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        if aRng:
            filt = 'aRng'
            filt_value = aRng
            for g in gt:
                if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                    g['_ignore'] = 1
                else:
                    g['_ignore'] = 0
        if nbBean:
            filt = 'nbBean'
            filt_value = nbBean
            for g in gt:
                if sum(g['keypoints'][2::3]) < nbBean[0] or sum(g['keypoints'][2::3]) > nbBean[1]:
                    g['_ignore'] = 1
                else:
                    g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[img_name][:, gtind] if len(self.ious[img_name]) > 0 else self.ious[img_name]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made, store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = 1
                    gtm[tind, m] = 1
        # set unmatched detections outside of area range to ignore
        if aRng:
            a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
        else:
            a = np.array(
                [sum(d['keypoints'][2::3]) < nbBean[0] or sum(d['keypoints'][2::3]) > nbBean[1] for d in dt]).reshape(
                (1, len(dt)))

        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            'image_id': img_name,
            filt: filt_value,
            'maxDet': maxDet,
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = 1
        if self.mode == 'area':
            A = len(p.areaRng)
        elif self.mode == 'nbBean':
            A = len(p.nbBeans)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        if self.mode == 'area':
            setA = set(map(tuple, _pe.areaRng))
        elif self.mode == 'nbBean':
            setA = set(map(tuple, _pe.nbBeans))
        setM = set(_pe.maxDets)
        setI = set(_pe.img_names)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        if self.mode == 'area':
            a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        elif self.mode == 'nbBean':
            a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.nbBeans)) if a in setA]

        i_list = [n for n, i in enumerate(p.img_names) if i in setI]

        I0 = len(_pe.img_names)
        if self.mode == 'area':
            A0 = len(_pe.areaRng)
        elif self.mode == 'nbBean':
            A0 = len(_pe.nbBeans)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        # Find the index when each recall changes, one-to-one correspondence with p.recThrs,
                        # the index closest to its value
                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                # Get the maximum accuracy corresponding to each recall threshold and store it in q
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        # Store the corresponding precision according to the size of the recall
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
            'mae_pod': np.mean(self.pod_ae),
            'mae_bean': np.mean(self.bean_ae)
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this function can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng='all', nb_bean='all', maxDets=100):
            mode = self.mode
            if mode == 'area':
                assert nb_bean == 'all', "invalid argument: nb_bean"
            elif mode == 'nbBean':
                assert areaRng == 'all', "invalid argument: areaRng"
            else:
                raise Exception('invalid mode')

            p = self.params
            if mode == 'area':
                iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
                aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]

            elif mode == 'nbBean':
                iStr = ' {:<18} {} @[ IoU={:<9} | nbBean={:>6s} | maxDets={:>3d} ] = {:0.3f}'
                aind = [i for i, nBean in enumerate(p.nbBeansLbl) if nBean == nb_bean]

            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM] [iouThr, recThr, category, area, maxDet]
                s = self.eval['precision']  # (10, 101, 1, 3, 1)
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]  # (1, 101, 1, 3, 1)
                s = s[:, :, :, aind, mind]  # (10, 101, 1, 1) or (1, 101, 1, 1)
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            if mode == 'area':
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            elif mode == 'nbBean':
                print(iStr.format(titleStr, typeStr, iouStr, nb_bean, maxDets, mean_s))

            return mean_s

        def _summarizeKps():
            stats = np.zeros((12,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='small')
            stats[4] = _summarize(1, maxDets=20, areaRng='medium')
            stats[5] = _summarize(1, maxDets=20, areaRng='large')
            stats[6] = _summarize(0, maxDets=20)
            stats[7] = _summarize(0, maxDets=20, iouThr=.5)
            stats[8] = _summarize(0, maxDets=20, iouThr=.75)
            stats[9] = _summarize(0, maxDets=20, areaRng='small')
            stats[10] = _summarize(0, maxDets=20, areaRng='medium')
            stats[11] = _summarize(0, maxDets=20, areaRng='large')

            return stats

        def _summarizeKps_nbBean():
            stats = np.zeros((12,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, nb_bean='one')
            stats[4] = _summarize(1, maxDets=20, nb_bean='two')
            stats[5] = _summarize(1, maxDets=20, nb_bean='three')
            stats[5] = _summarize(1, maxDets=20, nb_bean='four')
            stats[5] = _summarize(1, maxDets=20, nb_bean='five')
            stats[6] = _summarize(0, maxDets=20)
            stats[7] = _summarize(0, maxDets=20, iouThr=.5)
            stats[8] = _summarize(0, maxDets=20, iouThr=.75)
            stats[9] = _summarize(0, maxDets=20, nb_bean='one')
            stats[10] = _summarize(0, maxDets=20, nb_bean='two')
            stats[11] = _summarize(0, maxDets=20, nb_bean='three')
            stats[11] = _summarize(0, maxDets=20, nb_bean='four')
            stats[11] = _summarize(0, maxDets=20, nb_bean='five')

            return stats

        def _summarizeMae():
            stats = np.zeros((2,))
            print('MAE for pod count:\t\t', self.eval['mae_pod'])
            print('MAE for bean count:\t\t', self.eval['mae_bean'])
            print('Production accuracy:\t\t', self.acc_production)

        if not self.eval:
            raise Exception('Please run accumulate() first')
        if self.mode == 'area':
            summarize = _summarizeKps
        elif self.mode == 'nbBean':
            summarize = _summarizeKps_nbBean
        self.stats = summarize()
        _summarizeMae()

    def __str__(self):
        self.summarize()


class Params:
    """
    Params for evaluation
    """

    def setKpParams(self):
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.img_names = []
        self.catIds = []
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]  # max detected pod number per image
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 0
        self.kpt_oks_sigmas = np.array([1, 1, 1, 1, 1]) * 0.07
        self.nbBeans = [[1, 5], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
        self.nbBeansLbl = ['all', 'one', 'two', 'three', 'four', 'five']
        self.lambdas = [4.75, 1.35, 1, 0.78, 0.67]

    def __init__(self):
        self.setKpParams()
        # useSegm is deprecated
        self.useSegm = None


def normalize_annotations(anns):
    anns = copy.deepcopy(anns)
    # convert as much data as possible to numpy arrays to avoid every float
    # being turned into its own torch.Tensor()

    del anns['imageData']
    shapes = anns['shapes']
    dic = {}
    for sh in shapes:
        item = copy.deepcopy(sh)
        del item['group_id']
        if 'shape_type' in item:
            del item['shape_type']
        if 'flags' in item:
            del item['flags']
        dic.setdefault(sh['group_id'], []).append(item)

    for k, v in dic.items():
        v.sort(key=lambda i: i['label'])

    beans = []
    MAX_BEAN_COUNT = 5
    for k, v in dic.items():

        if len(v) > MAX_BEAN_COUNT:
            logging.warning(f"pod with {len(v)} beans!! (group_id: {k})"
                            f"Please check if there is an error in the annotation file:\n{anns['imagePath']}")
        bean = {'group_id': k,
                'keypoints': np.asarray([item['points'][0] + [1] for item in v[:MAX_BEAN_COUNT]], dtype=np.float32),
                'unknown_count': v[0]['label'] == '0-0'}
        if len(bean['keypoints']) < 5:
            bean['keypoints'] = np.concatenate(
                (bean['keypoints'], np.array([[0, 0, 0]] * (5 - len(bean['keypoints'])))))
        # print('bean:', bean)
        beans.append(bean)
    return {
        'image_name': anns['imagePath'],
        'annotations': beans
    }

