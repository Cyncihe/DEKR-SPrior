import os

from lib.dataset.COCODataset import get_soybean_dataset, loadRes
from lib.dataset.beaneval import BEANeval

SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def _do_python_keypoint_eval(detection_file):
    root = os.path.join(SOURCE_DIR, 'data')
    dir_path = os.path.join(root, 'bean/images/oks_test')

    img_paths, ann_paths = get_soybean_dataset(dir_path)

    beanGt = ann_paths  # load annotations
    beanDt = loadRes(detection_file)  # load model outputs

    # running evaluation
    beanEval = BEANeval(beanGt, beanDt, mode='nbBean')  # mode: nbBean or area
    beanEval.evaluate()
    beanEval.accumulate()
    beanEval.summarize()

    precisions = beanEval.eval['precision']

    pr_array1 = precisions[0, :, 0, 0, 0]
    print('OKS=0.50:\n', pr_array1)


det_file_path = os.path.join(SOURCE_DIR, 'lib/dataset/keypoints_oksregression_results.json')
_do_python_keypoint_eval(det_file_path)