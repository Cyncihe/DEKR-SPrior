from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import cv2
import matplotlib.pyplot as plt
from color import colors, draw_pods
from beanmodel import BeanModel
import numpy as np
from dtw import dtw

def range_filter(location, joints, img_size, dist_ration=5e-3, shape_ration=5e-3, pad_pixels = 20):
    w, h = img_size
    # idxs = np.where(joints_filter[:,:,2] > 0)
    # dist_thresh = max(w, h) * dist_ration
    # shape_thresh = max(w, h) * shape_ration
    dist_thresh = 20
    shape_thresh = 50
    joints_filter = joints.copy()
    location = np.array(location)

    # left_min = max(location[:,0] - pad_pixels, 0)
    # left_max = min(location[:,0] + pad_pixels, w)
    # right_min = max(location[:,2] - pad_pixels, 0)
    # right_max = min(location[:,2] + pad_pixels, w)
    # top_min = max(location[:,1] - pad_pixels, 0)
    # top_max = min(location[:,1] + pad_pixels, h)
    # down_min = max(location[:,3] - pad_pixels, 0)
    # down_max = min(location[:,3] + pad_pixels, h)
    # del_idx = []
    for i in range(len(location)):
        left_min = max(location[i, 0] - pad_pixels, 0)
        left_max = min(location[i, 0] + pad_pixels, w)
        right_min = max(location[i, 2] - pad_pixels, 0)
        right_max = min(location[i, 2] + pad_pixels, w)
        top_min = max(location[i, 1] - pad_pixels, 0)
        top_max = min(location[i, 1] + pad_pixels, h)
        down_min = max(location[i, 3] - pad_pixels, 0)
        down_max = min(location[i, 3] + pad_pixels, h)

        peaks_left = (joints_filter[:, :, 2] > 0) & \
                     (joints_filter[:, :, 0] > left_min) & \
                     (joints_filter[:, :, 0] < left_max) & \
                     (joints_filter[:, :, 1] > location[i, 1]) & \
                     (joints_filter[:, :, 1] < location[i, 3])
        peaks_right = (joints_filter[:, :, 2] > 0) & \
                     (joints_filter[:, :, 0] > right_min) & \
                     (joints_filter[:, :, 0] < right_max) & \
                     (joints_filter[:, :, 1] > location[i, 1]) & \
                     (joints_filter[:, :, 1] < location[i, 3])
        peaks_top = (joints_filter[:, :, 2] > 0) & \
                     (joints_filter[:, :, 1] > top_min) & \
                     (joints_filter[:, :, 1] < top_max) & \
                     (joints_filter[:, :, 0] > location[i, 0]) & \
                     (joints_filter[:, :, 0] < location[i, 2])
        peaks_down = (joints_filter[:, :, 2] > 0) & \
                     (joints_filter[:, :, 1] > down_min) & \
                     (joints_filter[:, :, 1] < down_max) & \
                     (joints_filter[:, :, 0] > location[i, 0]) & \
                     (joints_filter[:, :, 0] < location[i, 2])

        for pk in [peaks_left,peaks_right,peaks_top,peaks_down]:
            nonzero_pods = np.nonzero(pk)[0]
            if len(nonzero_pods) > 1:
                nonzero_pods = list(set(nonzero_pods))
                pick_pods = joints_filter[nonzero_pods]
                seeds_idx = np.nonzero(pick_pods[:, :, 2] > 0)
                seeds_loc = np.array([[pick_pods[i,j,0],pick_pods[i,j,1]] for i,j in zip(seeds_idx[0],seeds_idx[1])])

                num = len(seeds_loc)
                a = np.broadcast_to(seeds_loc[None], (num, num, 2))
                vec_raw = (seeds_loc[:, None, :] - a).reshape(-1, 1, 2)
                vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)

                relate_matrix = vec_norm.reshape((num,num))
                dist_keeps = [(i,j) for i in range(num) for j in range(i+1,num,1) if relate_matrix[i,j] < dist_thresh]
                # keep_pods = [(seeds_idx[0][i],seeds_idx[0][j]) for i,j in dist_keeps if seeds_idx[0][i]!= seeds_idx[0][j]]
                for i,j in dist_keeps:
                    #如果豆粒属于不同的豆荚，则进行处理
                    if seeds_idx[0][i] != seeds_idx[0][j]:
                        idx_i = nonzero_pods[seeds_idx[0][i]]
                        idx_j = nonzero_pods[seeds_idx[0][j]]
                        pod_i = joints_filter[idx_i]
                        pod_j = joints_filter[idx_j]
                        seeds_i = np.nonzero(pod_i[:, 2] > 0)[0]
                        seeds_j = np.nonzero(pod_j[:, 2] > 0)[0]
                        # seeds_num_i = len()
                        # seeds_num_j = len()

                        #如果该粒豆荚只有一个点，则直接将该点合并
                        if len(seeds_i) < 2:
                            # del_idx.append([idx_i,0])
                            joints_filter[idx_i][0,2] = 0
                        elif len(seeds_j) < 2:
                            joints_filter[idx_j][0,2] = 0
                        # 判断两粒豆荚之间在形态上的相似性
                        else:
                            ds = dtw(pod_i[seeds_i][:,:2], pod_j[seeds_j][:,:2], keep_internals=True, step_pattern='asymmetric',
                                     open_begin=True, open_end=True)
                            # ds1 = dtw(a_array, b_array, keep_internals=True, step_pattern='asymmetric')
                            # ds.plot(type="twoway",offset=2)
                            #如果在形态上距离较近，则进行合并
                            if ds.distance < shape_thresh:
                                #连接关系即为合并组合
                                visited = []
                                for x,y in zip(ds.index1,ds.index2):
                                    joints_filter[idx_i][x,:2] = \
                                        (joints_filter[idx_i][x,:2] + joints_filter[idx_j][y,:2]) / 2.0
                                    joints_filter[idx_j][y,2] = 0
                                    visited.append(y)
                                # for y in seeds_j:
                                #     if y not in visited:

                # for i, pi, in enumerate(pick_pods):
                #     color = colors(i, bgr=False)
                #     start_p = ()
                #     for x, y, s in pi:
                #         if s < 1e-3:
                #             continue
                #         if len(start_p) > 1:
                #             cv2.line(img, start_p, (int(x), int(y)), color, 3)
                #         start_p = (int(x), int(y))
                #         cv2.circle(img, (int(x), int(y)), 6, color, -1)
                # plt.figure(figsize=(40, 40))
                # plt.imshow(img)
                # plt.show()




    # del joints
    return joints_filter


def pods_counting(joints_filter):
    pod_dict = {}
    if len(joints_filter) > 0:
        mask = joints_filter[:,:,2] > 0
        sum_counting = np.sum(mask,axis=1)
        for i in range(1,6,1):
            pod_dict[i] = np.sum(sum_counting == i)
    return pod_dict

def show_result(joints_show, img_rgb):
    plt.figure(figsize=(40, 40))
    for i, pi, in enumerate(joints_show):
        color = colors(i, bgr=False)
        start_p = ()
        for x, y, s in pi:
            if s < 1e-3:
                continue
            if len(start_p) > 1:
                cv2.line(img_rgb, start_p, (int(x), int(y)), color, 3)
            start_p = (int(x), int(y))
            cv2.circle(img_rgb, (int(x), int(y)), 6, color, -1)
    plt.imshow(img_rgb)
    plt.show()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "6,7"
    print("Use GPU: {} for training".format(os.environ['CUDA_VISIBLE_DEVICES']))
    # 最开始的最佳model
    model_path = '/workspace/codes/DEKR-bean/output/1019_030759/bean/hrnet_dekr_graph/w32_4x_reg03_bs10_512_adam_lr1e-3_bean_x140/model_best.pth.tar'
    # model_path = '/workspace/codes/DEKR-bean/output/0329_093557/bean/hrnet_dekr_graph/w32_4x_reg03_bs10_512_adam_lr1e-3_bean_x140/model_best.pth.tar'
    # 加入12摄像头训练后的最佳模型
    # model_path = '/workspace/codes/DEKR-bean/tools/output/bean/hrnet_dekr_graph/graph_w32_4x_reg03_bs10_512_adam_lr1e-3_bean_x140/model_best.pth.tar'

    model = BeanModel(model_path)
    out_path = '/workspace/datas/beans/results/test_one'

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    image_size = 512
    step = 512
    # image_path = '/workspace/datas/beans/all_data/val/images'
    # prefix_path = '/workspace/datas/beans/fullsize_new'
    # prefix_path = '/workspace/datas/beans/single_plant_v5'
    # prefix_path = '/data1/datas/beans/datas_zjlab_new'
    prefix_path = '/data1/datas/beans/beans_20230906/data'
    for dir_path in os.listdir(prefix_path):
        # dir_path = '/workspace/datas/beans/beans_multicamera20230327/picked_marks'
        # imgs_files = glob.glob(os.path.join(dir_path, '*.jpg'))
        # for k,image_name in enumerate(os.listdir(os.path.join(prefix_path, dir_path))):
        # for k,image_name in enumerate(os.listdir(abs_dir)):
        # for k, image_name in enumerate(os.listdir(prefix_path)):
        imgs_files = glob.glob(os.path.join(prefix_path, dir_path, '*.jpg'))
        for k, image_file in enumerate(imgs_files):

            if k > 6:
                break
            # image_file = os.path.join(prefix_path, dir_path, image_name)
            # image_file = os.path.join(prefix_path, image_name)
            joints = []
            location = []

            img_data = cv2.imread(image_file,cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

            height, width, c = img_data.shape
            x_start = 0
            y_start = 0
            x_finish = min(int(x_start + image_size), width)
            y_finish = min(int(y_start + image_size), height)
            crop_num = 0
            while y_finish > y_start:
                x_start = 0
                x_finish = min(int(x_start + image_size), width)
                while x_finish > x_start:
                    crop_num += 1
                    print('process: {}'.format(crop_num))
                    crop_image = img_rgb[y_start:y_finish, x_start:x_finish,:]
                    poses, scores = model.inference(crop_image)
                    location.append([x_start,y_start,x_finish,y_finish])

                    # for i, (pi, si) in enumerate(zip(p, scores)):
                    #     color = colors(i, bgr=False)
                    #     for x, y, s in pi:
                    #         if s < 1e-3:
                    #             continue
                    #         cv2.circle(image_out, (int(x), int(y)), 6, color, -1)
                    #
                    #     plt.imshow(image_out)
                    #     plt.show()
                    if len(poses) > 0:
                        poses = np.array(poses)
                        mask = poses[:, :, 2] > 0
                        poses[:, :, 0] = (poses[:, :, 0] + x_start) * mask
                        poses[:, :, 1] = (poses[:, :, 1] + y_start) * mask
                        if len(joints) > 0:
                            joints = np.concatenate((joints, poses), axis=0)
                        else:
                            joints = poses

                    x_start = x_start + step
                    x_finish = min(int(x_start + image_size), width)

                y_start = y_start + step
                y_finish = min(int(y_start + image_size), height)
            # show_result(joints, img_rgb.copy())
            joints_filter = range_filter(location,joints,(width,height))
            del joints
            # show_result(joints_filter, img_rgb.copy())
            img_show = draw_pods(img_data, joints_filter, alpha=1.0)
            cv2.imwrite(os.path.join(out_path, os.path.basename(image_file)), img_show)
            plt.figure(figsize=(48,48))
            plt.imshow(img_show[:,:,::-1])
            plt.show()
            pods_dict = pods_counting(joints_filter)
            print(pods_dict)

            # out_name = dir_path + '_' + image_name
            # plt.savefig(os.path.join(out_path,out_name))
    print('done')

