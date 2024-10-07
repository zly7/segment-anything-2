import cv2
import os
import glob
from multiprocessing import Pool
from colorama import Fore, Style
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
'''
这个文件主要是为了把这个文件和验证的label做好
'''
def make_result_in_image(args):
    print(Fore.BLUE + f'正在处理: 把小于0或者超出图片的txt去掉')
    base_path, folder_path, sub_folder, result_txt_path = args
    morethan0_txt = open(os.path.join(os.path.dirname(result_txt_path), "gt>0.txt"), 'w')
    images = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
    height, width, _ = cv2.imread(images[0]).shape
    with open(result_txt_path, 'r') as f:
        for line in f.readlines():
            linelist = line.strip().split(',')
            # whether_pedestrain = int(linelist[7])
            # if whether_pedestrain !=  1: # 只处理行人,dancetrack和MOT都是1表示人 这里因为是测试所以
            #     continue
            img_id = int(linelist[0])
            obj_id = int(linelist[1])
            bbox = [float(linelist[2]), float(linelist[3]),
                    float(linelist[2]) + float(linelist[4]), 
                    float(linelist[3]) + float(linelist[5]), obj_id]
            left_up = (int(bbox[0]), int(bbox[1]))
            right_down = (int(bbox[2]), int(bbox[3]))
            left_up = (max(0, left_up[0]), max(0, left_up[1]))
            left_up = (min(width - 1, left_up[0]), min(height - 1, left_up[1]))
            right_down = (max(0, right_down[0]), max(0, right_down[1]))
            right_down = (min(width - 1, right_down[0]), min(height - 1, right_down[1]))
            linelist[2] = str(left_up[0])
            linelist[3] = str(left_up[1])
            linelist[4] = str(right_down[0]-left_up[0])
            linelist[5] = str(right_down[1]-left_up[1])
            morethan0_txt.write(','.join(map(str, linelist)) + '\n')
    morethan0_txt.close()

def one_process_video(args):
    '''
    这里显然应该对于args有一个解释
    '''
    base_path, folder_path, sub_folder, result_txt_path = args 
    print(Fore.BLUE + f'正在处理: {sub_folder}')
    output_video_path = os.path.join(base_path, "processed_video", f'{sub_folder}.avi') 
    mask_detection_img_base_path = os.path.join(base_path, "mask_detection",sub_folder)
    if not os.path.exists(mask_detection_img_base_path):
        os.makedirs(mask_detection_img_base_path)
    fps = 24  # 视频的帧率
    

    # 读取BBox数据
    txt_dict = {}
    with open(result_txt_path, 'r') as f:
        for line in f.readlines():
            linelist = line.strip().split(',')
            whether_pedestrain = int(linelist[7])
            # if whether_pedestrain !=  1: # 只处理行人,dancetrack和MOT都是1表示人 这里因为是测试所以
            #     continue
            img_id = int(linelist[0])
            obj_id = int(linelist[1])
            bbox = [float(linelist[2]), float(linelist[3]),
                    float(linelist[2]) + float(linelist[4]), 
                    float(linelist[3]) + float(linelist[5]), obj_id]
            if img_id in txt_dict:
                txt_dict[img_id].append(bbox)
            else:
                txt_dict[img_id] = [bbox]

        # 获取文件夹中所有的 jpg 文件
    images = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = cv2.imread(images[0]).shape[:2][::-1]
    print("frame_size" + str(frame_size))
    # frame_size = (1920, 1080)  # 视频的帧大小
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)


    id_color_map = plt.get_cmap("tab20") # 获取颜色图谱
    
    # 读取每一张图片，添加边界框，然后写入到视频文件中
    for image_path in images:
        
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, frame_size)
        img_id = int(os.path.basename(image_path).split('.')[0])  # 假设图片名称为帧ID
        cv2.putText(img_resized, f"Frame index: {img_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if img_id in txt_dict:
            for bbox in txt_dict[img_id]:
                obj_id = bbox[4]
                color = id_color_map(obj_id % 20)[:3]  # Retrieve color, mod with 20 to prevent index out of bounds
                color = (int(color[2]*255), int(color[1]*255), int(color[0]*255))   # 逆序数组以转换 RGB 到 BGR
                cv2.rectangle(img_resized, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.putText(img_resized, str(bbox[4]), (int(bbox[0]) + 5, int(bbox[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        whether_succces = cv2.imwrite(os.path.join(mask_detection_img_base_path, os.path.basename(image_path)), img_resized)
        assert whether_succces
        video_writer.write(img_resized)

    video_writer.release()  # 释放写入器
    print(f'处理完成: {sub_folder}')

def process_folder(base_folder_path,sub_folders=None):
    folder_paths = []
    if sub_folders is None:
        iter_sub_folder = os.listdir(base_folder_path)
    else:
        iter_sub_folder = sub_folders
    for sub_folder in iter_sub_folder:
        if "mask" in os.listdir(os.path.join(base_folder_path, sub_folder)):
            folder_path = os.path.join(base_folder_path, sub_folder, "mask")
            result_txt_path = os.path.join(base_folder_path, "track_results", f"{sub_folder}.txt")
            folder_paths.append((base_folder_path, folder_path, sub_folder,result_txt_path))
    return folder_paths

if __name__ == '__main__':
    # base_folder_path = '/data/zly/mot_data/MOT17/only_init_first_frame_each_id/train'
    # base_folder_path = '/data/zly/mot_data/MOT17/give_every_step10_frames_each_id/train'
    # base_folder_path = "/data/zly/mot_data/MOT17/no_overlap_give_every_step_8_frams_each_id_big_than_0/train"
    # base_folder_path = "/data2/zly/mot_data/MTMMC/no_overlap_give_every_step_10_frames_each_id_gt_big_than_0/val"
    # base_folder_path = "/data/zly/mot_data/MOT17/track_with_1_frames_bbox_from_truth_without_predict/train"
    base_folder_path = "/data/zly/mot_data/DanceTrack/track_with_1_frames_bbox_from_truth_without_predict/val"
    if not os.path.exists(os.path.join(base_folder_path, "processed_video")): # 这个函数在多进程
        os.makedirs(os.path.join(base_folder_path, "processed_video"))
    if not os.path.exists(os.path.join(base_folder_path,"mask_detection")):
        os.makedirs(os.path.join(base_folder_path,"mask_detection"))
    if "MTMMC" in base_folder_path and "val" in base_folder_path:
        mtmmc_val_first = ["s14","s17","s19","s31","s37"]
        camera = ["c"+"{:02}".format(i) for i in range(1,17)]
        iter_sub_folder = [os.path.join(i, j) for i in mtmmc_val_first for j in camera]
        folder_paths_args = process_folder(base_folder_path,iter_sub_folder)
    else:
        folder_paths_args = process_folder(base_folder_path)
    print(Fore.BLUE+str(folder_paths_args))
    with Pool(12) as pool:
        pool.map(one_process_video, folder_paths_args)
        # pool.map(make_result_in_image, folder_paths_args)
