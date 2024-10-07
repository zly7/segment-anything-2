from dataclasses import dataclass
from multiprocessing import Pool
import os
import numpy as np
from tqdm import tqdm
@dataclass
class occlusion_rate_data:
    how_many_objects: int
    average_occlusion_rate: float
    occlusion_two_objs_pixel: int
    occlusion_three_objs_pixel: int
    occlusion_more_than_three_objs_pixel: int
    
# def calculate_occlusion_rate(folder_path):
#     # 初始化像素统计数组
#     pixel_counts_dic= {}
#     total_masks_dic = {}
#     occlusion_rates = {}
#     # 遍历文件夹中的所有npz文件
#     path_list = os.listdir(folder_path)
#     path_list = path_list[:100]
#     for file_name in tqdm(path_list):
#         if file_name.endswith('.npz'):
#             file_name_frist = file_name.split('.')[0]
#             name_list = file_name_frist.split('_')
#             frame_index, object_id = int(name_list[1]), int(name_list[2])
#             file_path = os.path.join(folder_path, file_name)
#             data = np.load(file_path)
#             mask = data['mask_key_name'] 

            
#             if pixel_counts_dic.get(frame_index) is None:
#                 pixel_counts_dic[frame_index] = np.zeros_like(mask, dtype=np.int32)
#             pixel_counts_dic[frame_index] += mask
#             if total_masks_dic.get(frame_index) is None:
#                 total_masks_dic[frame_index] = 0
#             total_masks_dic[frame_index] += 1

    
#     for frame_index, counts in pixel_counts_dic.items():
#         corrected_counts = np.where(counts > 0, counts - 1, counts)
#         per_pixel_rates = corrected_counts / total_masks_dic[frame_index]
#         valid_rates = per_pixel_rates[per_pixel_rates > 0]
#         if valid_rates.size > 0:
#             average_rate = np.mean(valid_rates)
#             rate_75per = np.percentile(valid_rates, 75)
#             rate_50per = np.percentile(valid_rates, 50)
#             rate_25per = np.percentile(valid_rates, 25)
#             occlusion_rates[frame_index] = occlusion_rate_data(how_many_objects=total_masks_dic[frame_index], average_occlusion_rate=average_rate, occlusion_rate_75per=rate_75per, occlusion_rate_50per=rate_50per, occlusion_rate_25per=rate_25per)
#         else:
#             occlusion_rates[frame_index] = occlusion_rate_data(how_many_objects=total_masks_dic[frame_index], average_occlusion_rate=0, occlusion_rate_75per=0, occlusion_rate_50per=0, occlusion_rate_25per=0)
#     return occlusion_rates

# 定义用于计算单个 frame_id 遮挡率的函数
def process_frame(frame_data):
    frame_index, file_list, folder_path = frame_data
    pixel_counts = None
    total_masks = 0

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        data = np.load(file_path)
        mask = data['mask_key_name']
        
        if pixel_counts is None:
            pixel_counts = np.zeros_like(mask[0], dtype=np.int32)
        
        pixel_counts += mask[0] #[480, 640]
        total_masks += 1

    if total_masks > 0:
        corrected_counts = np.where(pixel_counts > 0, pixel_counts - 1, pixel_counts)
        valid_rates = corrected_counts
        # per_pixel_rates = corrected_counts / total_masks
        # valid_rates = per_pixel_rates[per_pixel_rates > 0]
        if valid_rates.size > 0:
            average_rate = np.mean(valid_rates)  # 每一张张片的平均遮挡率
            # rate_75per = np.percentile(valid_rates, 75)
            # rate_50per = np.percentile(valid_rates, 50)
            # rate_25per = np.percentile(valid_rates, 25)
            occlusion_two_objs_pixel = np.sum(valid_rates == 2)
            occlusion_three_objs_pixel = np.sum(valid_rates == 3)
            occlusion_more_than_three_objs_pixel = np.sum(valid_rates > 3)
            # print("Number of non-zero rates:", np.count_nonzero(valid_rates))
            # print("Minimum value in valid_rates:", np.min(valid_rates))
            # print("Maximum value in valid_rates:", np.max(valid_rates))
        else:
            average_rate = occlusion_two_objs_pixel = occlusion_three_objs_pixel = occlusion_more_than_three_objs_pixel = 0
    else:
        average_rate = occlusion_two_objs_pixel = occlusion_three_objs_pixel = occlusion_more_than_three_objs_pixel = 0
    
    return (frame_index, total_masks, average_rate, occlusion_two_objs_pixel, occlusion_three_objs_pixel, occlusion_more_than_three_objs_pixel)

def calculate_occlusion_rate(folder_path, num_processes=36):
    # 组织数据以用于多进程
    files_by_frame = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npz'):
            file_name_frist = file_name.split('.')[0]
            name_list = file_name_frist.split('_')
            frame_index = int(name_list[1])
            if frame_index not in files_by_frame:
                files_by_frame[frame_index] = []
            files_by_frame[frame_index].append(file_name)
    
    # 创建多进程池
    pool = Pool(processes=num_processes)
    frame_data_list = [(frame_index, files, folder_path) for frame_index, files in files_by_frame.items()]
    
    occlusion_rates = {}
    results = pool.map(process_frame, frame_data_list)
    
    # 结果组装
    for result in results:
        frame_index, total_masks, average_rate,occlusion_two_objs_pixel, occlusion_three_objs_pixel, occlusion_more_than_three_objs_pixel = result
        occlusion_rates[frame_index] = {
            "how_many_objects": total_masks,
            "average_occlusion_rate": average_rate,
            "occlusion_two_objs_pixel": occlusion_two_objs_pixel,
            "occlusion_three_objs_pixel": occlusion_three_objs_pixel,
            "occlusion_more_than_three_objs_pixel": occlusion_more_than_three_objs_pixel
        }
    
    pool.close()
    pool.join()
    
    return occlusion_rates


scene = "MOT17-04"
dataset_name = "MOT17"
exp_name = "no_overlap_give_every_step_8_frames_each_id"#/data/zly/mot_data/MOT17/no_overlap_give_every_step_8_frames_each_id
split = "train"
# folder_path = f'/data/zly/mot_data/MOT17/give_every_step10_frames_each_id/train/{scene}/npz_result'
folder_path = f'/data/zly/mot_data/{dataset_name}/{exp_name}/{split}/{scene}/npz_result'    
occlusion_rates = calculate_occlusion_rate(folder_path)
print("Occlusion rates calculated.")
boolarray_example = os.listdir(folder_path)[0]
mask_example = np.load(os.path.join(folder_path, boolarray_example))['mask_key_name']
height,weight = mask_example[0].shape
prefix_path = f"./occlusion/{dataset_name}/{exp_name}/{split}"
if not os.path.exists(prefix_path):
    os.makedirs(prefix_path)
import matplotlib.pyplot as plt
# 可视化
for vkey in ['how_many_objects', 'average_occlusion_rate', 'occlusion_two_objs_pixel', 'occlusion_three_objs_pixel', 'occlusion_more_than_three_objs_pixel']:
    rates = [occlusion_rates[frame_index][vkey] for frame_index in occlusion_rates.keys()]
    plt.scatter(list(occlusion_rates.keys()), rates)
    plt.xlabel('Frame index')
    plt.ylabel(vkey)
    plt.title(f'{vkey} over frames with {height}x{weight} resolution')
    plt.show()
    plt.savefig(f"{prefix_path}/occlusion_rate_{scene}_{vkey}.png")
    plt.close()

