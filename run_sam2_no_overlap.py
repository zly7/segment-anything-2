import argparse
import glob
import json
import os
import pstats
import sys
import time
import cv2
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from colorama import Fore, Style, init
from loguru import logger
import multiprocessing
import cProfile
from tqdm import tqdm
from pympler import asizeof
from memory_profiler import profile

class Args:
    def __init__(self):
        pass 
def get_visible_devices():
    # 从环境变量中获取 CUDA_VISIBLE_DEVICES
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', 'None')
    return cuda_visible_devices.replace(",", "_")  # 替换逗号以便文件命名

cuda_visible_devices = get_visible_devices()
log_file_name = f"logs/run_sam2_no_overlap_cuda_visable_{cuda_visible_devices}.log"
logger.add(log_file_name, rotation="1 week")
logger.info("------------------------------------start one test----------------------------------------------")

init(autoreset=True)  # 初始化 colorama 并设置自动重置颜色
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

def frame_index_2_dataset_index(frame_index,dataset): # 0-based frame index to 1-based dataset
    if dataset == "MTMMC":
        return frame_index
    elif dataset == "DanceTrack" or dataset == "MOT17" or dataset == "MOT20":
        return frame_index + 1
    else:
        raise NotImplementedError

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    


def extract_bbox(mask):
    """Extract bounding box (x1, y1, width, height) from a binary mask. 要处理没有bbox的情况"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return 0, 0, 0, 0
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1

@profile(stream=open('logs/memory_log_write_txt.txt', 'w+'))
def video_segments_to_evl_txt(video_segments, output_txt_path, np_save_path_folder, save_immediately=False, frame_names=None, dataset="MTMMC"):
    assert len(video_segments) == 0
    start_time = time.time()  # 开始时间
    total_frames = len(frame_names)  # 总帧数
    with open(output_txt_path, 'w') as f:
        for frame_idx in range(0, total_frames):
            frame_name = frame_names[frame_idx]
            frame_index = frame_idx
            frame_masks_info = glob_and_read_npz(np_save_path_folder, frame_index)
            
            for obj_id, mask in frame_masks_info.items():
                # logger.info(f"farme_masks_info {asizeof.asizeof(mask) / (1024 ** 3) } GB")
                x1, y1, w, h = extract_bbox(mask)
                if w == 0 or h == 0:
                    continue
                score = 0.9
                line = f"{frame_index_2_dataset_index(frame_idx, dataset)},{obj_id},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{score},-1,-1,-1\n"
                f.write(line)
            # logger.info("----------------------一个frame_masks_info结束--------------------------")
            # 每处理100帧log一次
            if frame_idx % 100 == 0 and frame_idx > 0:
                pid = os.getpid()
                process = psutil.Process(pid)
                memory_info = process.memory_info()
                logger.info(f"内存使用: {memory_info.rss / (1024 ** 3):.2f} GB")
                elapsed_time = time.time() - start_time  # 已经过的时间
                avg_time_per_frame = elapsed_time / frame_idx  # 每帧平均耗时
                remaining_frames = total_frames - frame_idx  # 剩余帧数
                estimated_remaining_time = remaining_frames * avg_time_per_frame  # 预计剩余时间
                logger.info(f"Processed frame {frame_idx}/{total_frames}. Estimated remaining time: {estimated_remaining_time:.2f} seconds.")

def save_mask(mask, ax, obj_id=None, random_color=False, pic_save_path = None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    if pic_save_path:
        plt.savefig(pic_save_path)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


def init_new_points_from_real_label_txt(txt_path,step=20,dataset="MTMMC") ->  dict:
    '''
    这里必须要每间隔step帧去模拟真实的grounding dino的情况,每个物体要相同
    '''
    assert step > 0
    result_dict = {}
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            linelist = line.split(',')
            if dataset == "DanceTrack" or dataset == "MOT17" or dataset == "MOT20":
                whether_pedestrain = int(linelist[7])
                if whether_pedestrain !=  1: # 只处理行人,dancetrack和MOT都是1表示人
                    continue
            elif dataset == "MTMMC":
                pass
            else:
                raise NotImplementedError
            img_id = linelist[0]
            obj_id = linelist[1]
            bbox = [float(linelist[2]), float(linelist[3]), 
                    float(linelist[2]) + float(linelist[4]), 
                    float(linelist[3]) + float(linelist[5])]
            if dataset == "MTMMC":
                frame_idx = int(img_id)
            elif dataset in ["DanceTrack","MOT17","MOT20"]:
                frame_idx = int(img_id) - 1
            else:
                raise NotImplementedError
            if frame_idx % step == 0:  # 每隔step帧初始化一次

                result_dict[frame_idx] = result_dict.get(frame_idx, {})
                result_dict[frame_idx][obj_id] = bbox
            else:
                continue
    return result_dict

def propagate_and_extend_video_segments(video_segments, predictor, inference_state,start_frame_index:int = None,max_frame_num_to_track:int = None):
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,start_frame_idx=start_frame_index,max_frame_num_to_track=max_frame_num_to_track):
        if out_frame_idx not in video_segments:
            video_segments[out_frame_idx] = {}
        
        for i, out_obj_id in enumerate(out_obj_ids):
            # This ensures that if out_obj_id already exists, it updates the mask; if not, it adds a new entry
            video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
            # mb_used = video_segments[out_frame_idx][out_obj_id].nbytes / 1024 / 1024 # 一个mask的大小大约是0.9MB
            # print(Fore.RED + f"frame: {out_frame_idx};obj_id: {out_obj_id};mb_used: {mb_used}")

def save_and_clear_video_segments(video_segments, save_path_folder):
    for frame_idx, segments in video_segments.items():
        npz_folder_path = os.path.join(save_path_folder,str(frame_idx))
        if not os.path.exists(npz_folder_path):
            os.makedirs(npz_folder_path)
        for obj_id, mask in segments.items():
            np_save_path = os.path.join(npz_folder_path, f"{obj_id}.npz") # 这个是npz的格式
            np.savez_compressed(np_save_path, mask_key_name = mask)
    video_segments.clear()
    
def glob_and_read_npz(np_save_path_folder,need_search_frame_idx:int) -> dict: # 这里的np_save_path_folder应该是带有这个文件夹是什么的
    '''
    need_search_frame_idx 是 frame_names的索引
    '''
    to_return_frame_masks_info = {}
    folder_path = os.path.join(np_save_path_folder, f"{need_search_frame_idx}")
    if not os.path.exists(folder_path):
        return {}
    for npz_file_name in os.listdir(folder_path):
        npz_path = os.path.join(folder_path, npz_file_name)
        obj_id = int(os.path.splitext(os.path.basename(npz_file_name))[0])
        npz_dict = np.load(npz_path)
        to_return_frame_masks_info[obj_id] = npz_dict["mask_key_name"][0]
        assert len(to_return_frame_masks_info[obj_id].shape) == 2
    return to_return_frame_masks_info


def vis_process(args,scene_output_img_path, video_dir, frame_names, np_save_path_folder, 
                    exp_path, video_sub_path, video_segments):
    '''
    使用方法:
    
    video_dir = "/data2/zly/mot_data/MTMMC/val/sxx/cxx/rgb"
    '''
    logger.info(f"~~~~~~~~vis subprocess start : {exp_path} {video_sub_path}~~~~~~~~")
    if args.vis == "True":
        vis_frame_stride = args.vis_frame_stride if args.vis_frame_stride is not None else 15
        if not os.path.exists(scene_output_img_path):
            os.makedirs(scene_output_img_path)
        id_color_map = plt.get_cmap("tab20")  # Using matplotlib's tab20 colormap which has 20 distinct colors
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            frame_image_path = os.path.join(video_dir, frame_names[out_frame_idx])
            frame_image = Image.open(frame_image_path)
            frame_masks_info = glob_and_read_npz(np_save_path_folder,out_frame_idx)
            
            # Create an RGB mask image with the same dimensions as the frame image
            mask_image = np.zeros((*frame_image.size[::-1], 3), dtype=np.uint8)
            
            for obj_id, obj_mask in frame_masks_info.items():
                if len(obj_mask.shape) == 3:
                    obj_mask = obj_mask[0]
                color = id_color_map(obj_id % 20)[:3]  # Retrieve color, mod with 20 to prevent index out of bounds
                color = (np.array(color) * 255).astype(np.uint8)  # Convert from [0,1] to [0,255]
                for c in range(3):  
                    mask_image[:, :, c][obj_mask] = color[c]
                # x1, y1, w, h = extract_bbox(obj_mask)
                # cv2.rectangle(img_resized, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                # cv2.putText(img_resized, str(bbox[4]), (int(bbox[0]) + 5, int(bbox[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Combine the original image with the colored mask
            combined_image = cv2.addWeighted(np.array(frame_image), 0.7, mask_image, 0.3, 0)
            
            # Save the combined image
            output_image_path = os.path.join(scene_output_img_path, f"{frame_names[out_frame_idx]}")
            cv2.imwrite(output_image_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
            
        
    out_put_txt_path = os.path.join(exp_path,"track_results", f"{video_sub_path}.txt")
    if not os.path.exists(os.path.dirname(out_put_txt_path)):
        os.makedirs(os.path.dirname(out_put_txt_path))
    if False: #DEBUG
        video_segments_to_evl_txt(video_segments, out_put_txt_path,np_save_path_folder,frame_names=frame_names,dataset=args.dataset)
    del video_segments
    logger.success(f"~~~~~~~~vis subprocess end : {video_sub_path}~~~~~~~~")
    
# if __name__ == '__main__':
#     logger.warning("正在进行Debug测试！！！！")
#     profiler = cProfile.Profile()
#     profiler.enable()  # 开始分析
#     args = Args()
#     args.vis = "False"
#     args.vis_frame_stride = 1
#     args.dataset = "MTMMC"
#     sub_path_folder = "s17/c01"
#     video_dir = F"/data2/zly/mot_data/MTMMC/val/{sub_path_folder}/rgb"
#     frame_names = [
#             p for p in os.listdir(video_dir)
#             if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG",'.png']
#     ]
#     # video_segments_to_evl_txt({},"/data2/zly/mot_data/MTMMC/no_overlap_give_every_step_10_frames_each_id/val/track_results/s17/c14.txt",
#     #                           "/data2/zly/mot_data/MTMMC/no_overlap_give_every_step_10_frames_each_id/val/s17/c14/npz_result",
#     #                           frame_names=frame_names,dataset="MTMMC")
#     vis_process(args,f"/data2/zly/mot_data/MTMMC/no_overlap_give_every_step_10_frames_each_id/val/{sub_path_folder}/mask",
#                 f"/data2/zly/mot_data/MTMMC/val/{sub_path_folder}/rgb", frame_names, f"/data2/zly/mot_data/MTMMC/no_overlap_give_every_step_10_frames_each_id/val/{sub_path_folder}/npz_result",
#                 "/data2/zly/mot_data/MTMMC/no_overlap_give_every_step_10_frames_each_id/val", sub_path_folder, {})
#     profiler.disable()  # 停止分析
#     with open("logs/profiling_results.txt", "w") as f:
#         stats = pstats.Stats(profiler, stream=f)
#         stats.sort_stats('time')  # 根据耗时排序
#         stats.print_stats()  # 打印并保存结果
#     profiler.dump_stats("logs/profiling_output.prof")
#     exit()
    
    
    

def main():
    multiprocessing.set_start_method('spawn')
    import hydra
    from sam2.build_sam import build_sam2_video_predictor
    parser = argparse.ArgumentParser(description="Run SAM2 video predictor.")
    parser.add_argument("--dataset", help="Name of the dataset directory", type=str, required=True)
    parser.add_argument("--split", help="Name of the data split directory", type=str, required=True)
    parser.add_argument("--step_num", help="Name of the data split directory", type=int, required=True)
    parser.add_argument("--vis", help="Name of the config file", type=str, required=True)
    parser.add_argument("--vis_frame_stride", help="Name of the config file", type=int, required=False)
    parser.add_argument("--copy_reset", help="copy file and generate a new predictor", type=bool, required=False)
    parser.add_argument("--gt_big_than_0", type=bool, required=False)
    args = parser.parse_args()
    logger.info(f"args: {args}")
    if args.dataset != "DanceTrack" and args.dataset != "MOT17" and args.dataset != "MOT20" and args.dataset != "MTMMC":   
        raise NotImplementedError
    if args.split not in ["train", "val", "test"]:
        raise NotImplementedError
    if args.step_num <= 0:
        whether_only_init_first_frame_each_id = True
    else:
        whether_only_init_first_frame_each_id = False
    # hydra is initialized on import of sam2, which sets the search path which can't be modified
    # so we need to clear the hydra instance
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # reinit hydra with a new search path for configs
    hydra.initialize_config_module('sam2_configs', version_base='1.2') # 这个相对路径和python文件的路径相关
    # sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    whether_use_pointjson_or_labeltxt = "txt"
    # video_root_path = "/data/zly/mot_data/DanceTrack/train/"
    if args.dataset !="MTMMC":
        video_root_path = f"/data/zly/mot_data/{args.dataset}/{args.split}/"
        video_path_list = os.listdir(video_root_path)
        video_path_list.sort()
    else:
        video_root_path = f"/data2/zly/mot_data/{args.dataset}/{args.split}/"
        mtmmc_train_first = ["s01","s10","s11","s13","s16","s18","s20","s34","s35","s36","s38","s39","s42","s47"]
        # mtmmc_train_first=["s00"] # DEBUG
        mtmmc_val_first = ["s14","s17","s19","s31","s37"]
        # mtmmc_val_first = ["s37"] # DEBUG
        camera = ["c"+"{:02}".format(i) for i in range(6,13)]
        # camera = ["c"+"{:02}".format(i) for i in range(1,7)]#tmux 4
        # camera = ["c"+"{:02}".format(i) for i in range(7,17)]#tmux 5
        # camera = ["c01","c02"] # DEBUG
        if args.split == "train" :
            video_path_list = [os.path.join(i, j) for i in mtmmc_train_first for j in camera]
        elif args.split == "val":
            video_path_list = [os.path.join(i, j) for i in mtmmc_val_first for j in camera]
        else:
            raise NotImplementedError
        list_of_debug = []
        # list_of_debug = [ # DEBUG
        #     "s14/c01", "s14/c02", "s14/c03", "s14/c04", "s14/c05", "s14/c06", "s14/c07", "s14/c08",
        #     "s14/c09", "s14/c10", "s14/c11", "s14/c12", "s14/c13", "s14/c14", "s14/c15", "s14/c16",
        #     "s17/c03", "s17/c09", "s17/c10", "s17/c11", "s17/c12", "s37/c15","s31/c09","s31/c10","s31/c11","s31/c15",
        #     "s19/c09","s19/c13","s19/c14","s19/c16","s17/c13","s17/c14"
        # ]
        video_path_list = list(set(video_path_list)-set(list_of_debug))
        video_path_list.sort()
    
    # video_path_list = ["dancetrack0025"]
    # video_path_list = ["MOT17-04"]
    # if args.dataset == "DanceTrack":
    #     video_path_list = [
    #         "dancetrack0000", 
    #         # "dancetrack0004", "dancetrack0005", "dancetrack0007",
    #         # "dancetrack0010", "dancetrack0014", "dancetrack0018", "dancetrack0019",
    #         # "dancetrack0025", "dancetrack0026", 
    #         # "dancetrack0030", 
    #         # "dancetrack0034",
    #         # "dancetrack0035", "dancetrack0041", "dancetrack0043", "dancetrack0047",
    #         # "dancetrack0058", "dancetrack0063", "dancetrack0065", "dancetrack0073",
    #         # "dancetrack0077", "dancetrack0079", 
    #         # "dancetrack0081", "dancetrack0090",
    #         # "dancetrack0094", "dancetrack0097"
    #     ]
    # elif args.dataset == "MOT17":
    # if args.dataset == "MOT17":
    #     video_path_list = [
    #         # "MOT17-02", 
    #         "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"
    #     ]
    logger.info(f"video_path_list: {video_path_list}")
    exp_name = f"no_overlap_give_every_step_{args.step_num}_frames_each_id"
    if args.gt_big_than_0:
        exp_name += "_gt_big_than_0"
    mul_process_list = []
    for video_sub_path in video_path_list:
        hydra_overrides_extra = [
        "++model.non_overlap_masks=" + "true"
        ]
        print(f"Processing {video_sub_path}")
        scene_base_path = os.path.join(video_root_path, video_sub_path) # 尺寸数据集的地方
        which_ssd = "data2" if "data2" in video_root_path else "data"
        if "val" in video_root_path:
            exp_path = f"/{which_ssd}/zly/mot_data/{args.dataset}/{exp_name}/val"
        elif "train" in video_root_path:
            exp_path = f"/{which_ssd}/zly/mot_data/{args.dataset}/{exp_name}/train"
        elif "test" in video_root_path:
            exp_path = f"/{which_ssd}/zly/mot_data/{args.dataset}/{exp_name}/test"
        else:
            raise NotImplementedError
        scene_output_path = os.path.join(exp_path, video_sub_path)
        scene_output_img_path = os.path.join(scene_output_path, "mask")
        if "dancetrack" in video_root_path.lower() or "MOT" in video_root_path:
            video_dir = os.path.join(scene_base_path ,"img1")
        elif "mtmmc" in video_root_path.lower():
            video_dir = os.path.join(scene_base_path ,"rgb")
        else:
            raise NotImplementedError
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG",'.png']
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        batch_size = args.step_num
        frame_names_batch  = [frame_names[i:i + batch_size] for i in range(0, len(frame_names), batch_size)]

        if whether_use_pointjson_or_labeltxt == "txt":
            if args.dataset in ["MOT17","MOT20","DanceTrack"]:
                txt_path = os.path.join(scene_base_path, "gt", "gt>0.txt" if args.gt_big_than_0 else "gt.txt")
            elif args.dataset == "MTMMC": # 这里是处理了重复的0的id error的问题
                txt_path = os.path.join(scene_base_path, "gt", "gt_mot>0.txt" if args.gt_big_than_0 else "gt_mot.txt")
            result_dict = init_new_points_from_real_label_txt(txt_path,step=args.step_num,dataset=args.dataset)
        else:
            raise NotImplementedError
        np_save_path_folder = os.path.join(scene_output_path, "npz_result")
        if not os.path.exists(np_save_path_folder):
            os.makedirs(np_save_path_folder)
        if False:
            if not args.copy_reset: # 不复制文件就要在开始前初始化predictor
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint,hydra_overrides_extra=hydra_overrides_extra)
                inference_state = predictor.init_state(video_path=video_dir, # 这一步会狂吃内存,因为load了整个视频
                                                offload_video_to_cpu=True,
                                                offload_state_to_cpu=True)
            else:
                video_temp_dir = os.path.join(video_dir, "batch_temp")
                if not os.path.exists(video_temp_dir):
                    os.makedirs(video_temp_dir)
        
            for frame_names_this_iter in frame_names_batch:
                if args.copy_reset:  # 复制文件的情况
                    for frame_name in frame_names_this_iter:
                        os.system(f"cp {os.path.join(video_dir, frame_name)} {os.path.join(video_temp_dir, frame_name)}")
                    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint,hydra_overrides_extra=hydra_overrides_extra)
                    inference_state = predictor.init_state(video_path=video_temp_dir,
                                                offload_video_to_cpu=False,
                                                offload_state_to_cpu=True)
                    start_frame_index = None
                    max_frame_num_to_track = None
                else:
                    predictor.reset_state(inference_state)
                    if args.dataset in ["DanceTrack", "MOT17", "MOT20"]:
                        start_frame_index = int(os.path.splitext(frame_names_this_iter[0])[0]) - 1  
                    elif args.dataset == "MTMMC":
                        start_frame_index = int(os.path.splitext(frame_names_this_iter[0])[0])
                    else:
                        raise NotImplementedError
                    max_frame_num_to_track = args.step_num - 1 
                
                if args.dataset in ["DanceTrack","MOT17","MOT20"]:
                    add_bbox_frame_index = int(os.path.splitext(frame_names_this_iter[0])[0]) - 1  
                elif args.dataset == "MTMMC":
                    add_bbox_frame_index = int(os.path.splitext(frame_names_this_iter[0])[0])
                else:
                    raise NotImplementedError
                if add_bbox_frame_index in result_dict.keys():
                    for obj_id,info in result_dict[add_bbox_frame_index].items():
                        predictor.add_new_points_or_box(inference_state, add_bbox_frame_index, int(obj_id),box = info)
                if add_bbox_frame_index in result_dict.keys():
                    video_segments = {}  
                    propagate_and_extend_video_segments(video_segments, predictor, inference_state,start_frame_index=start_frame_index,max_frame_num_to_track=max_frame_num_to_track)      
                    save_and_clear_video_segments(video_segments, np_save_path_folder)
                    if args.copy_reset:
                        for frame_name in frame_names_this_iter:
                            os.system(f"rm {os.path.join(video_temp_dir, frame_name)}")
                else:
                    logger.info(f"scene {video_sub_path} frame {add_bbox_frame_index} has no bbox")
                if start_frame_index % 80 == 0:
                    pid = os.getpid()
                    process = psutil.Process(pid)
                    memory_info = process.memory_info()
                    logger.info(f"内存使用 in {start_frame_index}: {memory_info.rss / (1024 ** 3):.2f} GB")
                    # size_memory_predictor =  asizeof.asizeof(predictor)  # 问题不在这
                    # logger.info(f"size of predictor in {start_frame_index}: {size_memory_predictor / (1024 ** 3) } GB ")
                    # size_memory_state = asizeof.asizeof(inference_state) # 这个测不准
                    # logger.info(f"size of inference state in {start_frame_index}: {size_memory_state / (1024 ** 3) } GB")
        # exit()#DEBUG
        p = multiprocessing.Process(target=vis_process, args=(args,scene_output_img_path, video_dir, frame_names, np_save_path_folder, 
                    exp_path, video_sub_path, {}))
        p.start()
        mul_process_list.append(p)
        if False:#Debug
            del inference_state, predictor
        logger.success(f"scene {video_sub_path} done")
    for p in mul_process_list:
        p.join()
        
# def vis_main():
#     process_list = []
#     p = multiprocessing.Process(target=vis_process, args=(args,scene_output_img_path, video_dir, frame_names, np_save_path_folder, 
#                     exp_path, video_sub_path, video_segments))
        

if __name__ == '__main__':
    main()