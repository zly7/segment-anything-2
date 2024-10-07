import argparse
import glob
import json
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from colorama import Fore, Style, init
init(autoreset=True)  # 初始化 colorama 并设置自动重置颜色
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

def frame_index_2_dataset_index(frame_index): # 0-based frame index to 1-based dataset
    return frame_index + 1

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

def video_segments_to_evl_txt(video_segments, output_txt_path,np_save_path_folder,save_immediately = False,frame_names= None):
    if save_immediately:
        assert len(video_segments) == 0
    if not save_immediately:
        with open(output_txt_path, 'w') as f:
            for frame_idx, segments in video_segments.items():
                for obj_id, mask in segments.items():
                    x1, y1, w, h = extract_bbox(mask)  # [1,h,w]
                    if w == 0 or h == 0:
                        continue
                    # Assume a constant score of 0.9 for this example
                    score = 0.9
                    line = f"{frame_index_2_dataset_index(frame_idx)},{obj_id},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{score},-1,-1,-1\n"
                    f.write(line)
    else:
        with open(output_txt_path, 'w') as f:
            for frame_idx in range(0, len(frame_names)):
                frame_name = frame_names[frame_idx]
                frame_index = frame_idx
                frame_masks_info = glob_and_read_npz(np_save_path_folder,frame_index)
                for obj_id, mask in frame_masks_info.items():
                    x1, y1, w, h = extract_bbox(mask)
                    if w == 0 or h == 0:
                        continue
                    # Assume a constant score of 0.9 for this example
                    score = 0.9
                    line = f"{frame_index_2_dataset_index(frame_idx)},{obj_id},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{score},-1,-1,-1\n"
                    f.write(line)
                

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

def init_new_points_from_json(jsons_path, predictor = None, inference_state = None,end_init_frame_index = None) ->  dict:
    '''
    这个函数主要是处理这个第一帧
    '''
    # 遍历给定路径下的所有json文件
    return_dict = {}
    if end_init_frame_index is None:
        end_init_frame_index = 0
    for filename in os.listdir(jsons_path):
        if filename.endswith('.json'):
            json_path = os.path.join(jsons_path, filename)
            with open(json_path, 'r') as file:
                data = json.load(file)
            
            # 解析帧索引，从文件名获取，假设文件名格式为'00000XXX.json'
            frame_idx = int(filename[:-5])
            frame_idx = frame_idx - 1
            # 创建一个字典来组织每个对象的点和标签
            points_dict = {}
            
            for shape in data['shapes']:
                label = int(shape['label'])
                if label not in points_dict:
                    points_dict[label] = {'points': [], 'labels': []}
                
                for point in shape['points']:
                    # 将点坐标添加到对应标签的列表中
                    points_dict[label]['points'].append(point)
                    points_dict[label]['labels'].append(1)  # 假设所有点都是正面标记，即'1'
            return_dict[frame_idx] = points_dict
            # 对每个对象使用add_new_points函数, 仅在第一帧调用
            if predictor is not None and inference_state is not None and frame_idx <= end_init_frame_index :
                for obj_id, info in points_dict.items():
                    points = np.array(info['points'], dtype=np.float32)
                    labels = np.array(info['labels'], np.int32)
                    
                    # 模拟调用函数进行点的添加和模型更新
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=points,
                        labels=labels
                    )
    return return_dict

def init_new_points_from_real_label_txt(txt_path, predictor, inference_state, 
                                        whether_only_init_first_frame_each_id = False,step=20,
                                        whether_multi_predictor = True,
                                        whether_give_all_id_in_first_init_free = True) ->  dict:
    '''
    这里必须要每间隔step帧去模拟真实的grounding dino的情况,每个物体要相同
    '''
    assert whether_give_all_id_in_first_init_free == True # 暂时的设计就是id不能断档
    assert whether_only_init_first_frame_each_id == True or step > 0
    id_init_last_frame_map = {}
    result_dict = {}
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            linelist = line.split(',')
            whether_pedestrain = int(linelist[7])
            if whether_pedestrain !=  1: # 只处理行人,dancetrack和MOT都是1表示人
                continue
            img_id = linelist[0]
            obj_id = linelist[1]
            bbox = [float(linelist[2]), float(linelist[3]), 
                    float(linelist[2]) + float(linelist[4]), 
                    float(linelist[3]) + float(linelist[5])]
            frame_idx = int(img_id) - 1
            if whether_only_init_first_frame_each_id and obj_id in id_init_last_frame_map: # 只初始化每个id的第一帧
                continue
            else:
                if whether_only_init_first_frame_each_id == True and (obj_id not in id_init_last_frame_map):
                    id_init_last_frame_map[obj_id] = frame_idx # 这里不立刻添加是因为后面有set
                if whether_only_init_first_frame_each_id == False and frame_idx%step == 0:  # 每隔step帧初始化一次
                    id_init_last_frame_map[obj_id] = frame_idx
                    if whether_multi_predictor:
                        result_dict[obj_id] = result_dict.get(obj_id, {})
                        result_dict[obj_id][frame_idx] = bbox
                    else:
                        predictor.add_new_points_or_box(inference_state, frame_idx, int(obj_id),box = bbox)
                else:
                    continue
    if whether_only_init_first_frame_each_id == True:
        frames_need_init_set = set(id_init_last_frame_map.values())
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                whether_pedestrain = int(linelist[7])
                if whether_pedestrain !=  1: # 只处理行人,dancetrack和MOT都是1表示人
                    continue
                img_id = linelist[0]
                obj_id = linelist[1]
                bbox = [float(linelist[2]), float(linelist[3]), 
                        float(linelist[2]) + float(linelist[4]), 
                        float(linelist[3]) + float(linelist[5])]
                frame_idx = int(img_id) - 1
                if frame_idx in frames_need_init_set:
                    if whether_multi_predictor:
                        result_dict[obj_id] = result_dict.get(obj_id, {})
                        result_dict[obj_id][frame_idx] = bbox
                    else:
                        predictor.add_new_points_or_box(inference_state, frame_idx, int(obj_id),box = bbox)
    print(Fore.BLUE + f"init_how_many_frames_each_id: {len(id_init_last_frame_map)}")
    return result_dict

def propagate_and_extend_video_segments(video_segments, predictor, inference_state):
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        if out_frame_idx not in video_segments:
            video_segments[out_frame_idx] = {}
        
        for i, out_obj_id in enumerate(out_obj_ids):
            # This ensures that if out_obj_id already exists, it updates the mask; if not, it adds a new entry
            video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
            # mb_used = video_segments[out_frame_idx][out_obj_id].nbytes / 1024 / 1024 # 一个mask的大小大约是0.9MB
            # print(Fore.RED + f"frame: {out_frame_idx};obj_id: {out_obj_id};mb_used: {mb_used}")

def save_and_clear_video_segments(video_segments, save_path_folder):
    for frame_idx, segments in video_segments.items():
        for obj_id, mask in segments.items():
            np_save_path = os.path.join(save_path_folder, f"boolarray_{frame_idx}_{obj_id}.npz") # 这个是npz的格式
            np.savez_compressed(np_save_path, mask_key_name = mask)
    video_segments.clear()
    
def glob_and_read_npz(np_save_path_folder,need_search_frame_idx:int) -> dict: # 这里的np_save_path_folder应该是带有这个文件夹是什么的
    '''
    need_search_frame_idx 是 frame_names的索引
    '''
    to_return_frame_masks_info = {}
    for npz_path in glob.glob(os.path.join(np_save_path_folder, f"boolarray_{need_search_frame_idx}_*.npz")): 
        obj_id = int(npz_path.split("_")[-1].split(".")[0])
        npz_dict = np.load(npz_path)
        to_return_frame_masks_info[obj_id] = npz_dict["mask_key_name"][0]
        assert len(to_return_frame_masks_info[obj_id].shape) == 2
    return to_return_frame_masks_info
[]
if __name__ == '__main__':
    import hydra
    from sam2.build_sam import build_sam2_video_predictor
    parser = argparse.ArgumentParser(description="Run SAM2 video predictor.")
    parser.add_argument("--dataset", help="Name of the dataset directory", type=str, required=True)
    parser.add_argument("--split", help="Name of the data split directory", type=str, required=True)
    parser.add_argument("--step_num", help="Name of the data split directory", type=int, required=True)
    parser.add_argument("--vis", help="Name of the config file", type=str, required=True)
    parser.add_argument("--vis_frame_stride", help="Name of the config file", type=int, required=False)
    parser.add_argument("--whether_multi_predictor", help="Name of the config file", type=bool, required=True)
    parser.add_argument("--max_ids_each_predictor", help="Name of the config file", type=int, required=False)
    args = parser.parse_args()
    if args.dataset != "DanceTrack" and args.dataset != "MOT17" and args.dataset != "MOT20":   
        raise NotImplementedError
    if args.split not in ["train", "val", "test"]:
        raise NotImplementedError
    if args.step_num <= 0:
        whether_only_init_first_frame_each_id = True
    else:
        whether_only_init_first_frame_each_id = False
    print(Fore.BLUE + f"whether_only_init_first_frame_each_id: {whether_only_init_first_frame_each_id};step: {args.step_num}")
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
    video_root_path = f"/data/zly/mot_data/{args.dataset}/{args.split}/"
    video_path_list = os.listdir(video_root_path)
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
    print(Fore.BLUE + f"video_path_list: {video_path_list}")
    if whether_only_init_first_frame_each_id:
        exp_name = "only_init_first_frame_each_id"
    else:
        exp_name = f"give_every_step{args.step_num}_frames_each_id"
    for video_sub_path in video_path_list:
        hydra_overrides_extra = [
        "++model.non_overlap_masks=" + "true"
        ]
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint,hydra_overrides_extra=hydra_overrides_extra)
        print(f"Processing {video_sub_path}")
        scene_base_path = os.path.join(video_root_path, video_sub_path) # 尺寸数据集的地方
        if "val" in video_root_path:
            exp_path = f"/data/zly/mot_data/{args.dataset}/{exp_name}/val"
        elif "train" in video_root_path:
            exp_path = f"/data/zly/mot_data/{args.dataset}/{exp_name}/train"
        elif "test" in video_root_path:
            exp_path = f"/data/zly/mot_data/{args.dataset}/{exp_name}/test"
        else:
            raise NotImplementedError
        scene_output_path = os.path.join(exp_path, video_sub_path)
        scene_output_img_path = os.path.join(scene_output_path, "mask")
        if "dancetrack" in video_root_path.lower() or "mot" in video_root_path.lower():
            video_dir = os.path.join(scene_base_path ,"img1")
        else:
            raise NotImplementedError
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG",'.png']
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        inference_state = predictor.init_state(video_path=video_dir, # 这里的设计是如果是多个predictor的话，就不用节省显存了
                                            #    offload_video_to_cpu=((not args.whether_multi_predictor) or args.dataset=="DanceTrack"),
                                               offload_video_to_cpu=True,
                                               offload_state_to_cpu=not args.whether_multi_predictor)
        if whether_use_pointjson_or_labeltxt == "json":
            jsons_path = os.path.join(scene_base_path, 'points')
            init_json = init_new_points_from_json(jsons_path, predictor,inference_state)
        elif whether_use_pointjson_or_labeltxt == "txt":
            txt_path = os.path.join(scene_base_path, "gt", "gt.txt")
            result_dict = init_new_points_from_real_label_txt(txt_path, predictor, inference_state,
                whether_only_init_first_frame_each_id = whether_only_init_first_frame_each_id,step=args.step_num,
                whether_multi_predictor = args.whether_multi_predictor)
            assert not (len(result_dict) == 0 and args.whether_multi_predictor == False) # 如果是单个predictor的话，应该result_dict为空
        else:
            raise NotImplementedError
        np_save_path_folder = os.path.join(scene_output_path, "npz_result")
        if not os.path.exists(np_save_path_folder):
            os.makedirs(np_save_path_folder)
        save_immediately = True
        if args.whether_multi_predictor:
            video_segments = {}  
            max_ids_each_predictor = args.max_ids_each_predictor if args.max_ids_each_predictor is not None else 10
            current_predictor_id_num = 0
            for obj_id, info in result_dict.items():
                for frame_idx, bbox in info.items():
                    predictor.add_new_points_or_box(inference_state, frame_idx, int(obj_id),box = bbox)
                current_predictor_id_num += 1
                if current_predictor_id_num >= max_ids_each_predictor:
                    propagate_and_extend_video_segments(video_segments, predictor, inference_state)
                    predictor.reset_state(inference_state)
                    current_predictor_id_num = 0
                    if save_immediately:
                        save_and_clear_video_segments(video_segments, np_save_path_folder)
            if save_immediately and current_predictor_id_num > 0:
                propagate_and_extend_video_segments(video_segments, predictor, inference_state)
                save_and_clear_video_segments(video_segments, np_save_path_folder)
        else:
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            if save_immediately:
                save_and_clear_video_segments(video_segments, np_save_path_folder)
        # for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        #     plt.figure(figsize=(6, 4))
        #     plt.title(f"frame {out_frame_idx}")
        #     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        #     output_pic_path = os.path.join(scene_output_path, "mask", f"{frame_names[out_frame_idx]}")
        #     if not os.path.exists(os.path.dirname(output_pic_path)):
        #         os.makedirs(os.path.dirname(output_pic_path))
        #     frame_masks_info = video_segments[out_frame_idx]
        #     mask = frame_masks_info.labels
        #     mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        #     for obj_id, obj_info in mask.items():
        #         mask_img[obj_info.mask == True] = obj_id
        #     mask_img = mask_img.numpy().astype(np.uint16)
        #     plt.axis("off")
        #     plt.close()
            
        if args.vis == "True":
            vis_frame_stride = args.vis_frame_stride if args.vis_frame_stride is not None else 15
            if not os.path.exists(scene_output_img_path):
                os.makedirs(scene_output_img_path)
            id_color_map = plt.get_cmap("tab20")  # Using matplotlib's tab20 colormap which has 20 distinct colors
            for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
                frame_image_path = os.path.join(video_dir, frame_names[out_frame_idx])
                frame_image = Image.open(frame_image_path)
                if save_immediately:
                    frame_masks_info = glob_and_read_npz(np_save_path_folder,out_frame_idx)
                else:
                    frame_masks_info = video_segments[out_frame_idx]
                
                # Create an RGB mask image with the same dimensions as the frame image
                mask_image = np.zeros((*frame_image.size[::-1], 3), dtype=np.uint8)
                
                for obj_id, obj_mask in frame_masks_info.items():
                    if len(obj_mask.shape) == 3:
                        obj_mask = obj_mask[0]
                    color = id_color_map(obj_id % 20)[:3]  # Retrieve color, mod with 20 to prevent index out of bounds
                    color = (np.array(color) * 255).astype(np.uint8)  # Convert from [0,1] to [0,255]
                    for c in range(3):  
                        mask_image[:, :, c][obj_mask] = color[c]
                
                # Combine the original image with the colored mask
                combined_image = cv2.addWeighted(np.array(frame_image), 0.7, mask_image, 0.3, 0)
                
                # Save the combined image
                output_image_path = os.path.join(scene_output_img_path, f"{frame_names[out_frame_idx]}")
                cv2.imwrite(output_image_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
            
        out_put_txt_path = os.path.join(exp_path,"track_results", f"{video_sub_path}.txt")
        if not os.path.exists(os.path.dirname(out_put_txt_path)):
            os.makedirs(os.path.dirname(out_put_txt_path))
        video_segments_to_evl_txt(video_segments, out_put_txt_path,np_save_path_folder,save_immediately=save_immediately,frame_names=frame_names)
        # for npz_path in glob.glob(os.path.join(np_save_path_folder, f"*.npz")):
        #     os.remove(npz_path)
        del video_segments, inference_state, predictor