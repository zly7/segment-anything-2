import os
from multiprocessing import Pool
from loguru import logger
logger.add("logs/change_txt_file.log",rotation="1 week")
'''
这个主要是控制这个ini文件的生成，尤其是rgb还是这个红外文件
'''
base_data_path = "/data2/zly/mot_data/MTMMC/train"
scene_list = os.listdir(base_data_path)

def process_txt_file(args):
    gt_path,gt_mot_path = args
    set_for_this_frame = set()
    if "MTMMC" in gt_mot_path:
        width, height = 1920, 1080
    else:
        raise NotImplementedError
    with open(gt_path, "r") as file_gt, open(gt_mot_path, "w") as file_gt_mot:
        last_frame_id = 0
        for line in file_gt:
            line = line.strip()  # 去掉换行符
            line_splits = line.split(",")
            frame_id = int(line_splits[0])
            obj_id = int(line_splits[1])
            bbox = [float(line_splits[2]), float(line_splits[3]),
                    float(line_splits[2]) + float(line_splits[4]), 
                    float(line_splits[3]) + float(line_splits[5]), obj_id]
            left_up = (int(bbox[0]), int(bbox[1]))
            right_down = (int(bbox[2]), int(bbox[3]))
            left_up = (max(0, left_up[0]), max(0, left_up[1]))
            left_up = (min(width - 1, left_up[0]), min(height - 1, left_up[1]))
            right_down = (max(0, right_down[0]), max(0, right_down[1]))
            right_down = (min(width - 1, right_down[0]), min(height - 1, right_down[1]))
            line_splits[2] = str(left_up[0])
            line_splits[3] = str(left_up[1])
            line_splits[4] = str(right_down[0]-left_up[0])
            line_splits[5] = str(right_down[1]-left_up[1])
            
            if frame_id != last_frame_id:
                set_for_this_frame = set()
                last_frame_id = frame_id    
            if obj_id in set_for_this_frame: # 有重复的情况跳过
                logger.warning(f"in the {gt_path} {frame_id} frame with id {obj_id} has duplicate errors.")
                continue
            else:
                set_for_this_frame.add(obj_id)
            new_line = ",".join(line_splits) + ",1,1,1"  # 在每行末尾添加 ",1,1,1"
            file_gt_mot.write(new_line + "\n")  # 写入新的 gt_mot.txt

    print(f"Processed {gt_path} and saved to {gt_mot_path}")

# 遍历每个scene
for scene_sub_path in scene_list:
    scene_path = os.path.join(base_data_path, scene_sub_path)
    camera_list = os.listdir(scene_path)
    _args_list = []
    # 遍历每个camera
    for camera_sub_path in camera_list:
        if "ini" in camera_sub_path:  # 跳过 ini 文件
            continue
        video_path = os.path.join(scene_path, camera_sub_path)
        gt_path = os.path.join(video_path, 'gt/gt.txt')
        gt_mot_path = os.path.join(video_path, 'gt/gt_mot>0.txt')
        _args_list.append((gt_path,gt_mot_path))
    
    with Pool(12) as pool:
        pool.map(process_txt_file, _args_list)
        
        

                    

        


