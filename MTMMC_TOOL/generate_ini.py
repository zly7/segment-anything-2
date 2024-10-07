import os
'''
这个主要是控制这个ini文件的生成，尤其是rgb还是这个红外文件
'''
base_data_path = "/data2/zly/mot_data/MTMMC/train"
scene_list = os.listdir(base_data_path)

# 遍历每个scene
for scene_sub_path in scene_list:
    scene_path = os.path.join(base_data_path, scene_sub_path)
    camera_list = os.listdir(scene_path)
    
    # 遍历每个camera
    for camera_sub_path in camera_list:
        if "ini" in camera_sub_path:
            continue
        video_path = os.path.join(scene_path, camera_sub_path)
        seqinfo_file_path = os.path.join(video_path, 'seqinfo.ini')
        content = f"""[Sequence]
name={scene_sub_path}/{camera_sub_path}
imDir=rgb
frameRate=23
seqLength=7362
imWidth=1920
imHeight=1080
imExt=.jpg"""
        with open(seqinfo_file_path, 'w') as f:
            f.write(content)

        print(f'Successfully created seqinfo.ini for {scene_sub_path}/{camera_sub_path}')


