由于一个环境里面可能pip install -e .安装了两个SAM2，所以可能有一些问题，建议之后换一个环境
python setup.py clean --all
python setup.py build_ext --inplace 

## 2024-8-13
如果只init第一帧
CUDA_VISIBLE_DEVICES=1 python run_sam2.py --dataset DanceTrack --split train --step_num -1 --vis True

CUDA_VISIBLE_DEVICES=1 python run_sam2.py --dataset DanceTrack --split val --step_num -1 --vis True

## 2024-8-19 现在在init的时候相当于是只要那一帧有新物体就全部给出，然后主要就是分到多个物体不报显存
CUDA_VISIBLE_DEVICES=1 python run_sam2.py --dataset MOT17 --split train --step_num -1 --vis True --vis_frame_stride 1 --whether_multi_predictor True --max_ids_each_predictor 10 && CUDA_VISIBLE_DEVICES=1 python run_sam2.py --dataset MOT20 --split train --step_num -1 --vis True --vis_frame_stride 1 --whether_multi_predictor True --max_ids_each_predictor 10

CUDA_VISIBLE_DEVICES=0 python run_sam2.py --dataset MOT17 --split train --step_num 10 --vis True --vis_frame_stride 1 --whether_multi_predictor True --max_ids_each_predictor 10 && CUDA_VISIBLE_DEVICES=0 python run_sam2.py --dataset MOT20 --split train --step_num 10 --vis True --vis_frame_stride 1 --whether_multi_predictor True --max_ids_each_predictor 10

CUDA_VISIBLE_DEVICES=0 python run_sam2.py --dataset DanceTrack --split train --step_num 10 --vis True --vis_frame_stride 15 --whether_multi_predictor True --max_ids_each_predictor 10

CUDA_VISIBLE_DEVICES=1 python run_sam2.py --dataset DanceTrack --split train --step_num -1 --vis True --vis_frame_stride 15 --whether_multi_predictor True --max_ids_each_predictor 10

CUDA_VISIBLE_DEVICES=1 python run_sam2.py --dataset MOT20 --split train --step_num 10 --vis True --vis_frame_stride 1 --whether_multi_predictor True --max_ids_each_predictor 10

CUDA_VISIBLE_DEVICES=0 python run_sam2.py --dataset MOT20 --split train --step_num -1 --vis True --vis_frame_stride 1 --whether_multi_predictor True --max_ids_each_predictor 10

## 2024-8-30 现在的情况是id还是不能随意的换到不同的场景，并且重新把id-non overlap给设置了
CUDA_VISIBLE_DEVICES=1 python run_sam2_no_overlap.py --dataset MOT17 --split train --step_num 8 --vis True --vis_frame_stride 1 --gt_big_than_0 True
CUDA_VISIBLE_DEVICES=1 python run_sam2_no_overlap.py --dataset DanceTrack --split val --step_num 8 --vis True --vis_frame_stride 1

## 2024-9-7 run_sam2_no_overlap MTMMC

CUDA_VISIBLE_DEVICES=0 python run_sam2_no_overlap.py --dataset MTMMC --split val --step_num 10 --vis True --vis_frame_stride 1 --gt_big_than_0 True

## 内存审查 mprof 随着时间变化
CUDA_VISIBLE_DEVICES=0 mprof run run_sam2_no_overlap.py --dataset MTMMC --split val --step_num 10 --vis False --vis_frame_stride 1
CUDA_VISIBLE_DEVICES=0 python -m memory_profiler run_sam2_no_overlap.py --dataset MTMMC --split val --step_num 10 --vis False --vis_frame_stride 1
python run_sam2_no_overlap.py --dataset MTMMC --split val --step_num 10 --vis False

## 
torchrun --nproc_per_node=2 trainer_ddp.py --config_path ./train_config/train_large.yaml
python3 trainer_ddp.py --config_path ./train_config/train_large.yaml

python3 ./pedestrainSAM/trainer_ddp.py --local_rank 0 --device_index 1 --config_path ./train_config/train_large.yaml

python3 -m pedestrainSAM.test_ochuman
CUDA_VISIBLE_DEVICES=1 python -m pedestrainSAM.test_ochuman

CUDA_VISIBLE_DEVICES=1 python -m pedestrainSAM.test_ochuman_optimu


## 启动optuna 可视化网站
optuna-dashboard sqlite:///optuna_study.db

