import random
import torch
from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, main_datasets, auxiliary_datasets, auxiliary_ratio=[0.05]):
        """
        将多个主数据集与多个辅助数据集组合成一个数据集。
        主数据集被完全迭代，辅助数据集按照指定比例随机采样。

        Args:
            main_datasets (list of Dataset): 主数据集列表。
            auxiliary_datasets (list of Dataset): 辅助数据集列表。
            auxiliary_ratio (list of float): 每个辅助数据集采样比例，相对于所有主数据集的总长度。
                                      例如 0.05 表示每个辅助数据集采样 5% 的主数据集总长度。
        """
        assert isinstance(main_datasets, list) and all(isinstance(d, Dataset) for d in main_datasets), \
            "main_datasets 应该是 Dataset 对象的列表。"
        assert isinstance(auxiliary_datasets, list) and all(isinstance(d, Dataset) for d in auxiliary_datasets), \
            "auxiliary_datasets 应该是 Dataset 对象的列表。"
        assert isinstance(auxiliary_ratio, list) , \
            "auxiliary_ratio 必须是 0 到 1 之间的浮点数。"

        self.main_datasets = main_datasets
        self.auxiliary_datasets = auxiliary_datasets
        self.auxiliary_ratio = auxiliary_ratio

        self.main_lengths = [len(d) for d in self.main_datasets]
        self.main_cumulative_lengths = torch.cumsum(torch.tensor(self.main_lengths), dim=0).tolist()
        self.total_main_length = self.main_cumulative_lengths[-1] if self.main_cumulative_lengths else 0

        self.num_auxiliary_samples = [int(self.total_main_length * self.auxiliary_ratio[i]) for i in range(len(self.auxiliary_datasets))]
        self.total_auxiliary_length = sum(self.num_auxiliary_samples)
        self.auxiliary_start_indices = []
        cumulative = self.total_main_length
        for num in self.num_auxiliary_samples:
            self.auxiliary_start_indices.append(cumulative)
            cumulative += num
        self.total_length = cumulative

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        if index < self.total_main_length:
            # 找到对应的主数据集及其在该数据集中的局部索引
            dataset_idx = self._find_main_dataset(index)
            if dataset_idx == 0:
                local_idx = index
            else:
                local_idx = index - self.main_cumulative_lengths[dataset_idx - 1]
            return self.main_datasets[dataset_idx][local_idx]
        else:
            # 对应辅助数据集的样本
            aux_index = index - self.total_main_length
            # Determine which auxiliary dataset this aux_index belongs to
            for aux_dataset_idx, num_samples in enumerate(self.num_auxiliary_samples):
                if aux_index < num_samples:
                    chosen_aux_dataset = self.auxiliary_datasets[aux_dataset_idx]
                    break
                aux_index -= num_samples
            else:
                # 如果没有匹配到，默认选择最后一个辅助数据集
                chosen_aux_dataset = self.auxiliary_datasets[-1]
                aux_index = len(self.auxiliary_datasets[-1]) - 1  # 最后一个索引

            # 随机选择一个样本索引
            if len(chosen_aux_dataset) == 0:
                raise ValueError(f"辅助数据集 {aux_dataset_idx} 为空，无法采样。")
            sample_idx = random.randint(0, len(chosen_aux_dataset) - 1)
            return chosen_aux_dataset[sample_idx]
    
    def get_certain_sample(self, index, positive=True):
        """
        用于从所有数据集中获取正样本的方法。

        Args:
            index (int): 全局索引。

        Returns:
            dict: 正样本字典。
        """
        if index < self.total_main_length:
            # 找到对应的主数据集及其在该数据集中的局部索引
            dataset_idx = self._find_main_dataset(index)
            if dataset_idx == 0:
                local_idx = index
            else:
                local_idx = index - self.main_cumulative_lengths[dataset_idx - 1]
            if positive:
                return self.main_datasets[dataset_idx].get_positive_sample(local_idx)
            else:
                return self.main_datasets[dataset_idx].get_negative_sample(local_idx)
        else:
            aux_index = index - self.total_main_length
            for aux_dataset_idx, num_samples in enumerate(self.num_auxiliary_samples):
                if aux_index < num_samples:
                    chosen_aux_dataset = self.auxiliary_datasets[aux_dataset_idx]
                    break
                aux_index -= num_samples
            else:
                chosen_aux_dataset = self.auxiliary_datasets[-1]
                aux_index = len(self.auxiliary_datasets[-1]) - 1
            sample_idx = random.randint(0, len(chosen_aux_dataset) - 1)
            if positive:
                return chosen_aux_dataset.get_positive_sample(sample_idx)
            else:
                return chosen_aux_dataset.get_negative_sample(sample_idx)

    def _find_main_dataset(self, index):
        """
        根据全局索引找到对应的主数据集索引。

        Args:
            index (int): 全局索引。

        Returns:
            int: 对应的主数据集索引。
        """
        for i, cum_len in enumerate(self.main_cumulative_lengths):
            if index < cum_len:
                return i
        # 理论上不会到达这里，因为 index < self.total_main_length 已经保证
        return len(self.main_cumulative_lengths) - 1

    def set_epoch(self, epoch=None):
        """
        可选方法，用于在每个 epoch 开始时设置随机种子，以确保不同 epoch 的辅助采样不同。
        如果需要，可以在训练循环中调用此方法。

        Args:
            epoch (int, optional): 当前的 epoch 数，作为随机种子的一部分。
        """
        if epoch is not None:
            random.seed(epoch)
        else:
            random.seed()