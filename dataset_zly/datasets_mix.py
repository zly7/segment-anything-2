import random
import torch
from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, datasets, ratios):
        """
        将多个数据集按照指定的比例混合成一个数据集。

        Args:
            datasets (list of Dataset): 要混合的多个数据集。
            ratios (list of float): 每个数据集的混合比例，比例之和应为1。
        """
        assert len(datasets) == len(ratios), "数据集数量和比例数量必须一致。"
        assert abs(sum(ratios) - 1.0) < 1e-6, "混合比例之和必须为1。"

        self.datasets = datasets
        self.ratios = ratios
        self.cumulative_ratios = torch.cumsum(torch.tensor(ratios), dim=0).tolist()

    def __len__(self):
        """
        定义数据集的长度。这里选择所有子数据集长度的最大值，
        以确保每个子数据集的样本都有机会被采样。
        """
        return max(len(d) for d in self.datasets)

    def __getitem__(self, index):
        """
        根据混合比例选择一个子数据集，并从中随机选择一个样本。

        Args:
            index (int): 样本索引（在本实现中未直接使用）。

        Returns:
            dict: 从选择的数据集中获取的样本。
        """
        rand = random.random()
        for i, threshold in enumerate(self.cumulative_ratios):
            if rand < threshold:
                chosen_dataset = self.datasets[i]
                break
        else:
            chosen_dataset = self.datasets[-1]

        # 从选择的数据集中随机选择一个样本
        dataset_length = len(chosen_dataset)
        sample_idx = random.randint(0, dataset_length - 1)
        sample = chosen_dataset[sample_idx]
        return sample
