import random
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional, Union
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .selected_data_maker import SelectedData
from ...datasets import SelectedDataset


class ReplayBuffer(SelectedDataset):
    """Replay buffer class. It stores selected data.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of selected samples. A number <= 0 means unlimited. Defaults to 0.
    """

    def __init__(
        self, 
        dataset_name: str, 
        tokenizer, 
        strategy, 
        input_template: dict,
        sample_batch_size: int, 
        limit: int = 0, 
    ) -> None:
        super().__init__(
            dataset_names=dataset_name,
            tokenizer=tokenizer,
            strategy=strategy,
            mode='train',
            input_template=input_template,
            shot_type='zero_shot',
            is_dataset=False,
        )
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.dataset = []
        self.dataset_with_gts = []

    @torch.no_grad()
    def append(self, selected_data: SelectedData) -> None:
        questions = selected_data.questions
        targets = selected_data.answers
        gts = selected_data.gts
        preprocessed_data = [(question, self.preprocess_prompt(question), target, gt) for question, target, gt in zip(questions, targets, gts)]
        self.dataset.extend(preprocessed_data)
        if self.limit > 0:
            samples_to_remove = len(self.dataset) - self.limit
            if samples_to_remove > 0:
                self.dataset = self.dataset[samples_to_remove:]

    def clear(self) -> None:
        self.dataset.clear()

    @torch.no_grad()
    def sample(self, sample_batch_size=None) -> SelectedData:
        batch = random.sample(self.dataset, self.sample_batch_size)
        selected_data = SelectedData(
            questions=[question for (question, _, _) in batch],
            prompts=[prompt for (_, prompt, _) in batch],
            gts=[None] * len(batch),
            answers=[answer for (_, _, answer) in batch],
            info={},
        )
        return selected_data
    
    def gather(self) -> None:
        world_size = torch.distributed.get_world_size()
        gatherd_replay_buffers_dataset = [None] * world_size
        torch.distributed.all_gather_object(gatherd_replay_buffers_dataset, self.dataset)
        self.dataset = []
        for replay_buffer_dataset in gatherd_replay_buffers_dataset:
            self.dataset.extend(replay_buffer_dataset)
        del gatherd_replay_buffers_dataset

    def save(self, path: str) -> None:
        # save dataset to a json file
        if self.strategy.is_rank_0():
            json_dataset = {}
            for i, (question, prompt, target, gt) in enumerate(self.dataset):
                json_dataset[i] = {
                    'question': question,
                    'prompt': prompt,
                    'target': target,
                    'gt': gt,
                }
            with open(path, 'w') as f:
                json.dump(json_dataset, f, indent=4)

    def load(self, path: str) -> None:
        # load dataset from a json file
        with open(path, 'r') as f:
            json_dataset = json.load(f)
        self.dataset = []
        for i in tqdm(json_dataset, desc='Loading replay buffer'):
            self.dataset.append(
                (
                    json_dataset[i]['question'],
                    json_dataset[i]['prompt'],
                    json_dataset[i]['target'],
                    json_dataset[i]['gt'],
                )
            )

    def is_empty(self) -> bool:
        return len(self.dataset) == 0
    
    def is_full(self) -> bool:
        return self.limit > 0 and len(self.dataset) >= self.limit

    def __len__(self) -> int:
        return len(self.dataset)


    # def collate_fn(self, batch) -> SelectedData:
    #     selected_data = make_selected_data_batch(batch, self.packing_samples)
    #     return selected_data