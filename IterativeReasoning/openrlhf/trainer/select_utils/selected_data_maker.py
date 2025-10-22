import logging
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

@dataclass
class SelectedData:
    """
    SelectedData is a batch of data that is selected from the buffer.
    """
    questions: Union[str, List[str]]
    prompts: Union[str, List[str]]
    gts: Union[str, List[str]]
    answers: Union[str, List[str]]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        # self.prompts = self.prompts.to(device)
        # self.gts = self.gts.to(device)
        # self.answers = self.answers.to(device)
        pass

    def pin_memory(self):
        # self.prompts = self.prompts.pin_memory()
        # self.gts = self.gts.pin_memory
        # self.answers = self.answers.pin_memory()
        pass

    def __len__(self):
        return len(self.prompts)


class SelectedDataMaker(ABC):
    def __init__(
        self, 
        generative_model: nn.Module,
        reward_model: nn.Module,
        tokenizer,
        prompt_max_len: int,
        strategy=None, 
        **generative_kwargs, 
    ) -> None:
        self.generative_model = generative_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.strategy = strategy
        self.generative_kwargs = generative_kwargs

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}
    
    @torch.no_grad()
    def make_selected_data(self, questions: Union[str, List[str]], prompts: Union[str, List[str]], gts: Union[str, List[str]]) -> Tuple[SelectedData, SelectedData]:
        """
        Generate answers for all prompts and select data based on the reward model.
        """
        # Generate answers
        args = self.strategy.args
        prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in prompts], [])
        prompt_lens = [len(prompt) for prompt in prompts]
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
        outputs = self.generative_model.generate(**inputs, tokenizer=self.tokenizer, **self.generative_kwargs)
        generations = self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)
        answers = [generattion[len_:] for generattion, len_ in zip(generations, prompt_lens)]
        generated_data = SelectedData(questions=questions, prompts=prompts, answers=answers, gts=gts, info={})

        # Select data with the reward model
        if self.reward_model:
            selected_data = self.reward_model.select(generated_data)
        else:
            selected_data = generated_data

        return generated_data, selected_data