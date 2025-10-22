from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none
from typing import Optional
import pandas as pd
import glob
import os
import re
import json
import time

from opencompass.datasets import MATHDataset, GSM8KDataset
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_infer_cfg as gsm8k_few_shot_infer_cfg, gsm8k_reader_cfg as gsm8k_few_shot_reader_cfg, gsm8k_eval_cfg as gsm8k_few_shot_eval_cfg
from opencompass.configs.datasets.gsm8k.gsm8k_0shot_gen_a58960 import gsm8k_infer_cfg as gsm8k_zero_shot_infer_cfg, gsm8k_reader_cfg as gsm8k_zero_shot_reader_cfg, gsm8k_eval_cfg as gsm8k_zero_shot_eval_cfg
from opencompass.configs.datasets.math.math_gen_265cce import math_infer_cfg as math_few_shot_infer_cfg, math_eval_cfg as math_few_shot_eval_cfg, math_reader_cfg as math_few_shot_reader_cfg
from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_infer_cfg as math_zero_shot_infer_cfg, math_eval_cfg as math_zero_shot_eval_cfg, math_reader_cfg as math_zero_shot_reader_cfg
from opencompass.datasets.math import extract_boxed_answer

from .parse_gsm_symbolic import generate_random_problems

MATH_DATASET_PATH = "./data/math"
GSM8K_DATASET_PATH = "./data/gsm8k"
GENERATED_GSM8K_DATASET_PATH = "./ml-gsm-symbolic/generated_data"
GEN_GSM_TEMPLATE_PATH = "./ml-gsm-symbolic/templates"
NUMINAMATH_DATASET_PATH = "./NuminaMath-CoT/data"


def preprocess_prompt(question, input_template=None, input_name="question", apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = question
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        if input_template:
            if isinstance(input_template, str):
                prompt = input_template.format(question)  # -> str
            elif isinstance(input_template, PromptTemplate):
                prompt = input_template.generate_item({input_name: question})  # -> PromptList
                res = []
                for i, item in enumerate(prompt):
                    if 'role' in item:
                        if isinstance(item, str):
                            res.append(item)
                        elif isinstance(item, dict):
                            if 'prompt' in item:
                                res.append(item['prompt'])
                prompt = "\n".join(res)
    return prompt


def preprocess_answer_for_NuminaMath(ref):
    assert isinstance(ref, str)
    ref = ref.replace("\boxed", "\\boxed")
    ref = ref.replace("\frac", "\\frac")
    ref = extract_boxed_answer(ref)
    if ref is None:  return None
    # 筛选出不合法的答案：文本，等式，带有冒号，长度为1且不是数字，区间
    # if "\\text" in ref:  return None
    # if "=" in ref:  return None
    # if ":" in ref:  return None
    # if len(ref) == 1 and not ref.isdigit():  return None
    # if re.search(r'*\[.*?\]', ref):  return None
    return ref


class SelectedDataset(Dataset):
    """
    Dataset for selected dataset

    Args:
        dataset_name: dataset name to use: Choices(["math", "gsm8k"])
        tokenizer: tokenizer for generative model
        strategy: strategy for generative model
        mode: mode for dataset: Choices(["train", "test"])
        shot_type: type of shot to use: Choices(["zero_shot", "few_shot"])
    """

    def __init__(
        self,
        dataset_names,
        tokenizer,
        strategy,
        mode='train',
        input_template=None,
        shot_type='few_shot',
        is_dataset=True,
    ) -> None:
        super().__init__()
        if type(dataset_names) == str:
            dataset_names = [dataset_names]
        self.basic_dataset = []
        for dataset_name in dataset_names:
            if dataset_name == "math":
                basic_dataset = MATHDataset.load(path=MATH_DATASET_PATH)[mode] if is_dataset else None
                dataset_type='math'
            elif dataset_name == "gsm8k":
                basic_dataset = GSM8KDataset.load(path=GSM8K_DATASET_PATH)[mode] if is_dataset else None
                dataset_type = 'gsm8k'
            elif dataset_name == "NuminaMath":
                assert mode == 'train'
                file_pattern = NUMINAMATH_DATASET_PATH + "/train-*.parquet"
                file_list = glob.glob(file_pattern)
                dataframes = [pd.read_parquet(file) for file in file_list]
                combined_df = pd.concat(dataframes, ignore_index=True)
                basic_dataset = combined_df.to_dict(orient='records')
                dataset_type='math'
            elif "generated_gsm8k" in dataset_name:  # Choices(["generated_gsm8k_symbolic", "generated_gsm8k_p1", "generated_gsm8k_p2"])
                level = dataset_name.split("_")[-1]
                path = os.path.join(GENERATED_GSM8K_DATASET_PATH, f"GSM_{level}.jsonl")
                with open(path, 'r', encoding='utf-8') as f:
                    basic_dataset = [json.loads(line) for line in f]
                dataset_type = 'gsm8k'
            elif "gen_gsm" in dataset_name:  # Choices(["gen_gsm_symbolic_{num}", "gen_gsm_p1_{num}", "gen_gsm_p2_{num}"])
                level = dataset_name.split("_")[-2]
                num_to_generate = int(dataset_name.split("_")[-1]) * 7473
                # dataset 放到 strategy.args.save_path 的前三个父文件夹下
                # dataset_path = os.path.dirname(os.path.dirname(os.path.dirname(strategy.args.save_path)))
                dataset_path = os.path.dirname(strategy.args.save_path)
                temp_path = os.path.join(dataset_path, f"synthetic_GSM_{level}.jsonl")
                while not os.path.exists(temp_path):
                    if strategy.is_rank_0():
                        template_folder = os.path.join(GEN_GSM_TEMPLATE_PATH, level)
                        basic_dataset = generate_random_problems(template_folder, num_to_generate=num_to_generate)
                        # Save the generated dataset to a temporary file
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            for item in basic_dataset:
                                f.write(json.dumps(item) + '\n')
                        del basic_dataset
                    time.sleep(0.1)
                with open(temp_path, 'r', encoding='utf-8') as f:
                    basic_dataset = [json.loads(line) for line in f]
                dataset_type = 'gsm8k'
            else:
                raise ValueError(f"Invalid dataset_name: {dataset_names}")
            if basic_dataset is not None:
                self.basic_dataset.extend(basic_dataset)
        
        self.infer_cfg, self.eval_cfg, self.reader_cfg = self._get_cfg(dataset_type=dataset_type, shot_type=shot_type)
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.dataset_postprocess = self.eval_cfg.get("dataset_postprocessor", {}).get("type", None)
        self.data_postprocess = self.eval_cfg.get("pred_postprocessor", {}).get("type", None)
        evaluator_cfg = self.eval_cfg["evaluator"]
        evaluator_type = evaluator_cfg["type"]
        self.evaluator = evaluator_type(**{k: v for k, v in evaluator_cfg.items() if k != "type"})
        self.input_name = self.reader_cfg.get("input_columns", [None])[0]
        self.output_name = self.reader_cfg.get("output_column", "")

        # input template
        if shot_type == 'few_shot':
            self.prompt_template = PromptTemplate(template=self.infer_cfg["prompt_template"]["template"])
        elif shot_type == 'zero_shot':
            # self.prompt_template = input_template
            self.prompt_template = PromptTemplate(template=self.infer_cfg["prompt_template"]["template"])
        else:
            raise ValueError(f"Invalid shot_type: {shot_type}")
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template

        if is_dataset:
            self.dataset = []
            for data in tqdm(self.basic_dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
                # preprocessed_data = preprocess_data(data, tokenizer, self.prompt_template, input_name, output_name, apply_chat_template)
                question = data[self.input_name]
                prompt = preprocess_prompt(question, self.prompt_template, self.input_name, self.apply_chat_template)
                ground_truth = data[self.output_name]
                if dataset_names == "NuminaMath":
                    ground_truth = preprocess_answer_for_NuminaMath(ground_truth)
                if ground_truth is None:
                    continue
                target = ground_truth
                preprocessed_data = (question, prompt, target, ground_truth)
                self.dataset.append(preprocessed_data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def preprocess_prompt(self, question) -> str:
        return preprocess_prompt(question, self.prompt_template, self.input_name, self.apply_chat_template)
    
    def _get_cfg(self, dataset_type='gsm8k', shot_type='few_shot'):
        if dataset_type == 'gsm8k':
            if shot_type == 'few_shot':
                return gsm8k_few_shot_infer_cfg, gsm8k_few_shot_eval_cfg, gsm8k_few_shot_reader_cfg
            elif shot_type == 'zero_shot':
                return gsm8k_zero_shot_infer_cfg, gsm8k_zero_shot_eval_cfg, gsm8k_zero_shot_reader_cfg
            else:
                raise ValueError(f"Invalid shot_type: {shot_type}")
        elif dataset_type == 'math':
            if shot_type == 'few_shot':
                return math_few_shot_infer_cfg, math_few_shot_eval_cfg, math_few_shot_reader_cfg
            elif shot_type == 'zero_shot':
                return math_zero_shot_infer_cfg, math_zero_shot_eval_cfg, math_zero_shot_reader_cfg
            else:
                raise ValueError(f"Invalid shot_type: {shot_type}")
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")