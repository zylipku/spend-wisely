import math
import os
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import copy
from copy import deepcopy

import ray
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import cycle

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss
from openrlhf.models.utils import masked_mean
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.datasets.utils import zero_pad_sequences

from .select_utils import SelectedDataMaker, ReplayBuffer, SelectedData

from opencompass.datasets import BaseDataset


class SelectTrainer(ABC):
    """
        Trainer for Iterative Learning on Selected Data.
    """

    def __init__(
        self,
        strategy, 
        generative_model,
        reward_model,
        optim, 
        scheduler,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        gradient_checkpointing: bool = False,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 1024,
        # dataloader_pin_memory: bool = True,
        **generate_kwargs,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.generative_model = generative_model
        self.reward_model = reward_model
        self.optim = optim
        self.scheduler = scheduler
        # self.dataloader_pin_memory = dataloader_pin_memory
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.gradient_checkpointing = gradient_checkpointing
        self.generate_kwargs = generate_kwargs

        self.loss_fn = GPTLMLoss()

        self.selected_data_maker = SelectedDataMaker(
            strategy=self.strategy,
            generative_model=self.generative_model,
            reward_model=self.reward_model,
            tokenizer=tokenizer,
            prompt_max_len=prompt_max_len,
            **self.generate_kwargs,
        )
        # packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = ReplayBuffer(
            dataset_name=self.args.dataset_name,
            tokenizer=self.tokenizer,
            strategy=self.strategy,
            input_template=self.args.input_template,
            sample_batch_size=micro_train_batch_size,
            limit=buffer_limit
        )
        if self.args.resume_replay_buffer:
            self.replay_buffer.load(self.args.resume_replay_buffer)
        
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(
        self, 
        args, 
        train_dataset, 
        train_dataloader, 
        eval_datasets,
        eval_dataloaders,
        iterative_policy, 
    ):
        self.train_dataloader = cycle(train_dataloader)
        self.eval_datasets = eval_datasets
        self.eval_dataloaders = eval_dataloaders
        training_step = 0
        iteration = 1
        self.replay_buffer.limit = iterative_policy.num_samples[iteration] // self.strategy.world_size
        # num_generated = 0
        # num_selected = 0

        for step, (questions, prompts, targets, gts) in enumerate(self.train_dataloader):
            generate_pbar = tqdm(
                # total=iterative_policy.max_steps,
                desc="Generate and Select",
                disable=not self.strategy.is_rank_0(),
            )

            # 1. Generate: Prepare Prompts
            generated_data, selected_data = self.selected_data_maker.make_selected_data(questions, prompts, gts)
            num_generated = len(generated_data)
            num_selected = len(selected_data)
            generate_status = {"step": step, "iteration": iteration, "num_generated": num_generated, "num_selected": num_selected, "replay_buffer_len": len(self.replay_buffer), "limit": self.replay_buffer.limit}
            generate_pbar.set_postfix(generate_status)
            generate_pbar.update()
            self._save_logs(step, 'generate', generate_status)
            self.replay_buffer.append(selected_data)
            if not self.strategy.all_reduce(self.replay_buffer.is_full(), op="sum") == self.strategy.world_size:
                continue
            # if not self.strategy.all_reduce(len(self.replay_buffer), op="sum") >= iterative_policy.num_samples[iteration]:
            #     continue
            # Save selected data
            if self.args.save_selected_data:
                torch.distributed.barrier()
                self.replay_buffer.gather()
                self.replay_buffer.save(os.path.join(args.save_path, f"selected_dataset.json"))
                raise NotImplementedError
            torch.distributed.barrier()

            # 2. Train
            torch.cuda.empty_cache()
            train_status = self.select_train(iteration, start_training_step=training_step, is_gather=False)
            self.replay_buffer.clear()
            torch.cuda.empty_cache()
            training_step += iterative_policy.num_samples[iteration] // args.train_batch_size

            # 3. Evaluate
            if args.eval_steps <= 0:
                for eval_dataset, eval_dataloader, eval_dataset_name in zip(eval_datasets, eval_dataloaders, args.eval_dataset_name):
                    self.evaluate(eval_dataset, eval_dataloader, steps=training_step, dataset_name=eval_dataset_name)

            # 4. Save checkpoints
            if args.save_steps == 0:
                self._save_checkpoint(args, f"step_{training_step}")

            # 5. Update iteration
            iteration = iteration + 1
            self.replay_buffer.limit = math.ceil(iterative_policy.num_samples[iteration] / self.strategy.world_size)
            if iteration > len(iterative_policy.num_samples):
                break

        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


    def train_on_GT(
        self, 
        args, 
        dataset, 
        train_dataloader, 
        eval_datasets,
        eval_dataloaders,
    ):
        # Prepare dataloader
        self.train_dataloader = cycle(train_dataloader)
        self.eval_datasets = eval_datasets
        self.eval_dataloaders = eval_dataloaders
        device = torch.cuda.current_device()
        status_list = []
        # Train step
        status_list = []
        train_pbar = tqdm(
            self.train_dataloader,
            desc=f"Train",
            disable=not self.strategy.is_rank_0(),
        )
        for step, (questions, prompts, targets, gts) in enumerate(train_pbar):
            answers = [dataset.dataset_postprocess(gt) for gt in gts]
            status = self.training_step(prompts, answers)
            status_list.append(status)
            train_pbar.set_postfix(status)
            if (step+1) % self.strategy.accumulated_gradient == 0:
                status_mean = status_list[0]
                for m in status_list[1:]:
                    for k, v in m.items():
                        status_mean[k] += v
                for k in status_mean.keys():
                    status_mean[k] /= len(status_list)
                self._save_logs((step+1) // self.strategy.accumulated_gradient, 'train', status_mean)
                list.clear(status_list)
                # Evaluate
                if self.args.eval_steps > 0 and ((step+1) // self.strategy.accumulated_gradient) % self.args.eval_steps == 0:
                    for eval_dataset, eval_dataloader, eval_dataset_name in zip(self.eval_datasets, self.eval_dataloaders, self.args.eval_dataset_name):
                        self.evaluate(eval_dataset, eval_dataloader, steps=(step+1) // self.strategy.accumulated_gradient, dataset_name=eval_dataset_name)
                # Save checkpoints
                # if self.args.save_steps > 0 and (start_training_step + (step+1) // self.strategy.accumulated_gradient) % self.args.save_steps == 0:
                #     self._save_checkpoint(self.args, f"step_{start_training_step + (step+1) // self.strategy.accumulated_gradient}")
            if (step+1) // self.strategy.accumulated_gradient >= 1000:
                break
        return None


    def sft_wo_selection(
        self,
        args,
        dataset,
        generate_dataloader,
    ):
        # 1. Generate
        is_gather = not self.replay_buffer.is_empty()
        if self.replay_buffer.is_empty():
            self.generative_model.eval()
            for (questions, prompts, _, gts) in tqdm(generate_dataloader, desc="Generating", disable=not self.strategy.is_rank_0()):
                data = self.selected_data_maker.make_selected_data(questions, prompts, gts)
                self.replay_buffer.append(data)
            self.generative_model.train()
        torch.distributed.barrier()
        # 2. Train
        torch.cuda.empty_cache()
        status = self.select_train(iteration=1, is_gather=is_gather)
        self.replay_buffer.clear()
        torch.cuda.empty_cache()

        # logs/checkpoints
        self._save_checkpoint(args, "generative_model")

        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


    def sft_w_selection(
        self,
        args,
        dataset,
        generate_dataloader,
    ):
        # 1. Generate
        is_gather = not self.replay_buffer.is_empty()
        if self.replay_buffer.is_empty():
            self.generative_model.eval()
            for (questions, prompts, _, gts) in tqdm(generate_dataloader, desc="Generating", disable=not self.strategy.is_rank_0()):
                data = self.selected_data_maker.make_selected_data(questions, prompts, gts)
                self.replay_buffer.append(data)
            self.generative_model.train()
            torch.distributed.barrier()
            # 2. Gather
            self.replay_buffer.gather()
        # 3. Train
        torch.cuda.empty_cache()
        status = self.select_train(iteration=1, is_gather=is_gather)
        torch.distributed.barrier()
        self.replay_buffer.clear()
        torch.cuda.empty_cache()

        # logs/checkpoints
        self._save_checkpoint(args, "generative_model")

        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


    def generate(
        self,
        args,
        dataset,
        generate_dataloader,
    ):
        self.generative_model.eval()
        self.generated_replay_buffer = copy.deepcopy(self.replay_buffer)
        self.selected_replay_buffer = copy.deepcopy(self.replay_buffer)
        for i, (questions, prompts, _, gts) in enumerate(tqdm(generate_dataloader, desc="Generating", disable=not self.strategy.is_rank_0())):
            # Generate
            generated_data, selected_data = self.selected_data_maker.make_selected_data(questions, prompts, gts)
            self.generated_replay_buffer.append(generated_data)
            self.selected_replay_buffer.append(selected_data)
        torch.distributed.barrier()
        self.generated_replay_buffer.gather()
        self.selected_replay_buffer.gather()
        torch.distributed.barrier()
        self.generated_replay_buffer.save(os.path.join(args.save_path, "generated_dataset.json"))
        self.selected_replay_buffer.save(os.path.join(args.save_path, "selected_dataset.json"))

    
    def select_train(self, iteration=0, start_training_step=0, is_gather=False):
        # replay buffer may be empty at first, we should rebuild at each training
        if is_gather:
            dataloader = self.strategy.setup_dataloader(
                self.replay_buffer,
                batch_size=self.replay_buffer.sample_batch_size,
                pin_memory=False,
                shuffle=True,
                drop_last=True,
            )
        else:
            dataloader = DataLoader(
                self.replay_buffer,
                batch_size=self.replay_buffer.sample_batch_size,
                shuffle=True,
                drop_last=True,
                # collate_fn=self.replay_buffer.collate_fn,
            )
        status_list = []
        train_pbar = tqdm(
            dataloader,
            desc=f"Train [{iteration}]",
            disable=not self.strategy.is_rank_0(),
        )
        for step, (questions, prompts, targets, gts) in enumerate(train_pbar):
            status = self.training_step(prompts, targets)
            status_list.append(status)
            train_pbar.set_postfix(status)
            if (step+1) % self.strategy.accumulated_gradient == 0:
                status_mean = status_list[0]
                for m in status_list[1:]:
                    for k, v in m.items():
                        status_mean[k] += v
                for k in status_mean.keys():
                    status_mean[k] /= len(status_list)
                self._save_logs(start_training_step + (step+1) // self.strategy.accumulated_gradient, 'train', status_mean)
                list.clear(status_list)
                # Evaluate
                if self.args.eval_steps > 0 and (start_training_step + (step+1) // self.strategy.accumulated_gradient) % self.args.eval_steps == 0:
                    for eval_dataset, eval_dataloader, eval_dataset_name in zip(self.eval_datasets, self.eval_dataloaders, self.args.eval_dataset_name):
                        self.evaluate(eval_dataset, eval_dataloader, steps=start_training_step + (step+1) // self.strategy.accumulated_gradient, dataset_name=eval_dataset_name)
                # Save checkpoints
                if self.args.save_steps > 0 and (start_training_step + (step+1) // self.strategy.accumulated_gradient) % self.args.save_steps == 0:
                    self._save_checkpoint(self.args, f"step_{start_training_step + (step+1) // self.strategy.accumulated_gradient}")
        return None

    
    def training_step(self, prompts, targets):
        status = {}

        # Prepare inputs, ref to sft_dataset.py
        prompt_id_lens = []
        input_ids = []
        attention_masks = []
        for (prompt, target) in zip(prompts, targets):
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.args.max_len,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_id_lens.append(prompt_token["attention_mask"].int().sum().item())
            prompt = prompt.rstrip("\n") + "\n"
            text = (prompt + target).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
            input_token = self.tokenizer(
                text,
                max_length=self.args.max_len,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            # to avoid EOS_token truncation
            input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            input_token["attention_mask"][0][-1] = True
            input_ids.append(input_token["input_ids"])
            attention_masks.append(input_token["attention_mask"])
        # Tokenize by right padding
        inputs = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_mask = zero_pad_sequences(attention_masks, "right")
        inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
        attention_mask = attention_mask.to(torch.cuda.current_device()).squeeze(1)

        # Forward pass
        output = self.generative_model(input_ids=inputs, attention_mask=attention_mask, return_output=True)

        # Calculate loss
        labels = torch.where(
            attention_mask.bool(),
            inputs,
            self.loss_fn.IGNORE_INDEX,
        )
        for label, source_len in zip(labels, prompt_id_lens):
            label[:source_len] = self.loss_fn.IGNORE_INDEX
        loss = self.loss_fn(output.logits, labels)

        # Backward pass
        self.strategy.backward(loss, self.generative_model, self.optim)
        self.strategy.optimizer_step(self.optim, self.generative_model, self.scheduler)

        # status
        status = {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
        return status
    
    def save_logs_and_checkpoints(self, args, global_iteration, step_bar, logs_dict={}, client_states={}):
        # TensorBoard
        self._save_logs(global_iteration, logs_dict)

        # save ckpt
        tag = f"global_iteration {global_iteration}"
        self._save_checkpoint(args, tag, client_states)

    def _save_logs(self, step, tag='train', logs_dict={}):
        if self._tensorboard is not None and self.strategy.is_rank_0():
            for k, v in logs_dict.items():
                self._tensorboard.add_scalar(f"{tag}/{k}", v, step)

    def _save_checkpoint(self, args, tag, client_states={}):
        self.strategy.save_ckpt(
            self.generative_model.model,
            os.path.join(args.save_path, "checkpoints"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )

    def _load_checkpoint(self, args, tag):
        self.strategy.load_ckpt(
            self.generative_model.model,
            os.path.join(args.save_path, "checkpoints"),
            tag,
        )
    
    def evaluate(
        self, 
        eval_dataset: BaseDataset, 
        eval_dataloader: DataLoader, 
        steps=0, 
        dataset_name="gsm8k",
    ):
        self.generative_model.eval()
        dataset_postprocess = eval_dataset.dataset_postprocess
        data_postprocess = eval_dataset.data_postprocess
        with torch.no_grad():
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            predictions, references = [], []
            for (questions, prompts, _, gts) in eval_dataloader:
                prompt_lens = [len(prompt) for prompt in prompts]
                # tokenizer
                batch = self.tokenizer(prompts, return_tensors="pt", add_special_tokens=False, max_length=self.prompt_max_len, padding=True, truncation=True)
                inputs = {k: v.to('cuda') for k, v in batch.items()}
                # generate
                eval_generate_kwargs = deepcopy(self.generate_kwargs)
                eval_generate_kwargs["temperature"] = 0.0
                eval_generate_kwargs["do_sample"] = False
                outputs = self.generative_model.generate(**inputs, tokenizer=self.tokenizer, **eval_generate_kwargs)
                # detokenizer
                generations = self.tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)
                generations = [generattion[len_:] for generattion, len_ in zip(generations, prompt_lens)]
                preds = [data_postprocess(generation) if data_postprocess else generation for generation in generations]
                refs = [dataset_postprocess(gt) if dataset_postprocess else gt for gt in gts]
                predictions.extend(preds)
                references.extend(refs)
                step_bar.update()

            # Accuracy
            torch.distributed.barrier()
            # Gather predictions and references from all processes
            global_predictions = [None] * self.strategy.world_size
            global_references = [None] * self.strategy.world_size
            torch.distributed.all_gather_object(global_predictions, predictions)
            torch.distributed.all_gather_object(global_references, references)
            # Flatten the lists
            global_predictions = [item for sublist in global_predictions for item in sublist]
            global_references = [item for sublist in global_references for item in sublist]
            if self.strategy.is_rank_0():
                result = eval_dataset.evaluator.score(predictions=global_predictions, references=global_references)
                accuracy = {"eval accuracy": result['accuracy']}
                # step_bar.set_postfix(logs)
            # Log to tensorboard
            if self.strategy.is_rank_0():
                print("tensorboard:", self._tensorboard)
                print("accuracy:", accuracy)
                if self._tensorboard is not None:
                    for k, v in accuracy.items():
                        self._tensorboard.add_scalar(f"eval_{dataset_name}/{k}", v, steps)
            torch.distributed.barrier()
        self.generative_model.train()  # reset model state