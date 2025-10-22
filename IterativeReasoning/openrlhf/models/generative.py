from typing import Optional, Tuple, Union

import torch
from transformers import StoppingCriteriaList, StoppingCriteria, PreTrainedTokenizer

from .actor import Actor

from opencompass.models.huggingface import MultiTokenEOSCriteria


class Improved_MultiTokenEOSCriteria(StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
    ):
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence,
                                             add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # Compare the last len(self.sequence_id) tokens
        lookback_ids_batch = input_ids[:, -self.sequence_id_len:]
        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        all_done = True
        for i, done in enumerate(self.done_tracker):
            if not done:
                if self.sequence in lookback_tokens_batch[i]:
                    self.done_tracker[i] = True
                else:
                    all_done = False
            if self.done_tracker[i]:
                input_ids[i, -1] = self.pad_token_id

        return all_done


class GenerativeModel(Actor):
    """
    Generative model base class.

    Args:
        model (nn.Module): Generative Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        packing_samples=False,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrain_or_model,
            use_flash_attention_2=use_flash_attention_2,
            bf16=bf16,
            load_in_4bit=load_in_4bit,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            ds_config=ds_config,
            device_map=device_map,
            packing_samples=packing_samples,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer = None, **kwargs) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        generate_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": kwargs.get("early_stopping", True),
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")
        # Update stopping criteria
        stopping_criteria = kwargs.get("stopping_criteria", [])
        if stopping_criteria:
            if tokenizer.eos_token is not None:
                stopping_criteria = stopping_criteria + [
                    tokenizer.eos_token
                ]
            stopping_criteria = StoppingCriteriaList([
                *[
                    Improved_MultiTokenEOSCriteria(sequence, tokenizer,
                                          input_ids.shape[0])
                    for sequence in stopping_criteria
                ],
            ])
            generate_args['stopping_criteria'] = stopping_criteria


        # Call generate
        return self.model.generate(**generate_args)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Returns action log probs"""
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output