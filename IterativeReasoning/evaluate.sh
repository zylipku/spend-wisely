MODEL_PATH="path_to_your_pretrained_model"
DATASET_NAME=gsm8k
SETTING=evaluate
LEARNING_RATE=5e-7
SAVE_PATH=./checkpoint/$SETTING/$(basename $MODEL_PATH)/$DATASET_NAME
SAVE_PATH=./checkpoint/$SETTING/test

deepspeed --include localhost:4,5,6,7 --module openrlhf.cli.train_select \
   --setting $SETTING \
   --eval_dataset_name $DATASET_NAME \
   --input_template $"Question: {}\nLet's think step by step\nAnswer: " \
   --train_batch_size 256 \
   --micro_train_batch_size 4 \
   --rollout_batch_size 512 \
   --eval_batch_size 128 \
   --pretrain $MODEL_PATH \
   --save_path $SAVE_PATH \
   --eval_steps -1 \
   --prompt_max_len 2048 \
   --generate_max_len 512 \
   --max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --learning_rate $LEARNING_RATE \
   --lora_rank 64 \
   --lora_alpha 64 \
   --flash_attn \
   --gradient_checkpointing \
   --temperature 0.0 \
   --use_tensorboard $SAVE_PATH/log \