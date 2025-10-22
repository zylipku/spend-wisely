MODEL_PATH="/path/to/your/model/checkpoint"
DATASET_NAME=gsm8k_mix3_172
SETTING=iterative
LEARNING_RATE=1e-7
ITERATIVE_POLICY=exponential
MAX_STEPS=1000
INITIAL_SIZE=10
GROWTH_RATE=1.1
SCHEDULER=constant
TEMPERATURE=0.3
SAVE_PATH=./checkpoint/$SETTING/$(basename $MODEL_PATH)/$DATASET_NAME/policy=$ITERATIVE_POLICY/steps=$MAX_STEPS-initial=$INITIAL_SIZE-growth=$GROWTH_RATE/lr=$LEARNING_RATE-schedule=$SCHEDULER-temp=$TEMPERATURE

deepspeed --num_gpus=8 --module openrlhf.cli.train_select \
   --setting $SETTING \
   --dataset_names gsm8k gen_gsm_symbolic_7 gen_gsm_p1_2 \
   --eval_dataset_name gsm8k generated_gsm8k_symbolic generated_gsm8k_p1 \
   --input_template $"Question: {} Let's think step by step. \n Answer: " \
   --train_batch_size 256 \
   --micro_train_batch_size 4 \
   --rollout_batch_size 512 \
   --eval_batch_size 128 \
   --scheduler $SCHEDULER \
   --iterative_policy $ITERATIVE_POLICY \
   --max_steps $MAX_STEPS \
   --initial_size $INITIAL_SIZE \
   --growth_rate $GROWTH_RATE \
   --pretrain $MODEL_PATH \
   --save_path $SAVE_PATH \
   --eval_steps 80 \
   --save_steps 10 \
   --prompt_max_len 2048 \
   --generate_max_len 512 \
   --max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --learning_rate $LEARNING_RATE \
   --flash_attn \
   --temperature $TEMPERATURE \
   --use_tensorboard $SAVE_PATH/log \