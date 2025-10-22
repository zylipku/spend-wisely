from dataclasses import dataclass
from openrlhf.configs import BaseConfig

@dataclass
class VerifyConfig(BaseConfig):
    # 验证相关配置
    min_verified_samples: int = 64  # 开始训练所需的最小验证通过样本数
    save_verified_samples: bool = True  # 是否保存验证通过的样本
    
    # 生成相关配置
    max_length: int = 512
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    
    # 训练相关配置
    batch_size: int = 8
    learning_rate: float = 1e-5
    max_steps: int = 1000
    save_steps: int = 100
    eval_steps: int = 100