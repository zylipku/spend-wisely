conda create -n Iterative
conda activate Iterative
conda install python=3.11 -y
conda install -c conda-forge gcc=11.2 gxx=11.2 -y
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install flash_attn-2.6.1+cu118torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# git clone git@github-LiAlH4:OpenRLHF/OpenRLHF.git
cd OpenRLHF/
git reset --hard v0.5.0
pip install -e .
pip uninstall deepspeed -y
cd ..
# git clone git@github-LiAlH4:microsoft/DeepSpeed.git
cd DeepSpeed/
git reset --hard v0.15.0
DS_BUILD_UTILS=1 DS_BUILD_FUSED_ADAM=1 pip install .
ds_report