### how to run in kaggle notebook environment 

%%bash
```
%%bash
set -e
cd /kaggle/working
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /kaggle/miniconda
source /kaggle/miniconda/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -y -n py310 python=3.10
rm -rf Miniconda3-latest-Linux-x86_64.sh
```

```
!rm -rf sheeprl
!/kaggle/miniconda/envs/py310/bin/python --version
!git clone -b my-exp https://github.com/jfojfo/sheeprl
!/kaggle/miniconda/envs/py310/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
!cd sheeprl && /kaggle/miniconda/envs/py310/bin/pip install . && /kaggle/miniconda/envs/py310/bin/pip install .[atari]
!rm -rf sheeprl/.git
```

```
!cd sheeprl && MPLBACKEND=TkAgg /kaggle/miniconda/envs/py310/bin/python sheeprl.py exp=dreamer_ppo env=atari env.id=PongNoFrameskip-v4 env.num_envs=4 fabric.accelerator=gpu checkpoint.every=20000 metric.log_every=2000 buffer.checkpoint=False exp_name='${algo.name}_${env.id}_seperate_opti'
```