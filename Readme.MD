This work takes inspiration and performs experiments on pick and place setup from [this](https://arxiv.org/pdf/2010.14500) paper. 
The objective is to:
1. compare the performance of offline RL algorithm TD3+BC with CQL
2. compare the performance of CQL leveraging pretrained resnet vs CNN

# Outcomes

TD3+BC gives somewhat better returns than CQL. In addition, CQL pre-trained with Resnet18 learns faster
and performs better than CQL leveraging CNN on the pick and place task.

[TD3+BC vs CQL](img2.png)

[CQL_CNN vs CQL_ResNet18](img1.png)

# Videos

The video for TD3_BC experiment is in [this](data/td3_bc_cnn/videos_eval) folder.


# Installation

Roboverse installtion: https://github.com/montrealrobotics/roboverse

`pip -r requirements.txt`

# Running Experiments

## CQL

`python examples/cog_cnn.py --env=Widow250PickTray-v0 --max-path-length=40 --alg_name=CQL
--prior-buffer='/path/to/file/pickplace_prior.npy' --task-buffer=''/path/to/file/pickplace_task.npy`

## CQL_Resnet

`python examples/cog_resnet.py --env=Widow250PickTray-v0 --max-path-length=40 --alg_name=CQL
--prior-buffer='/path/to/file/pickplace_prior.npy' --task-buffer=''/path/to/file/pickplace_task.npy`

## TD3_BC

`python examples/cog_cnn.py --env=Widow250PickTray-v0 --max-path-length=40 --alg_name=TD3_BC
--prior-buffer='/path/to/file/pickplace_prior.npy' --task-buffer=''/path/to/file/pickplace_task.npy`

# Acknowledgement

 This repository is built on top of this codebase: https://github.com/avisingh599/cog

