# DPO_Quant

Learning from human preferences is a paradigm adopted in the natural language processing literature to better align LLM to human desiderata. Recently [RLHF](https://arxiv.org/abs/2203.02155) has been used successfully in many senses to get a better performance. In 2023 NeurIPS, [DPO](https://arxiv.org/abs/2305.18290)  was proposed for addressing the problem of huge resource consumption in training. However, for people who don't have enough GPUs, training a model with DPO is still a difficult situation. In this reposity, I implemented a code reproduction of the DPO algorithm and the [BitsandBytes](https://github.com/TimDettmers/bitsandbytes) is used for the model quantization to make run of DPO on a 24G 4090 possible.



## Getting Started

- System requirement: Ubuntu20.04/Windows 11, Cuda 12.1
- Tested GPUs: RTX4090

### Create conda environment:

```
 conda create -n dpo python=3.10
 conda activate dpo
```

### Install packages with `pip`

```
 pip install -r requirements.txt
```

### Switch Source

For some reason, user may not be able to access HuggingFace conveniently. Run the code bellow to handle this.

```
export HF_ENDPOINT="https://hf-mirror.com"
```



## Training

Model pythia2.8B and dataset Anthropic/hh-rlhf is used in the training. For customized training, you need change  `dataset_name` and `model_name` in the file config.yaml.

Run the code bellow to start a training.

```
python Train.py
```



## Experiment Analysis

I conducted the experiment with BitsandBytes to load the quantization model. The main pipline of DPO is (1)Training the model using SFT on a preference dataset and (2)Traing the model using DPO on the same dataset.

In SFT, the run lasts 2h20min, we can see from the figure that the eval_loss snowly decreases when the step grows.

![Figure1](.\output\W&B Chart 2024_4_19 17_25_30.png)

For more details, check [here](https://wandb.ai/qiyuwu/pythia2_8B_DPO_Quant/runs/co6guc8k?nw=nwuserwqy123202108) 

In DPO, the run lasts 7h30min, we can see from the figure that the accuracies and margins snowly increases when the step grows.

![Figure2](.\output\W&B Chart 2024_4_19 17_24_13.png)
![Figure3](.\output\W&B Chart 2024_4_19 17_25_00.png)

For more details, check [here](https://wandb.ai/qiyuwu/pythia2_8B_DPO_Quant/runs/0tejjuhj?nw=nwuserwqy123202108) 
