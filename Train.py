from transformers import AutoTokenizer, AutoModelForCausalLM
from Trainer import DPOTrainer
from Model_Quant import get_quant_model
from omegaconf import OmegaConf, DictConfig
import wandb
import os
import torch

def main(config: DictConfig):

    model_name = config.model_name
    cache_dir = config.cache_dir
    output_dir = config.output_dir
    quantization_config = config.quantization_config

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    #set for wandb
    if config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = config.wandb.wandb_dir
        run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=config.wandb.wandb_dir,
            name=config.dpo_exp_name,
            reinit=True,
        )
        

    print(f"loading tokenizer {model_name} from cache {cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True, cache_dir = cache_dir)

    print(f"loading quant model {model_name} from cache {cache_dir}")
    ###set quantization model
    model = get_quant_model(model_name, cache_dir, quantization_config)
    ###set reference model
    model_ref = get_quant_model(model_name, cache_dir, quantization_config)

    dpo_trainer = DPOTrainer(model, config, tokenizer, model_ref)
    
    dpo_trainer.train(sft_mode=True)
    dpo_trainer.save_model(output_dir=output_dir)
    
    run.finish()
    
    #reload model for dpo mode
    dpo_trainer.reload_ref_model()

    dpo_trainer.train(sft_mode=False)
    dpo_trainer.save_model(output_dir=output_dir, sft_mode=False)


if __name__ == '__main__':
    config = OmegaConf.load("config.yaml")
    main(config)