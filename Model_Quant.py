import torch.nn as nn
from transformers import BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM
import bitsandbytes as bnb
from omegaconf import DictConfig
import torch

def get_quant_model(model_name: str, cache_dir: str, quant_config: DictConfig) -> nn.Module:

    '''Get a quantized model with the specified configuration.'''
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                quantization_config=BitsAndBytesConfig(load_in_4bit=quant_config.load_in_4bit,
                                                                                        bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
                                                                                        bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
                                                                                        bnb_4bit_compute_dtype=getattr(torch, quant_config.bnb_4bit_compute_dtype),
                                                                                        ),
                                                device_map = 'auto', cache_dir = cache_dir, use_cache=False)
    model = prepare_model_for_kbit_training(model)
    def find_all_linear_names(model):
 
        cls = bnb.nn.Linear4bit 
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])


        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)
    modules = find_all_linear_names(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    
    return model