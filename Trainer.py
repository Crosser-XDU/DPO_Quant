import torch
from typing import Tuple, Optional, Dict
import torch.nn.functional as F
import torch.nn as nn
from Dataset import get_batch_iter
import random
import numpy as np
from collections import defaultdict
import tqdm
from typing import List, Union
from omegaconf import DictConfig, OmegaConf
from utils import formatted_dict, pad_to_length, get_micro_batch, move_to_device
import os
import wandb
import time

def DPO_Loss(policy_chosen_logps: torch.Tensor,
             policy_reject_logps: torch.Tensor,
             refer_chosen_logps: torch.Tensor,
             refer_reject_logps: torch.Tensor,
             beta: float = 0.1,
             reference_free: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Compute the DPO loss for the given log probabilities and beta value.'''
    # Compute the policy logpsratio
    policy_logps_ratio = policy_chosen_logps - policy_reject_logps
    refer_logps_ratio = refer_chosen_logps - refer_reject_logps

    # Compute the DPO loss
    if reference_free:
        refer_logps_ratio = 0

    logits = policy_logps_ratio - refer_logps_ratio

    losses = -F.logsigmoid(beta * logits) #(Batch_size, )

    chosen_reward = beta * (policy_chosen_logps - refer_chosen_logps).detach()
    reject_reward = beta * (policy_reject_logps - refer_reject_logps).detach()

    return losses, chosen_reward, reject_reward

def concat_chosen_rejected(batch: Dict[str, Union[List, torch.LongTensor]]):
    '''concatenate the chosen and rejected inputs to the same length and return the cocatenated batch'''

    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}

    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated').replace('rejected', 'concatenated')
            if 'chosen' in k:
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value)
            elif 'rejected' in k:
                concatenated_batch[concatenated_key] = torch.cat((
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value),
                ), dim=0)

    return concatenated_batch

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False):
    '''compute the log probabilities of the given logits for the given labels'''
    assert logits.shape[:-1] == labels.shape, "logits and labels must have the same shape, except for the last dimension."

    labels_clone = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels_clone != -100)

    labels_clone[labels_clone == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels_clone.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)

class DPOTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, tokenizer: nn.Module, reference_model: Optional[nn.Module] = None, seed : int = 0) -> None:
        "Trainer for DPO; Train the polcy model using DPO or SFT"

        self.policy = policy
        self.reference_model = reference_model
        self.seed = seed
        self.config = config
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.run_dir = os.path.join(config.output_dir+"/", config.exp_name)
        
        self.dataset_iterator_config = dict(
            tokenizer = tokenizer,
            dataset_name = config.dataset_name,
            max_len = config.max_length,
            max_prompt_len = config.max_prompt_length,
            cache_dir = config.cache_dir,
            num_workers=self.config.num_workers,
        )

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu').index

    def reload_ref_model(self):
        '''reload the reference model from the saved state dict'''
        state_dict_path = self.config.output_dir + "/SFT_policy.pt"
        assert os.path.exists(state_dict_path), f"state_dict_path {state_dict_path} does not exist"
        state_dict = torch.load(state_dict_path, map_location='cpu')
        step = state_dict['step_idx']
        print(f'loading pre-trained weights at step {step} from {state_dict_path}')

        self.reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]):
        '''forward pass for the concatenated batch'''
        concatenated_batch = concat_chosen_rejected(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True, sft_mode: bool = True):
        """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'

        if sft_mode:
            
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            losses = -policy_chosen_logps
        
        else:
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free}

            losses, chosen_rewards, rejected_rewards = DPO_Loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            
            policy_rejected_logps = policy_rejected_logps.detach()
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        policy_chosen_logps = policy_chosen_logps.detach()
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_losses = losses.detach()
        metrics[f'loss/{train_test}'] = all_losses.cpu().numpy().tolist()

        return losses.mean(), metrics
    
    def train(self, sft_mode = True):
        "Train the policy model using SFT"

        sft_dpo = "SFT" if sft_mode else "DPO"

        if not sft_mode:
            #reset for wandb
            if self.config.wandb.enabled:
                os.environ['WANDB_CACHE_DIR'] = self.config.wandb.wandb_dir
                wandb.init(
                    entity=self.config.wandb.entity,
                    project=self.config.wandb.project,
                    config=OmegaConf.to_container(self.config),
                    dir=self.config.wandb.wandb_dir,
                    name=self.config.dpo_exp_name,
                )

        #prepare the dataset
        print(f"Beging {sft_dpo} training")

        print(f'Load train data iterator with batch size {self.config.batch_size} and {self.config.n_examples} examples')
        self.train_iterator = get_batch_iter(**self.dataset_iterator_config, train_test='train', sft_mode=sft_mode, n_examples=self.config.n_examples, batch_size=self.config.batch_size)
        
        print(f'Load eval batches with batch size {self.config.eval_batch_size}')
        self.eval_iterator = get_batch_iter(**self.dataset_iterator_config, train_test='test', sft_mode=sft_mode, n_examples=self.config.n_eval_examples, batch_size=self.config.eval_batch_size)
        self.eval_batches = list(self.eval_iterator)

        #initialize the optimizer and scheduler
        print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1))) #warmup_steps
    
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        if not sft_mode:
            self.reference_model.eval()

        for batch in self.train_iterator:
            
            #train
            self.policy.train()
            start_time = time.time()
            batch_metrics = defaultdict(list)

            microbatch_size = len(list(batch.values())[0]) // self.config.gradient_accumulation_steps

            for microbatch_idx in range(self.config.gradient_accumulation_steps):

                microbatch = get_micro_batch(batch, microbatch_idx, microbatch_size, self.device)
                microbatch = move_to_device(microbatch, self.device)
                loss, metrics = self.get_batch_metrics(microbatch, self.config.loss, train=True, sft_mode=sft_mode)
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            
            #evaluate
            if self.example_counter % self.config.eval_every == 0 and self.example_counter > 0:
                self.evaluate(sft_mode)
    
    def evaluate(self, sft_mode: bool = True):
        """Evaluate the policy model using the eval dataset."""
        print(f'Running evaluation after {self.example_counter} train examples')
        self.policy.eval()

        all_eval_metrics = defaultdict(list)

        for eval_batch in tqdm.tqdm(self.eval_batches, desc='Computing eval metrics'):
            local_eval_batch = move_to_device(eval_batch, self.device)
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False, sft_mode=sft_mode)

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v) #with torch.no_grad
        
        mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
        print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')

        if self.config.wandb.enabled:
            wandb.log(mean_eval_metrics, step=self.example_counter)
            
        #save model
        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
        if self.example_counter % (self.config.save_ratio * self.config.n_examples) == 0:
            print(f'creating checkpoint to write to {output_dir}...')
            self.save_model(output_dir, mean_eval_metrics)

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None, sft_mode: bool = True):
        """Write a checkpoint to disk."""
        if not sft_mode:
            filename = "DPO_" + filename
        else:
            filename = "SFT_" + filename
            
        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save_model(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None, sft_mode: bool = True):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir, sft_mode)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir, sft_mode)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir, sft_mode)