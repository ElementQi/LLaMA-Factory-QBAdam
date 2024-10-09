import warnings
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch.optim import Optimizer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from .utils import dequantize_model_blocks, quantize_back_model_block


class QBlockOptimizer(Optimizer):
    """Wrap the original optimizer to update trainable parameters periodically based on a specified block list."""

    def __init__(
        self,
        base_optimizer: Optimizer,
        named_parameters_list,
        block_prefix_list: List[str],
        switch_block_every: int = 10,
        start_block: Optional[int] = None,
        switch_mode: str = "ascending",
        active_modules: List[str] = [],
        include_embedding=False,
        include_lm_head=False,
        log_fn=None,
        model=None,
        # device="cuda:1",
        verbose=1,
    ):
        """
        Args:
            base_optimizer (Optimizer): The base optimizer being wrapped by the BlockOptimizer.
            named_parameters_list: A function that generates the named parameters of the model.
            block_prefix_list (List[List[str]]): The list of blocks of parameters to be updated.
            switch_block_every (int, optional): The number of optimization steps before switching to the next block. Defaults to 10.
            start_block (Optional[int], optional): The index of the block to start with. Defaults to None.
            switch_mode (str, optional): The mode for switching between different blocks of parameters. Defaults to "descending".
            active_modules (List[str]): The list of modules that are always active during optimization. Defaults to None.

            log_fn: A logging function for recording information during optimization. Defaults to None.
        """
        if block_prefix_list is None:
            block_prefix_list = self.infer_param_groups(
                [n for n, _ in named_parameters_list], include_embedding, include_lm_head
            )

        assert switch_mode in ["random", "descending", "ascending", "fixed"]
        assert isinstance(block_prefix_list, list)

        (_, param_) = named_parameters_list[0]

        self.device = param_.device
        self.dequantized_names = []
        self.model = model
        self.switch_mode = switch_mode
        self.switch_block_every = switch_block_every
        self.named_parameters_list = named_parameters_list
        self.weight_decay = base_optimizer.param_groups[0]["weight_decay"]
        self.block_prefix_list = block_prefix_list
        self.block_num = len(block_prefix_list)
        self.log_fn = log_fn
        self.global_step = 0
        self.base_optimizer = base_optimizer
        self.active_modules = active_modules
        self.defaults = base_optimizer.defaults
        self.dequantized = False
        self.verbose = verbose

        self.param_groups = base_optimizer.param_groups
        self.state_dict = base_optimizer.state_dict  # for compatibility of hf Trainer

        self.save_flag = False
        self.block_count = 0

        # testing only
        # import copy
        # self.ref_model = copy.deepcopy(self.model)

        if start_block is not None:
            self.current_block_idx = start_block
        elif switch_mode == "descending":
            self.current_block_idx = self.block_num - 1
        elif switch_mode == "ascending":
            self.current_block_idx = 0
        elif self.switch_mode == "random":
            self.block_order = torch.randperm(self.block_num).tolist()
            print("next block epoch's update order:", self.block_order[::-1])
            self.current_block_idx = self.block_order.pop()

        # detect if in lora mode or not
        self.lora_mode = False
        if any("lora" in n for n, _ in named_parameters_list):
            self.lora_mode = True
            print("LoRA mode detected. Will only train the lora parameters.")

        if any(isinstance(p, torch.FloatTensor) for _, p in named_parameters_list):
            warnings.warn(
                "BAdam expect model to be loaded in fp16 precision while detect fp32 weight. \
                This will cause additional memory usage and lose the benefit of mixed precision training."
            )

        super().__init__(self.param_groups, base_optimizer.defaults)

        self.update_trainable_params(self.verbose)

    @property
    def embedding_layer(self):
        for n, p in self.named_parameters_list:
            if "embed" in n:
                return p

    @property
    def lm_head_layer(self):
        for n, p in self.named_parameters_list:
            if "lm_head" in n:
                return p

    def infer_param_groups(self, param_names, include_embedding, include_lm_head):
        """automatic inference of the parameter groups based on the parameter names.
        divide groups into:
            * embedding
            * transformer layers
            * lm_head and others
        """
        import re

        block_prefix_list = []
        lm_head_and_other_params = []
        embed_pattern = r".*embed[^.]*\."
        layer_pattern = r".*layers.[^.]*\."

        for name in param_names:
            if any(prefix[0] in name for prefix in block_prefix_list):
                continue

            if re.findall(layer_pattern, name):
                block_prefix_list.append(re.findall(layer_pattern, name))
            elif re.findall(embed_pattern, name) and include_embedding:
                block_prefix_list.append(re.findall(embed_pattern, name))
            else:
                lm_head_and_other_params.append(name)

        if include_lm_head:
            block_prefix_list.append(lm_head_and_other_params)

        return block_prefix_list

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        return self.base_optimizer.load_state_dict(state_dict)

    def _update_lr(self):
        # Make sure the learning rate of the base_optimizer is consistent with the BlockOptimizer
        for group in self.base_optimizer.param_groups:
            group["lr"] = self.param_groups[0]["lr"]

    def step(self, *args, **kwargs) -> None:
        self._update_lr()

        self._grad_to_hp()
        self.base_optimizer.step(*args, **kwargs)
        self._update_param()
        self._clean_hp_grad()

        self.global_step += 1
        torch.cuda.empty_cache()
        # if self.global_step >= 5:
        #     breakpoint()
        if (self.global_step + 1) % self.switch_block_every == 0:
            # print(list(self.model.named_parameters()))
            self.block_count += 1
            self.update_trainable_params(self.verbose)

    def _clean_hp_grad(self) -> None:
        """Clean the gradients of the high precision parameters."""
        for hp_param in self.param_idx2hp.values():
            hp_param.grad = None

    def _update_param(self) -> None:
        """Update the low precision parameters with the values of the high precision parameters."""
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            lp_param.data.copy_(hp_param.to(lp_param.dtype).data)

    def _grad_to_hp(self, clear_lp_grads: bool = True) -> None:
        """
        Convert the gradients of the low precision parameters to high precision and calculate the gradient norm.

        Args:
            clear_lp_grads (bool, optional): Whether to clear the gradients of the low precision parameters. Defaults to True.
        """
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            assert lp_param.grad is not None, "The low precision parameter's gradient is None."
            hp_param.grad = lp_param.grad.float()

            if clear_lp_grads:
                lp_param.grad = None

    def update_trainable_params(self, verbose: Optional[int] = None) -> None:
        """
        Update the trainable parameters based on the current block index and the specified verbosity level.
        """
        self.active_param_prefixs = self.block_prefix_list[self.current_block_idx] + self.active_modules

        self.save_flag = False
        # save rule
        # save_step_every corresponds to the optimized block number
        save_step_every = 500
        K = self.switch_block_every
        batch_size = 16
        optimized_block_every = save_step_every // K
        layer_num = 32
        # if self.block_count >= 1 and self.block_count % optimized_block_every == 0:

        # badam-epoch save
        if self.block_count >= 1 and self.block_count % layer_num == 0:
            self.save_flag = True

            save_path_prefix = "/home/ubuntu/date/mq_tst/inner_delta_test/llamafactory"
            # save_path = (
            #     f"{save_path_prefix}/gsm8k_inner_K50_gc16_8bit_test/block_{self.block_count}_step_{self.global_step}"
            # )
            save_path = (
                f"{save_path_prefix}/gsm8k_inner_K50_gc16_4bit_test/block_{self.block_count}_step_{self.global_step}"
            )

            # If save, quantize first, then save
            # if self.save_flag and self.global_step >= 1:

            # next epoch's logic
            temp_current_block_idx = (self.current_block_idx - 1) % self.block_num
            print(f"quantize block:{temp_current_block_idx} \nsaving now\n")
            # quantize back the former
            self.named_parameters_list = quantize_back_model_block(
                self.model, temp_current_block_idx, self.dequantized_names, self.device
            )

            self.model.save_pretrained(save_path)
            print("Model saved at:", save_path)

        # First Dequantize
        self.dequantized_names, self.named_parameters_list = dequantize_model_blocks(
            self.model, self.current_block_idx, self.device
        )

        # Make sure there are trainable parameters in the current block when using lora
        while self.lora_mode:
            active_param_names = [
                n for n, _ in self.named_parameters_list if any(p in n for p in self.active_param_prefixs)
            ]
            if all("lora" not in n for n in active_param_names):
                print(
                    f"In LoRA mode but no lora parameters in the current block with prefix: {self.active_param_prefixs}. Switching to the next block."
                )
                self._update_active_block()
                self.active_param_prefixs = self.block_prefix_list[self.current_block_idx] + self.active_modules
                continue
            break

        if verbose >= 1:
            print("Parameters with the following prefix will be trainable:", self.active_param_prefixs)
            # print("model:\n", list(self.model.named_parameters()))

        # Reset parameters to be optimized
        self.param_idx2lp = {}
        self.param_idx2hp = {}

        active_param_groups = [
            {"params": [], "weight_decay": self.param_groups[0]["weight_decay"], **self.defaults},
            {"params": [], "weight_decay": 0.0, **self.defaults},
        ]

        # loop all params
        for i, (name, param) in enumerate(self.named_parameters_list):
            # if "mlp" in name:
            #     print("Here! MLP", name, param.dtype)

            if not any(p in name for p in self.active_param_prefixs):
                # print("No grad:", name, param.dtype)
                param.requires_grad_(False)
                param.grad = None
            else:
                # if self.global_step >= 3:
                #     breakpoint()
                if self.lora_mode and "lora" not in name:
                    continue
                if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                    # if self.verbose==1:
                    #     print("TheName:", name, "dtype", param.dtype)

                    # if "mlp" in name:
                    #     print(name)

                    # print("Require grads:", name, "dtype:", param.dtype)

                    param.requires_grad_(True)
                    param_hp = param.clone().float().detach().to(param.device)
                    param_hp.requires_grad = True

                    self.param_idx2lp[i] = param
                    self.param_idx2hp[i] = param_hp

                    if "bias" not in name and not isinstance(param, tuple(ALL_LAYERNORM_LAYERS)):
                        active_param_groups[0]["params"].append(param_hp)
                    else:
                        active_param_groups[1]["params"].append(param_hp)

                # else:
                #     print("Require grads but dtype is not fit in bf/f16, f32:", name, "dtype", param.dtype)

        self.base_optimizer.param_groups = active_param_groups

        import gc

        gc.collect()
        # Clean the optimizer state
        self.base_optimizer.state = defaultdict(lambda: {})
        # change current_block_idx

        if self.global_step >= 1 and not self.save_flag:
            # next epoch's logic
            temp_current_block_idx = (self.current_block_idx - 1) % self.block_num
            # quantize back the former
            self.named_parameters_list = quantize_back_model_block(
                self.model, temp_current_block_idx, self.dequantized_names, self.device
            )
        self._update_active_block()

    def _update_active_block(self):
        # Update the trainable block
        if self.switch_mode == "random":
            # self.current_block_idx = random.randint(0, self.block_num - 1)
            if len(self.block_order) == 0:
                self.block_order = torch.randperm(self.block_num).tolist()
                print("Next block epoch's update order:", self.block_order[::-1])
            self.current_block_idx = self.block_order.pop()
        elif self.switch_mode == "ascending":
            self.current_block_idx = (self.current_block_idx + 1) % self.block_num
        elif self.switch_mode == "descending":
            self.current_block_idx = (self.current_block_idx - 1) % self.block_num
        elif self.switch_mode == "fixed":
            pass

        # release dequantized_names, to make sure
        # dequantize when step into next block
        # print("before release, dequantized_names:")
        # print(self.dequantized_names)
        # self.dequantized_names = []
