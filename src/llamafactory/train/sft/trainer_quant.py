# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
from collections import defaultdict
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .quant_utils.quantization import dequantize_layer, quantize_layer


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)

mean_abs = lambda x: torch.mean(torch.abs(x))

class QuantizedBCDTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.original_L2_output = []
        self.quantized_L2_output = []
        self.active_layer_idx = 1
        self.prev_active_layer_idx = 0
        self.quantization_type = finetuning_args.qbcd_quant_type
        self.switch_freq = finetuning_args.badam_switch_interval
        self.quant_alpha = 0.0

        assert (finetuning_args.use_qabcd != finetuning_args.use_qbcd) or (not finetuning_args.use_qabcd and not finetuning_args.use_qbcd), "QABCD and QBCD cannot be used together unless both are False"
        
        if finetuning_args.use_qabcd:
            assert finetuning_args.qabcd_quant_alpha != 0.0, "QABCD must set alpha != 0"
            self.quant_alpha = finetuning_args.qabcd_quant_alpha

        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def capture_original_L2_output(self, module, inputs, outputs):
        self.original_L2_output.append(outputs[0])

    def capture_quantized_L2_output(self, module, inputs, outputs):
        self.quantized_L2_output.append(outputs[0])

    def switch_active_block(self, model):
        # quantize the previous active layer
        # TODO 8bit quantization
        if self.quantization_type in ["4bit"]:
            prev_active_layer = model.model.layers[self.prev_active_layer_idx]
            quantized_prev_active_layer = quantize_layer(copy.deepcopy(prev_active_layer), quantization_type="4bit").to("cuda")
            model.model.layers[self.prev_active_layer_idx] = quantized_prev_active_layer

        num_layers = len(model.model.layers)

        # change the active layer
        self.active_layer_idx = (self.active_layer_idx + 1) % num_layers
        self.prev_active_layer_idx = (self.prev_active_layer_idx + 1) % num_layers

        # dequantize the active layer
        active_layer = model.model.layers[self.active_layer_idx]
        for module in active_layer.modules():
            if isinstance(module, bnb.nn.Linear4bit) or isinstance(module, bnb.nn.Linear8bitLt):
                model.model.layers[self.active_layer_idx] = dequantize_layer(copy.deepcopy(active_layer)).to("cuda")
                break

        # re-initialize parameter groups
        active_param_groups = [
            {
                "params": [],
                "weight_decay": self.optimizer.param_groups[0]['weight_decay'],
                **self.optimizer.defaults
            },
            {
                "params": [],
                "weight_decay": 0.0,
                **self.optimizer.defaults
            },
        ]

        for p in model.parameters():
            p.requires_grad = False
            p.grad = None

        # print("Switch to new block with trainable parameters:")
        for n, p in model.model.layers[self.active_layer_idx].named_parameters():
            p.requires_grad = True
            # print(n, list(p.size()))

            if "bias" not in n and not isinstance(p, tuple(ALL_LAYERNORM_LAYERS)):
                active_param_groups[0]['params'].append(p)
            else:
                active_param_groups[1]['params'].append(p)
        self.optimizer.param_groups = active_param_groups
        self.optimizer.state = defaultdict(lambda: {})

        # print("switch to new layer with idx: ", self.active_layer_idx)
        print("num_layers: ", num_layers, "active_layer_idx: ", self.active_layer_idx, "prev_active_layer_idx: ", self.prev_active_layer_idx)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        if torch.is_grad_enabled() and (self.state.global_step + 1) % (self.switch_freq * self.args.gradient_accumulation_steps) == 0:
            self.switch_active_block(model)

        for n, p in model.named_parameters():
            p.requires_grad = False

        for n, p in model.model.layers[self.active_layer_idx].named_parameters():
            p.requires_grad = True

        # active_layer = model.model.layers[self.active_layer_idx]
        prev_active_layer = model.model.layers[self.prev_active_layer_idx]

        # layer_for_measurement = model.model.layers[-1]
        layer_for_measurement = model.lm_head

        hook_prev_out = layer_for_measurement.register_forward_hook(self.capture_original_L2_output)
        # loss_sft = model(**inputs).loss
        outputs = model(**inputs)
        loss_sft = outputs.loss
        hook_prev_out.remove()

        loss_quant = 0.

        # only compute the quantization loss when in training mode
        if torch.is_grad_enabled():
            quantized_prev_active_layer = quantize_layer(copy.deepcopy(prev_active_layer), quantization_type="4bit").to("cuda") # TODO: re-use it
            model.model.layers[self.prev_active_layer_idx] = quantized_prev_active_layer
            hook_quantized_out = layer_for_measurement.register_forward_hook(self.capture_quantized_L2_output)
            loss_sft_quant = model(**inputs)

            # loss_quant = torch.norm(self.original_L2_output[0] - self.quantized_L2_output[0], p=2) / (self.original_L2_output[0].size(0) * self.original_L2_output[0].size(1))
            loss_quant = torch.norm(self.original_L2_output[0] - self.quantized_L2_output[0], p=2)
            # loss_quant = self._cal_quantization_loss()
            relative_error = mean_abs(self.original_L2_output[0] - self.quantized_L2_output[0]) / mean_abs(self.original_L2_output[0]).detach()
            # print(f"loss_sft: {loss_sft}, loss_quant: {loss_quant}, relative_error: {relative_error}")

            self.log(
                {
                    "loss_sft": loss_sft.detach().item(),
                    "loss_sft_quant": loss_sft_quant.loss.detach().item(),
                    "loss_quant": loss_quant.detach().item(),
                    "relative_error": relative_error.detach().item()
                }
            )

            model.model.layers[self.prev_active_layer_idx] = prev_active_layer
            hook_quantized_out.remove()

        self.original_L2_output = []
        self.quantized_L2_output = []

        # alpha = 0.001 if step <= 50 else 0.
        loss = loss_sft + self.quant_alpha * loss_quant

        torch.cuda.empty_cache()
        if not torch.is_grad_enabled():
            return loss, outputs
        else:
            return loss

    def _cal_quantization_loss(self):
        criterion = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        origin_output = self.original_L2_output[0].float()
        quantized_output = self.quantized_L2_output[0].float()

        origin_output = F.log_softmax(origin_output.view(-1, origin_output.size(-1)), dim=-1)
        quantized_output = F.log_softmax(quantized_output.view(-1, quantized_output.size(-1)), dim=-1)
        loss = criterion(origin_output, quantized_output)

        return loss

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
