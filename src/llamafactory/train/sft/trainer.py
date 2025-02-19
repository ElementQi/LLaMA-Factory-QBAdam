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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)

mean_abs = lambda x: torch.mean(torch.abs(x))
FULL_LAYER_MODE = True
FULL_SFT_MODE = False

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
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
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.quantized_L2_output = []
        self.original_L2_output = []
        self.original_lm_head = []

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

    def capture_original_L2_output(self, module, inputs, outputs):
        # self.original_L2_output.append(outputs[0])

        l2_norm = outputs[0].detach().norm().item()  # Convert to scalar immediately
        # self.original_L2_output.append(l2_norm)
        # self.original_L2_output.append(outputs[0].shape)
        self.original_L2_output.append(outputs[0])

    def capture_quantized_L2_output(self, module, inputs, outputs):
        self.quantized_L2_output.append(outputs[0])

    def capture_original_lm_head(self, module, inputs, outputs):
        self.original_lm_head.append(outputs[0])

    @override
    def compute_loss(self, model, inputs, return_outputs=False):
        layer_for_measurement = model.lm_head

        # hook_prev_out = layer_for_measurement.register_forward_hook(self.capture_original_L2_output)

        # FULL_SFT_MODE = True

        if FULL_SFT_MODE:
            hook_prev_out = layer_for_measurement.register_forward_hook(self.capture_original_lm_head)
        else:
            hook_prev_out = layer_for_measurement.register_forward_hook(self.capture_quantized_L2_output)


        # model.eval()  # Disable dropout/batchnorm
        # with torch.no_grad():  
        #     outputs = model(**inputs)

        outputs = model(**inputs)

        # QUANT_METHOD = "awq"
        # save_dir = f"saved_outputs_for_sft/{QUANT_METHOD}/"

        save_dir = f"saved_outputs_for_sft"

        # breakpoint()
        if FULL_SFT_MODE:
            # full sft only
            original_lm_head_tensor = self.original_lm_head[0].detach().cpu()  # Detach and move to CPU
            
            os.makedirs(save_dir, exist_ok=True)  # Create directory if needed
            lm_head_save_path = os.path.join(save_dir, "lm_head.pt")
            loss_sft_save_path = os.path.join(save_dir, "loss.pt")

            loss_sft = outputs.loss.detach().cpu()
            torch.save(original_lm_head_tensor, lm_head_save_path)
            torch.save(loss_sft, loss_sft_save_path)

            self.log({
                "INFO": "original sft files saved",
                "lm_head shape": original_lm_head_tensor.shape,
                "loss_sft": loss_sft
                })
        else:
            # for comparison only
            original_tensor = torch.load(f"{save_dir}/lm_head.pt")
            loss_sft = torch.load(f"{save_dir}/loss.pt")
            original_tensor = original_tensor.to("cuda")  # Move back to GPU if needed
            loss_sft = loss_sft.to("cuda")
            
            loss_sft_quant = outputs.loss
            loss_quant = torch.norm(original_tensor - self.quantized_L2_output[0], p=2)
            relative_error = mean_abs(original_tensor - self.quantized_L2_output[0]) / mean_abs(original_tensor).detach()

        # self.log({"loaded_tensor": loaded_tensor})
        hook_prev_out.remove()

        if not FULL_SFT_MODE:
            self.log(
                {
                    "loss_sft": loss_sft.detach().item(),
                    "loss_sft_quant": loss_sft_quant.detach().item(),
                    "loss_quant": loss_quant.detach().item(),
                    "relative_error": relative_error.detach().item()
                }
            )

        self.original_L2_output = []

        loss = loss_sft
        
        torch.cuda.empty_cache()
        raise ValueError("Stop here")

        if not torch.is_grad_enabled():
            return loss, outputs
        else:
            return loss
