from typing import Iterable, Union

import bitsandbytes.functional as F
import torch
import torch.nn as nn
from bitsandbytes.nn import Linear4bit, Params4bit
from peft.utils.integrations import dequantize_bnb_weight


def dequantize_model_blocks(model, current_block_idx, device="cuda:0"):
    # dequant identifier
    names = []

    for name_outer, child_outer in model.model.layers[current_block_idx].named_children():
        for name, child in child_outer.named_children():
            if isinstance(child, Linear4bit):
                # child is the module
                # dequantize_module_weight(child)

                quantized_weight = child.weight
                # dequantized_weight = dequantize_bnb_weight(quantized_weight, state=quantized_weight.quant_state)

                dequantized_weight = F.dequantize_4bit(quantized_weight.data, quantized_weight.quant_state)

                new_linear = nn.Linear(
                    in_features=child.in_features, out_features=child.out_features, bias=(child.bias is not None)
                ).to(device)

                new_linear.weight.data = dequantized_weight.data
                new_linear.weight.requires_grad = True
                if child.bias is not None:
                    new_linear.bias.data = child.bias.data
                    new_linear.bias.requires_grad = True

                # # change Linear layer
                setattr(getattr(model.model.layers[current_block_idx], name_outer), name, new_linear)
                del child  # Release the old quantized layer

                if name not in names:
                    names.append(name)

    # update named_parameters_list
    named_parameters_list = list(model.named_parameters())

    return names, named_parameters_list


def quantize_back_model_block(model, current_block_idx, names, device="cuda:0"):
    for name_outer, child_outer in model.model.layers[current_block_idx].named_children():
        for name, child in child_outer.named_children():
            if name in names:
                quantized_layer = Linear4bit(
                    input_features=child.in_features, output_features=child.out_features, bias=(child.bias is not None)
                ).to(device)

                tensor_data, state = F.quantize_4bit(child.weight.data)
                quantized_layer.weight = Params4bit(data=tensor_data, quant_state=state)
                quantized_layer.weight.requires_grad = False

                if child.bias is not None:
                    quantized_layer.bias.data = child.bias.data
                    quantized_layer.bias.requires_grad = False

                setattr(getattr(model.model.layers[current_block_idx], name_outer), name, quantized_layer)
                del child  # Release the old dequantized layer

    # update named_parameters_list
    named_parameters_list = list(model.named_parameters())

    return named_parameters_list


# For torch>=2.1, `_foreach_norm` is used when implementing `clip_grad_norm_`, which doesn't support sparse tensor yet.
# We can temporarily fix this issue by using the older torch version's implementation:
# self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)
def clip_grad_norm_for_sparse_tensor(self, parameters, max_norm, norm_type=2):
    """
    Modification of the accelerator.clip_grad_norm_ to enable gradient clipping for sparse tensor.
    Used for torch version >= 2.1
    """
    from accelerate.utils import DistributedType
    from torch import inf

    if self.distributed_type == DistributedType.FSDP:
        self.unscale_gradients()
        parameters = list(parameters)
        for model in self._models:
            if parameters == list(model.parameters()):
                return model.clip_grad_norm_(max_norm, norm_type)
    elif self.distributed_type == DistributedType.DEEPSPEED:
        # `accelerator.backward(loss)` is doing that automatically. Therefore, its implementation is not needed
        # We cannot return the gradient norm because DeepSpeed does it.
        return None
    self.unscale_gradients()

    def clip_func_(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
    ) -> torch.Tensor:
        r"""torch 1.13 version clip_grad_norm_, works well with sparse tensor.
        Clips gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of the gradients from :attr:`parameters` is ``nan``,
                ``inf``, or ``-inf``. Default: False (will switch to True in the future)

        Returns:
            Total norm of the parameter gradients (viewed as a single vector).
        """

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grads = [p.grad for p in parameters if p.grad is not None]
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if len(grads) == 0:
            return torch.tensor(0.0)
        device = grads[0].device
        if norm_type == inf:
            norms = [g.detach().abs().max().to(device) for g in grads]
            total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
        else:
            total_norm = torch.norm(
                torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type
            )
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f"The total norm of order {norm_type} for gradients from "
                "`parameters` is non-finite, so it cannot be clipped. To disable "
                "this error and scale the gradients by the non-finite norm anyway, "
                "set `error_if_nonfinite=False`"
            )
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
        return total_norm

    return clip_func_(parameters, max_norm, norm_type=norm_type)
