"""Preserve the trained edge prior while introducing semantic channels and LoRA."""


def upgrade_conditioning(model):
    import torch
    from torch import nn
    old = model.controlnet_cond_embedding.conv_in
    if old.in_channels != 3:
        raise ValueError('Expected a three-channel parent ControlNet')
    new = nn.Conv2d(7, old.out_channels, old.kernel_size, old.stride, old.padding,
                    bias=old.bias is not None, device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        # RGB parent received repeated grayscale boundaries: sum the three kernels.
        new.weight[:, 0] = old.weight.sum(dim=1)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.controlnet_cond_embedding.conv_in = new
    model.register_to_config(conditioning_channels=7)
    return model


def add_facade_lora(unet, rank=8):
    import torch
    from peft import LoraConfig
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights='gaussian',
                               target_modules=['to_q', 'to_k', 'to_v', 'to_out.0']))
    for parameter in unet.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)
    return [parameter for parameter in unet.parameters() if parameter.requires_grad]


def save_lora(unet, directory):
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import get_peft_model_state_dict
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(str(directory), unet_lora_layers=state,
        safe_serialization=True, weight_name='facade_lora.safetensors')


def restore_lora(unet, directory):
    from pathlib import Path
    from safetensors.torch import load_file
    from diffusers.utils import convert_unet_state_dict_to_peft
    from peft import set_peft_model_state_dict, get_peft_model_state_dict
    state = load_file(str(Path(directory) / 'facade_lora.safetensors'))
    # save_lora prefixes keys with unet for Diffusers pipeline compatibility.
    state = {key.removeprefix('unet.'): value for key, value in state.items()}
    state = convert_unet_state_dict_to_peft(state)
    if set(state) != set(get_peft_model_state_dict(unet)):
        raise ValueError('Missing or unexpected adapter keys in checkpoint')
    result = set_peft_model_state_dict(unet, state)
    if result.unexpected_keys:
        raise ValueError('Unexpected adapter keys in checkpoint')
