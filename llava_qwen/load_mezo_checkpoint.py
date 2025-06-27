import torch
import warnings


def get_trainable_params(model: torch.nn.Module, trainable_params_names: list = None) -> list:
    """
    Get the trainable parameters of the model in the order in the list of named parameters.
    """
    trainable_params = []

    params_dict = dict(model.named_parameters())

    if len(params_dict) != len(list(model.parameters())):
        raise ValueError("Mismatch between named parameters and model parameters.")

    for trainable_param_name in trainable_params_names:
        param = params_dict.get(trainable_param_name)
        if param is not None:
            trainable_params.append((trainable_param_name, param))

    print(f"Trainable parameters: {trainable_params_names}")
    return trainable_params


def adjust_noise_to_param(noise: torch.Tensor, param: torch.nn.Parameter, param_name: str) -> torch.Tensor:
    """
    Adjust the noise tensor (generated using the old shape) to match the current param.data size.
    If the noise tensor is larger, it gets truncated. If it's smaller, it gets padded with zeros.
    A warning is issued if adjustments are made.
    """
    current_shape = list(param.data.size())
    noise_shape = list(noise.size())

    if current_shape == noise_shape:
        return noise

    adjusted_noise = torch.zeros(current_shape, device=noise.device, dtype=noise.dtype)

    min_shape = [min(c, n) for c, n in zip(current_shape, noise_shape)]

    slice_tuple_new = tuple(slice(0, ms) for ms in min_shape)
    slice_tuple_noise = tuple(slice(0, ms) for ms in min_shape)

    adjusted_noise[slice_tuple_new] = noise[slice_tuple_noise]

    warnings.warn(
        f"Noise for parameter '{param_name}' adjusted from shape {noise_shape} to {current_shape}."
    )
    return adjusted_noise


def zo_perturb_parameters(trainable_params, seed, zo_eps, trainable_params_sizes, scaling_factor=1.0):
    """
    Perturb the model parameters with a random vector z.
    """
    torch.manual_seed(seed)

    for name, param in trainable_params:
        checkpoint_shape = trainable_params_sizes.get(name)
        if checkpoint_shape is None:  # Fallback if old shape not available.
            z = torch.normal(
                mean=0, std=1, size=param.data.size(),
                device=param.device, dtype=param.dtype
            )
        else:
            z_old = torch.normal(
                mean=0, std=1, size=tuple(checkpoint_shape),
                device=param.device, dtype=param.dtype
            )
            z = adjust_noise_to_param(z_old, param, name)

        param.data += scaling_factor * z * zo_eps


def apply_mezo_state_from_path(model: torch.nn.Module, checkpoint_path: str,
                               steps: int = None) -> torch.nn.Module:
    """
    Replay the MeZO update history on the model parameters using the original noise shape from
    the checkpoint, and then adjust it to the current parameter shape.
    """
    mezo_state = torch.load(checkpoint_path, map_location="cpu")
    print("Loaded MeZO checkpoint from:", checkpoint_path)

    weight_decay = mezo_state.get("weight_decay", 0.0)
    full_update_history = mezo_state.get("update_history", [])
    trainable_params_names = mezo_state.get("trainable_params_names", [])
    trainable_params_sizes = mezo_state.get("trainable_params_sizes", {})

    trainable_params = get_trainable_params(model, trainable_params_names)

    if steps is not None:
        update_history = []
        for update in full_update_history:
            global_step = update["global_step"]
            if global_step >= steps:
                break
            update_history.append(update)
        print(f"Applying only {steps} update steps.")
    else:
        update_history = full_update_history
        print(f"Applying all {len(update_history)} update steps.")

    for update in update_history:
        update_type = update["type"]
        global_step = update["global_step"]

        if update_type == "mezo_update":
            seed_group = update["seed_group"]
            learning_rate = update["learning_rate"]

            for seed, grad_sum in seed_group.items():
                torch.manual_seed(seed)

                for name, param in trainable_params:
                    checkpoint_shape = trainable_params_sizes.get(name)
                    if checkpoint_shape is None:  # Fallback if old shape not available.
                        z = torch.normal(
                            mean=0, std=1, size=param.data.size(),
                            device=param.device, dtype=param.dtype
                        )
                    else:
                        z_old = torch.normal(
                            mean=0, std=1, size=tuple(checkpoint_shape),
                            device=param.device, dtype=param.dtype
                        )
                        z = adjust_noise_to_param(z_old, param, name)

                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        param.data -= learning_rate * (grad_sum * z + weight_decay * param.data)
                    else:
                        param.data -= learning_rate * grad_sum * z

                print(f"Applied mezo_update at global_step {global_step} with learning rate {learning_rate}.")

        elif update_type == "zo_step":
            seed = update["seed"]
            zo_eps = update["zo_eps"]

            zo_perturb_parameters(trainable_params, seed, zo_eps, trainable_params_sizes, scaling_factor=1.0)
            zo_perturb_parameters(trainable_params, seed, zo_eps, trainable_params_sizes, scaling_factor=-2.0)
            zo_perturb_parameters(trainable_params, seed, zo_eps, trainable_params_sizes, scaling_factor=1.0)

            print(f"Applied perturbation (zo_step) with seed {seed} and epsilon {zo_eps}.")

    return model
