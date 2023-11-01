import random
from dataclasses import dataclass
from math import log, sqrt

import torch as t
from tqdm import tqdm

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")


@dataclass
class SGLDParams:
    gamma: float = 1
    epsilon: float = 0.001
    n_steps: int = 10000
    m: int = 512


def eval_model(model, loader, max_label):
    model.eval()
    model.to(DEVICE)

    avg_loss = 0
    n = 0
    with t.no_grad():
        for data, target in loader:
            out = model(data.to(DEVICE)).cpu()
            if len(out.shape) == 1:
                out = out.unsqueeze(0)  # add batch dimension
            avg_loss += cross_entropy_loss(out.cpu(), target, max_label=max_label)
            n += 1
    return avg_loss / n


def cross_entropy_loss(logits, y_s, max_label=9):
    """
    logits: outputs of model
    y: target labels

    returns: mean cross entropy loss
    """
    preds = t.nn.functional.softmax(logits[:, : max_label + 1], dim=1)
    return -1 * t.mean(t.log(preds[t.arange(len(preds)), y_s] + 1e-6))


def sgld(
    model,
    sgld_params,
    dataset,
    data_loader,
    max_label=9,
):
    n = len(dataset)
    model = model.to(DEVICE)
    beta = 1 / log(n)

    init_loss = eval_model(model, data_loader, max_label)
    n_ln_wstar = n * init_loss
    idx = list(range(len(dataset)))
    optimizer = t.optim.SGD(
        model.parameters(),
        weight_decay=0,
        lr=1,
    )

    w_0 = (
        t.nn.utils.parameters_to_vector(model.parameters()).detach().clone().to(DEVICE)
    )

    array_loss = []
    array_weight_norm = []
    full_losses = []

    for _ in tqdm(range(sgld_params.n_steps)):
        batch_idx = random.choices(idx, k=sgld_params.m)
        X = t.stack([dataset[b][0] for b in batch_idx]).to(DEVICE)
        Y = t.stack([t.tensor(dataset[b][1]) for b in batch_idx]).to(DEVICE)
        optimizer.zero_grad()
        out = model(X)

        cross_entropy_loss_value = cross_entropy_loss(out, Y, max_label=max_label)
        array_loss.append(cross_entropy_loss_value.item())

        w = t.nn.utils.parameters_to_vector(model.parameters())
        array_weight_norm.append(w.norm(p=2).item())

        elasticity_loss_term = (sgld_params.gamma / 2) * t.sum(((w_0 - w) ** 2))
        log_likelihood_loss_term = cross_entropy_loss_value * n * beta

        full_loss = (sgld_params.epsilon / 2) * (
            elasticity_loss_term + log_likelihood_loss_term
        )
        full_losses.append(
            (elasticity_loss_term.item(), log_likelihood_loss_term.item())
        )

        full_loss.backward()
        optimizer.step()

        eta = t.randn_like(w, device=DEVICE) * sqrt(sgld_params.epsilon)
        with t.no_grad():
            new_params = t.nn.utils.parameters_to_vector(model.parameters()) + eta
            t.nn.utils.vector_to_parameters(new_params, model.parameters())

    wbic = n * sum(array_loss) / len(array_loss)
    lambda_hat = (wbic - n_ln_wstar) / log(n)

    print(f"lambda_hat: {lambda_hat}")
    # print(f"wbic: {wbic}")
    # print(f"n_ln_wstar: {n_ln_wstar}")
    # print(f"init_loss: {init_loss}")
    # print(f"sgld_params: {sgld_params}")
    print(f"array_loss: {array_loss[::len(array_loss)//20]}")
    # print(f"array_weight_norm: {array_weight_norm[::len(array_weight_norm)//20]}")
    # print(f"full_losses: {full_losses[::len(full_losses)//20]}")
    return model, lambda_hat
