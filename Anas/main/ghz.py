import jax
import jax.numpy as jnp
from jax import random
import optax

from models import RBM, FFN, CNN, NeuralQuantumState


def enumerate_spin_basis(L):
    """
    Returns all spin configurations in {-1, +1}^L
    shape: (2^L, L)
    """
    states = ((jnp.arange(2**L)[:, None] >> jnp.arange(L)) & 1)
    states = 2 * states - 1
    return states.astype(jnp.int32)


def ghz_target_state(L):
    """
    Returns the normalized GHZ state amplitudes in the spin basis {-1,+1}^L.

    Convention:
      |GHZ> = (|+ + ... +> + |- - ... ->) / sqrt(2)
    """
    basis = enumerate_spin_basis(L)
    amps = jnp.zeros((basis.shape[0],), dtype=jnp.float32)

    all_up = jnp.all(basis == 1, axis=1)
    all_down = jnp.all(basis == -1, axis=1)

    amps = amps.at[all_up].set(1.0 / jnp.sqrt(2.0))
    amps = amps.at[all_down].set(1.0 / jnp.sqrt(2.0))

    return basis, amps


def build_model(
    model_name,
    L,
    key,
    hidden=20,
    hidden_layers=(32, 32),
    channels=16,
    kernel=3,
    n_conv_layers=1,
):
    if model_name == "RBM":
        arch = RBM(L, hidden=hidden)
        model_info = {"hidden": hidden}

    elif model_name == "FFN":
        hidden_layers = list(hidden_layers)
        arch = FFN(L, hidden_layers=hidden_layers)
        model_info = {"hidden_layers": hidden_layers}

    elif model_name == "CNN":
        arch = CNN(L, channels=channels, kernel=kernel, n_conv_layers=n_conv_layers)
        model_info = {
            "channels": channels,
            "kernel": kernel,
            "n_conv_layers": n_conv_layers,
        }

    else:
        raise ValueError(f"Unknown model: {model_name}")

    params = arch.init_params(key)
    return arch, params, model_info


def normalized_wavefunction(wf, params, basis):
    """
    Returns normalized amplitudes psi(sigma) on the full basis.
    """
    logpsi = wf.vmap_logpsi(params, basis)
    psi = jnp.exp(logpsi)
    norm = jnp.linalg.norm(psi)
    psi = psi / (norm + 1e-12)
    return psi


def ghz_fidelity(wf, params, basis, target_amps):
    """
    Fidelity with the target GHZ state:
        F = |<GHZ|psi>|^2
    """
    psi = normalized_wavefunction(wf, params, basis)
    overlap = jnp.vdot(target_amps, psi)
    return jnp.abs(overlap) ** 2


def ghz_loss(wf, params, basis, target_amps):
    return 1.0 - ghz_fidelity(wf, params, basis, target_amps)


def ghz_probability_on_special_states(wf, params, basis):
    """
    Returns probabilities on the two GHZ basis states:
      all-up and all-down
    """
    psi = normalized_wavefunction(wf, params, basis)
    probs = jnp.abs(psi) ** 2

    all_up = jnp.all(basis == 1, axis=1)
    all_down = jnp.all(basis == -1, axis=1)

    p_up = jnp.sum(probs[all_up])
    p_down = jnp.sum(probs[all_down])
    p_rest = 1.0 - p_up - p_down

    return p_up, p_down, p_rest


def train_ghz(
    model_name="RBM",
    *,
    L=10,
    n_steps=500,
    lr=1e-2,
    seed=0,
    hidden=20,
    hidden_layers=(32, 32),
    channels=16,
    kernel=3,
    n_conv_layers=1,
    verbose=True,
):
    key = random.PRNGKey(seed)
    key, model_key = random.split(key)

    basis, target_amps = ghz_target_state(L)

    arch, params, model_info = build_model(
        model_name=model_name,
        L=L,
        key=model_key,
        hidden=hidden,
        hidden_layers=hidden_layers,
        channels=channels,
        kernel=kernel,
        n_conv_layers=n_conv_layers,
    )

    wf = NeuralQuantumState(arch, params, L)

    optimizer = optax.adam(lr)
    flat_params, unravel = wf.flatten_params(params)
    opt_state = optimizer.init(flat_params)

    def loss_from_flat(flat_params):
        p = unravel(flat_params)
        return ghz_loss(wf, p, basis, target_amps)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_from_flat))

    losses = []
    fidelities = []
    p_up_hist = []
    p_down_hist = []
    p_rest_hist = []

    for step in range(n_steps):
        loss_val, grad = loss_and_grad(flat_params)
        updates, opt_state = optimizer.update(grad, opt_state, flat_params)
        flat_params = optax.apply_updates(flat_params, updates)

        params = unravel(flat_params)
        fid = ghz_fidelity(wf, params, basis, target_amps)
        p_up, p_down, p_rest = ghz_probability_on_special_states(wf, params, basis)

        losses.append(loss_val)
        fidelities.append(fid)
        p_up_hist.append(p_up)
        p_down_hist.append(p_down)
        p_rest_hist.append(p_rest)

        if verbose and (step % max(1, n_steps // 20) == 0 or step == n_steps - 1):
            print(
                f"step {step:4d} | "
                f"loss = {float(loss_val):.8f} | "
                f"fidelity = {float(fid):.8f} | "
                f"p_up = {float(p_up):.6f} | "
                f"p_down = {float(p_down):.6f} | "
                f"p_rest = {float(p_rest):.6f}"
            )

    return {
        "model": model_name,
        "model_info": model_info,
        "L": L,
        "basis": basis,
        "target_amps": target_amps,
        "final_params": params,
        "wavefunction": wf,
        "loss": jnp.array(losses),
        "fidelity": jnp.array(fidelities),
        "p_up": jnp.array(p_up_hist),
        "p_down": jnp.array(p_down_hist),
        "p_rest": jnp.array(p_rest_hist),
    }