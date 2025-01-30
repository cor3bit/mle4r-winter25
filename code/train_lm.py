import os.path
import time
from functools import partial
from itertools import product
from typing import Optional
import sqlite3
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

import jax
from jax.tree import map as tree_map
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import optax
from optax import softmax_cross_entropy, softmax_cross_entropy_with_integer_labels
from jaxopt import OptaxSolver

from model_zoo import NanoLM

TASK = 'tiny_shakespeare'
MODEL = 'NanoLM'

GPU = True
JIT = True

SAVE_TO_DB = False
OVERWRITE_RUN = False

WANDB = False
WANDB_PROJECT = 'mle4r'

# STEPS = 250_000
# EVAL_EVERY = 2_500

# STEPS = 100_000
# EVAL_EVERY = 1_000

STEPS = 2_000
EVAL_EVERY = 100

N_SEEDS = 1

SOLVER_HPS = {
    # 'sgd': {
    #     'b': [32, ],
    #     'lr': [0.05,  ],
    # },

    # 'adam': {
    #     'b': [32, ],
    #     'lr': [0.005, ],  # 0.0001,
    # },
    #
    'adamw': {
        'b': [128, ],
        'lr': [0.005, ],  # 0.0001,
        'wd': [0.0001, ],  # 0.0001,
    },
}


# ---------------------------- DB Utils ----------------------------

def get_connection():
    db_folder = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'db')
    db_path = os.path.join(db_folder, 'experiments.db')

    if not os.path.exists(db_path):
        raise FileNotFoundError(f'Database file not found: {db_path}')

    conn = sqlite3.connect(db_path)
    return conn


def get_run_config(task_id, model_id, solver_type, seed, total_timesteps, eval_every, config):
    # !! Matches the columns in the DB

    full_run_config = {
        "task_id": task_id,
        "model": model_id,
        "seed": seed,
        "batch_size": config['b'],
        "total_timesteps": total_timesteps,
        "eval_every": eval_every,

        "optimizer": solver_type,
        "learning_rate": config['lr'],
    }

    if 'wd' in config:
        full_run_config['weight_decay'] = config['wd']

    return full_run_config


def get_hash_id(run_config):
    # serialize the full run dictionary
    unique_run_desc = json.dumps(run_config, sort_keys=True)

    # hash the serialized string
    hash_id = hashlib.blake2s(unique_run_desc.encode(), digest_size=8).hexdigest()

    return hash_id


def get_full_run_config(task_id, model_id, solver_type, seed, total_timesteps, eval_every, config):
    run_config = get_run_config(task_id, model_id, solver_type, seed, total_timesteps, eval_every, config)
    hash_id = get_hash_id(run_config)
    run_config['RunID'] = hash_id
    return run_config


def get_run_id(task_id, model_id, solver_type, seed, total_timesteps, eval_every, config):
    run_config = get_run_config(task_id, model_id, solver_type, seed, total_timesteps, eval_every, config)
    hash_id = get_hash_id(run_config)
    return hash_id


def run_exists(conn, run_id):
    return conn.execute("SELECT EXISTS(SELECT 1 FROM Run WHERE RunID = ?)",
                        (run_id,)).fetchone()[0] == 1


def create_step_df(run_id, metric, results, eval_every):
    df = pd.DataFrame(np.array(results), columns=["Time", "Value"])
    df["RunID"] = run_id
    df["Metric"] = metric
    df["Step"] = np.arange(len(results)) * eval_every
    df = df[["RunID", "Metric", "Step", "Time", "Value"]]  # ensure column order as in DB
    return df


def create_run_df(task_id, model_id, solver_type, seed, steps, eval_every, config):
    full_run_config = get_full_run_config(task_id, model_id, solver_type, seed, steps, eval_every, config)
    return pd.DataFrame([full_run_config])


# ---------------------------- Solver ----------------------------


def create_solver(
        solver_type,
        config,
        w_params,
        X_train,
        y_train,
        n_classes,
        rng_key,
):
    # TODO verify that w_params are not modified
    # opt_params = tree_map(lambda x: x, w_params)

    solver_config = config.values()

    if solver_type == 'sgd':
        b, lr = solver_config
        solver_desc = f'{solver_type}_b{b}_lr{lr}'
        solver = OptaxSolver(loss_fun, opt=optax.sgd(lr))
        opt_state = solver.init_state(w_params, X_train[0:b], y_train[0:b], rng_key)
    elif solver_type == 'adam':
        b, lr = solver_config
        solver_desc = f'{solver_type}_b{b}_lr{lr}'
        solver = OptaxSolver(loss_fun, opt=optax.adam(lr))
        opt_state = solver.init_state(w_params, X_train[0:b], y_train[0:b], rng_key)
    elif solver_type == 'adamw':
        b, lr, wd = solver_config
        solver_desc = f'{solver_type}_b{b}_lr{lr}'
        solver = OptaxSolver(loss_fun, opt=optax.adamw(lr, weight_decay=wd))
        opt_state = solver.init_state(w_params, X_train[0:b], y_train[0:b], rng_key)
    else:
        raise ValueError(f'Unknown solver: {solver_type}')

    update_fn = jax.jit(solver.update)

    return update_fn, opt_state, solver_desc


# ---------------------------- Datasets and Models ----------------------------
def get_size(params):
    if 'batch_stats' in params:
        # exclude batch_stats from params
        return ravel_pytree({'params': params['params']})[0].shape[0]

    return ravel_pytree(params)[0].shape[0]


def format_params(n):
    if n >= 1e9:
        return f'{n / 1e9:.1f}B'  # Billions
    elif n >= 1e6:
        return f'{n / 1e6:.1f}M'  # Millions
    elif n >= 1e3:
        return f'{n / 1e3:.1f}K'  # Thousands
    else:
        return str(n)


# ---------------------------- Viz ----------------------------


def plot_metric(results):
    # Define common time points for interpolation
    time_limit = min([values[-1][0] for values in results.values()])
    n_points = 100  # Number of points for interpolation
    common_time_points = np.linspace(0, time_limit, n_points)

    # Organize data by optimizer
    runs_by_optimizer = {}
    for (optimizer_name, seed), values in results.items():
        if optimizer_name not in runs_by_optimizer:
            runs_by_optimizer[optimizer_name] = []
        times, run_values = zip(*values)
        runs_by_optimizer[optimizer_name].append((np.array(times), np.array(run_values)))

    # Interpolate runs for each optimizer
    interpolated_stats = {}
    for optimizer_name, runs in runs_by_optimizer.items():
        n_seeds = len(runs)
        interpolated_values = np.zeros((n_points, n_seeds))
        for i, (times, values) in enumerate(runs):
            interp_fn = interp1d(times, values, kind='linear', bounds_error=True)
            interpolated_values[:, i] = interp_fn(common_time_points)
        interpolated_stats[optimizer_name] = interpolated_values.mean(axis=1), interpolated_values.std(axis=1)

    # plotting
    fig, ax = plt.subplots()

    for optimizer_name, (avg_values, std_values) in interpolated_stats.items():
        ax.plot(common_time_points, avg_values, label=f'{optimizer_name}', linewidth=1)
        ax.fill_between(common_time_points, avg_values - std_values, avg_values + std_values, alpha=0.25)

        # print mean and std
        print(f"{optimizer_name}: {avg_values[-1]:.3f} +- {std_values[-1]:.3f}")

    # save plot on disk
    ax.set_ylabel(f'Accuracy on Test Set')
    ax.set_xlabel('Wall Time (s)')
    ax.legend(loc='upper right')
    # ax.set_yscale('log')
    # ax.set_xlabel(f'Environment Steps ($\\times {scale_str}%$)')
    # ax.set_title(env_name)

    # TODO limiter
    # if TASK == 'california_housing':
    #     ax.set_ylim(None, 1.1)
    # elif TASK == 'superconduct':
    #     ax.set_ylim(None, 20.1)

    # TASK = 'NanoLM'

    ax.set_title(f'{TASK}, Iterations={STEPS}')

    # modifier = os.path.basename(__file__).replace('.py', '')
    foldname = os.path.join('..', 'artifacts', 'examples', TASK)
    if not os.path.exists(foldname):
        os.makedirs(foldname)

    fname = os.path.join(foldname, f'{TASK}.png')
    fig.savefig(fname, bbox_inches='tight', dpi=600)


def calc_size(steps_in_run, configs):
    n_runs = sum(len(c) for _, c in configs.items())
    print(f'Number of distinct solvers: {n_runs}')
    return n_runs * steps_in_run * N_SEEDS


def create_configs(solvers):
    configs = {}

    for solver_type, solver_params in solvers.items():
        # solver_configs = list(product(*solver_params.values()))
        solver_configs = [dict(zip(solver_params.keys(), values)) for values in product(*solver_params.values())]
        configs[solver_type] = solver_configs

    return configs


@partial(jax.jit, static_argnums=(2, 3))
def get_batch(random_key, data, batch_size, block_size):
    """Prepares a random batch of training data.

    Args:
        random_key: A random seed for sampling a batch.
        data: The complete training dataset.

    Returns:
        x: Input sequences.
        y: Target sequences (shifted inputs).
    """
    ix = jax.random.randint(
        random_key, shape=(batch_size, 1), minval=0, maxval=len(data) - block_size
    )
    x = dynamic_slice_vmap(data, ix, (block_size,))
    y = dynamic_slice_vmap(data, ix + 1, (block_size,))
    return x, y


@jax.jit
def eval_step(params, x, y):
    logits = model.apply(params, x, training=False)
    loss = ce_loss(logits=logits, labels=y)
    return loss


#
# # we define one iteration of the optimizer and JIT this function
# @partial(jax.jit, static_argnums=(3, 4))
# def step(key, params, opt_state, batch_size, block_size):
#     key, subkey = jax.random.split(key)
#     batch = get_batch(key, train_data, batch_size, block_size)
#
#     loss, grad = jax.value_and_grad(loss_fun)(params, *batch, subkey)
#     updates, opt_state = opt.update(grad, opt_state, params)
#     params = optax.apply_updates(params, updates)
#
#     return params, key, opt_state, loss


def ce_loss(logits, labels):
    return softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()


# ---------------------------- Main Script ----------------------------


if __name__ == '__main__':
    def loss_fun(params, x, y, dropout_key):
        logits = model.apply(params, x, training=True, rngs={"dropout": dropout_key})
        loss = ce_loss(logits=logits, labels=y)
        return loss


    #
    # @jax.jit
    # def ce(params, x, y):
    #     logits = predict_fn(params, x)
    #     return jnp.mean(softmax_cross_entropy(logits, y))
    #
    #
    # @jax.jit
    # def accuracy(params, X, Y_true):
    #     logits = predict_fn(params, X)
    #     predicted_classes = jnp.argmax(logits, axis=1)
    #     correct_predictions = predicted_classes == Y_true
    #     return jnp.mean(correct_predictions)

    # --------------- config ---------------
    # force jax to use CPU
    if not GPU:
        jax.config.update('jax_platform_name', 'cpu')

    # for debugging JAX-related issues
    # jax.config.update('jax_enable_x64', True)
    # jax.config.update('jax_debug_nans', True)

    jax.config.update('jax_disable_jit', not JIT)

    task_hps = SOLVER_HPS

    configs = create_configs(task_hps)
    total_size = calc_size(STEPS, configs)

    # --------------- DB ---------------
    conn = get_connection() if SAVE_TO_DB else None

    # --------------- dataset & models ---------------
    ds = tfds.load("tiny_shakespeare", data_dir=os.path.join('..', 'artifacts', 'data'))

    # combine train and test examples into a single string
    text_train = ""
    for example in ds["train"].concatenate(ds["test"]).as_numpy_iterator():
        text_train += example["text"].decode("utf-8")

    # similarly, create a single string for validation
    text_validation = ""
    for example in ds["validation"].as_numpy_iterator():
        text_validation += example["text"].decode("utf-8")

    vocab = sorted(list(set(text_train)))

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    encode = lambda s: [
        stoi[c] for c in s
    ]  # encoder: take a string, output a list of integers
    decode = lambda l: "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string

    # transfer train and validation data to JAX arrays
    train_data = jnp.array(encode(text_train))
    eval_data = jnp.array(encode(text_validation))

    # prepare a function for retrieving a batch of data
    dynamic_slice_vmap = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))

    # --------------- Model ---------------
    # TODO: move from hardcode to config
    block_size = 64
    vocab_size = len(vocab)
    model_params = {
        'vocab_size': vocab_size,
        'num_layers': 6,
        'num_heads': 8,
        'head_size': 32,
        'dropout_rate': 0.2,
        'embed_size': 256,
        'block_size': block_size,
    }
    model = NanoLM(**model_params)

    # --------------- training loop ---------------
    results = {}

    for solver_type, solver_params in task_hps.items():
        for config in configs[solver_type]:
            for seed in range(N_SEEDS):

                if SAVE_TO_DB:
                    run_id = get_run_id(TASK, MODEL, solver_type, seed, STEPS, EVAL_EVERY, config)

                    if run_exists(conn, run_id):
                        if OVERWRITE_RUN:
                            print(f'Run {run_id} already exists in the DB. Overwriting.')
                            # TODO delete corresponding Run/Steps from DB
                            raise NotImplementedError('delete run from DB not implemented yet')
                        else:
                            print(f'Run {run_id} already exists in the DB. Skipping.')
                            # TODO fix!!
                            # pbar.update(STEPS)
                            continue

                if WANDB:
                    import wandb

                    wandb.init(
                        project=WANDB_PROJECT,
                        entity='dysco',
                        config=get_full_run_config(TASK, MODEL, solver_type, seed, STEPS, EVAL_EVERY, config),
                        name=get_run_id(TASK, MODEL, solver_type, seed, STEPS, EVAL_EVERY, config),
                    )

                # --------------- random part ---------------
                batch_size = config['b']

                # ! RANDOMNESS (1): the seed is used to initialize the model
                rng_key = jax.random.PRNGKey(seed)

                train_sample_x = jnp.ones((batch_size, block_size), dtype=jnp.int32)

                params = model.init(
                    rng_key,
                    jnp.ones((batch_size, block_size), dtype=jnp.int32),
                    training=False,
                )

                n_params = get_size(params)
                print(f'Selected model: {MODEL}, d={n_params:,}')

                # generate
                rng_key, subkey = jax.random.split(rng_key)
                text = model.generate(rng_key, params, 1000)[:, 0, 0].tolist()
                print(decode(text))

                train_sample_y = jnp.ones((batch_size, block_size), dtype=jnp.int32)

                predict_fn = jax.jit(model.apply, static_argnames=['training'])

                # --------------- init solvers ---------------

                update_fn, opt_state, solver_desc = create_solver(
                    solver_type, config, params, train_sample_x, train_sample_y, vocab_size, rng_key)

                results[(solver_desc, seed)] = []

                # --------------- warm up ---------------
                # compile JAX functions to make fair comparison of Wall Time
                n_warmup = 2
                for i in range(n_warmup):
                    dummy_rng_key = jax.random.PRNGKey(1337)
                    # var_params, key, opt_state, loss = step(key, var_params, opt_state, batch_size, block_size)
                    dummy_x, dummy_y = get_batch(dummy_rng_key, train_data, batch_size, block_size)

                    # on the full Test Set
                    eval_step(params, dummy_x, dummy_y)

                    # on batches like in training
                    preds = predict_fn(params, dummy_x, training=False)
                    preds = predict_fn(params, dummy_x, training=True, rngs={"dropout": dummy_rng_key})
                    loss = loss_fun(params, dummy_x, dummy_y, dummy_rng_key)

                    update_fn(params, opt_state, train_sample_x, train_sample_y, dummy_rng_key)

                # --------------- training loop ---------------
                total_update_time = 0

                for step in tqdm(range(STEPS + 1)):
                    # EVALUATION
                    if step % EVAL_EVERY == 0:
                        rng_key, subkey = jax.random.split(rng_key)
                        eval_batch = get_batch(subkey, eval_data, batch_size, block_size)

                        # on Test
                        # loss = accuracy_fn(opt_params, X_test, Y_test)
                        loss = eval_step(params, *eval_batch)

                        # first time is zero
                        if not results[(solver_desc, seed)]:
                            start_time = time.time()
                            results[(solver_desc, seed)].append((0, loss))

                            if WANDB:
                                wandb.log({'eval/accuracy': loss, 'eval/time': 0, 'eval/step': step, })

                        else:
                            delta_t = time.time() - start_time
                            results[(solver_desc, seed)].append((delta_t, loss))

                            if WANDB:
                                wandb.log({'eval/accuracy': loss, 'eval/time': delta_t, 'eval/step': step,
                                           'train/avg_update_time': total_update_time / step, })

                        # print(f'{solver_desc}, step: {step}, loss: {loss:.6f}')

                    # TRAINING
                    rng_key, subkey = jax.random.split(rng_key)
                    batch_X, batch_y = get_batch(subkey, train_data, batch_size, block_size)

                    # UPDATE PARAMS
                    start_update_time = time.time()
                    params, opt_state = update_fn(params, opt_state, batch_X, batch_y, rng_key)

                    # fair update time, not including the evaluation
                    update_time = time.time() - start_update_time
                    total_update_time += update_time

                # --------------- save results after training completion ---------------
                avg_update_time = total_update_time / step
                print(f'{solver_desc}, avg_update_time: {avg_update_time:.6f}')

                # generate
                rng_key, subkey = jax.random.split(rng_key)
                text = model.generate(rng_key, params, 1000)[:, 0, 0].tolist()
                print(decode(text))

                if SAVE_TO_DB:
                    run_df = create_run_df(TASK, MODEL, solver_type, seed, STEPS, EVAL_EVERY, config)
                    run_df.to_sql("Run", conn, if_exists="append", index=False)
                    step_df = create_step_df(run_id, 'accuracy', results[(solver_desc, seed)], EVAL_EVERY)
                    step_df.to_sql("Step", conn, if_exists="append", index=False)

                if WANDB:
                    wandb.finish()

    # --------------- plotting ---------------
    if results:
        plt.style.use('bmh')
        plot_metric(results)

    if SAVE_TO_DB:
        conn.close()
