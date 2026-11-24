"""Compute-budgeted proposal policies for the early-fusion BioVid comparison."""

from dataclasses import replace
import time

import numpy as np
import optuna
import tensorflow as tf

from painnas.data import (
    compute_source_normalization,
    indices_for_subjects,
    make_tf_dataset,
)
from painnas.io import atomic_write_json, read_json
from painnas.model import build_model, compile_model
from painnas.runtime import reset_runtime
from painnas.search import architecture_from_parameters, suggest_architecture


def candidate_pool(seed, size):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    candidates = []
    while len(candidates) < size:
        trial = study.ask()
        try:
            suggest_architecture(trial)
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        study.tell(trial, 0.0)
        candidates.append(dict(trial.params))
    return candidates


def synflow(model, shape):
    # Differentiate the sum of pre-softmax logits (softmax sums to one).
    for weight in model.trainable_variables:
        weight.assign(tf.abs(weight))
    normalization = (
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.LayerNormalization,
        tf.keras.layers.GroupNormalization,
    )
    with tf.GradientTape() as tape:
        hidden = tf.ones((1, *shape))
        # Centering an all-ones input erases the positive flow in normalized paths.
        for layer in model.layers[1:-1]:
            if not isinstance(layer, normalization):
                hidden = layer(hidden, training=False)
        score = tf.reduce_sum(hidden @ model.layers[-1].kernel)
    gradients = tape.gradient(score, model.trainable_variables)
    return sum(
        float(tf.reduce_sum(tf.abs(w * g)))
        for w, g in zip(model.trainable_variables, gradients)
        if g is not None
    )


class Deadline(tf.keras.callbacks.Callback):
    def __init__(self, deadline, heartbeat=None):
        super().__init__()
        self.deadline = deadline
        self.heartbeat = heartbeat

    def on_train_batch_end(self, batch, logs=None):
        if self.heartbeat:
            self.heartbeat()
        if time.monotonic() >= self.deadline:
            raise optuna.TrialPruned("Reviewer time budget exhausted")


class ReviewerPolicy:
    def __init__(self, method, seconds, arrays, config, inner, seed, pool_size):
        self.method = method
        self.seconds = seconds
        self.arrays = arrays
        self.config = config
        self.inner = inner
        self.seed = seed
        self.pool_size = pool_size
        self.random_sampler = method != "tpe"
        self.enqueue_baseline = True
        self.manifest = dict(
            method=method, seconds=seconds, pool_size=pool_size, seed=seed
        )

    def deadline_callback(self):
        return Deadline(self.deadline, getattr(self, "heartbeat", None))

    def rank_pool(self, candidates):
        arrays, config = self.arrays, self.config
        shape = (arrays.num_modalities, arrays.sequence_length, 1)
        scores = np.zeros(len(candidates))
        eligible = np.ones(len(candidates), dtype=bool)
        folds = self.inner if self.method == "weight_sharing" else [()]
        for fold, validation in enumerate(folds):
            # Materialize a supernet path at a time. Compatible named tensors
            # share weights; each inner fold starts with an empty bank.
            bank = {}
            if validation:
                development = set().union(*self.inner)
                train = indices_for_subjects(
                    arrays,
                    development - set(validation),
                    split_code=arrays.train_split_code,
                )
                val = indices_for_subjects(
                    arrays, validation, split_code=arrays.test_split_code
                )
                mean, std = compute_source_normalization(arrays.X, train)

                def dataset(indices, training):
                    return make_tf_dataset(
                        arrays,
                        indices,
                        mean=mean,
                        std=std,
                        batch_size=config.batch_size,
                        training=training,
                        seed=self.seed + fold,
                    )

            for training in [True, False] if validation else [False]:
                for i, params in enumerate(candidates):
                    if hasattr(self, "heartbeat"):
                        self.heartbeat()
                    if time.monotonic() >= self.deadline:
                        raise TimeoutError("Screening exceeded its budget")
                    if not eligible[i]:
                        continue
                    reset_runtime(self.seed + i + fold * 1000)
                    spec = replace(
                        architecture_from_parameters(params), learning_rate=1e-4
                    )
                    model = build_model(
                        spec, input_shape=shape, num_classes=config.num_classes
                    )
                    if model.count_params() > config.max_parameters:
                        eligible[i] = False
                        continue
                    if self.method == "synflow":
                        scores[i] = synflow(model, shape)
                        continue
                    for w in model.weights:
                        key = (w.path, tuple(w.shape))
                        if key in bank:
                            w.assign(bank[key])
                    if training:
                        compile_model(model, spec)
                        model.fit(dataset(train, True).take(4), epochs=1, verbose=0)
                        for w in model.weights:
                            bank[w.path, tuple(w.shape)] = w.numpy()
                    else:
                        predicted = model.predict(
                            dataset(val, False), verbose=0
                        ).argmax(axis=1)
                        correct = predicted == arrays.y[val]
                        scores[i] += sum(
                            correct[arrays.subjects[val] == s].mean()
                            for s in validation
                        )
        return [
            int(i)
            for i in np.argsort(-scores, kind="stable")
            if eligible[i] and np.isfinite(scores[i])
        ], scores

    def optimize(self, study, objective, persist, directory):
        path = directory / "budget.json"
        state = read_json(path) if path.exists() else {"seconds": 0.0, "done": False}
        if state["done"]:
            return
        start = time.monotonic()
        prior = state["seconds"]
        self.deadline = start + self.seconds - prior
        last_write = start

        def heartbeat():
            nonlocal last_write
            now = time.monotonic()
            if now - last_write >= 15:
                state["seconds"] = prior + now - start
                atomic_write_json(path, state)
                last_write = now

        self.heartbeat = heartbeat
        rank_path = directory / "screening.json"
        try:
            while time.monotonic() < self.deadline:
                trials = study.trials
                if not any(t.state == optuna.trial.TrialState.WAITING for t in trials):
                    count = len(trials)
                    if self.method in {"synflow", "weight_sharing"}:
                        if not rank_path.exists():
                            pool = candidate_pool(self.seed, self.pool_size)
                            order, scores = self.rank_pool(pool)
                            atomic_write_json(
                                rank_path, dict(pool=pool, scores=scores, order=order)
                            )
                        ranking = read_json(rank_path)
                        position = count - 1  # baseline confirmation runs first
                        if position >= len(ranking["order"]):
                            break
                        study.enqueue_trial(ranking["pool"][ranking["order"][position]])
                    elif self.method == "generative":
                        # Smoothed product of elite empirical marginals over a
                        # random valid architecture pool, learned per outer block.
                        pool = candidate_pool(self.seed + count, 16)
                        elite = sorted(
                            study.get_trials(
                                states=(optuna.trial.TrialState.COMPLETE,)
                            ),
                            key=lambda t: t.value,
                            reverse=True,
                        )
                        elite = (
                            elite[: (len(elite) + 1) // 2] if len(elite) >= 3 else []
                        )
                        weights = [
                            sum(
                                np.log1p(sum(t.params.get(k) == v for t in elite))
                                for k, v in p.items()
                                if k != "learning_rate"
                            )
                            for p in pool
                        ]
                        probs = np.exp(weights - np.max(weights))
                        rng = np.random.default_rng(self.seed + count)
                        study.enqueue_trial(
                            pool[rng.choice(len(pool), p=probs / probs.sum())]
                        )
                study.optimize(
                    objective,
                    n_trials=1,
                    callbacks=[persist],
                    catch=(tf.errors.ResourceExhaustedError,),
                    gc_after_trial=True,
                )
                state["seconds"] = prior + time.monotonic() - start
                atomic_write_json(path, state)
        except TimeoutError:
            pass
        completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        state.update(
            seconds=prior + time.monotonic() - start,
            done=True,
            confirmed=len(completed),
            confirmed_after_baseline=sum(t.number != 0 for t in completed),
        )
        atomic_write_json(path, state)


class PanelConfirmation:
    """Confirm one prespecified panel member across every inner fold."""

    random_sampler = True
    enqueue_baseline = False

    def __init__(self, params):
        self.params = params
        self.manifest = {"diagnostic_candidate": params}

    def deadline_callback(self):
        return Deadline(float("inf"))

    def optimize(self, study, objective, persist, directory):
        if not study.trials:
            study.enqueue_trial(self.params)
        if not study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
            if not any(
                t.state == optuna.trial.TrialState.WAITING for t in study.trials
            ):
                study.enqueue_trial(self.params)
            study.optimize(objective, n_trials=1, callbacks=[persist])
