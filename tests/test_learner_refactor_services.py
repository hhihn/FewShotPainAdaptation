import unittest
import csv
import logging
import tempfile
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from learner.cross_validation_results import CrossValidationResultRecorder
from learner.episode_evaluation_service import EpisodeEvaluationService
from learner.heldout_adaptation_service import HeldoutAdaptationService
from learner.few_shot_pain_learner import FewShotPainLearner
from learner.task_batch_pipeline import TaskBatchPipeline
from data_loaders.meta_ds_sampler import SixWayKShotSampler


class _FakeSampler:
    def __init__(self):
        self.n_way = 2
        self.k_shot = 1
        self.q_query = 1
        self.support_size = 2
        self.query_size = 2
        self.rng = np.random.default_rng(123)

    def get_task(self):
        self.rng.integers(0, 1_000_000)
        return {
            "support_X": np.zeros((self.support_size, 3, 1), dtype=np.float32),
            "support_y": np.repeat(np.arange(self.n_way), self.k_shot).astype(np.int32),
            "query_X": np.zeros((self.query_size, 3, 1), dtype=np.float32),
            "query_y": np.repeat(np.arange(self.n_way), self.q_query).astype(np.int32),
        }


class _CountingPhase2Sampler:
    def __init__(self):
        self.n_way = 2
        self.k_shot = 2
        self.q_query = 3
        self.support_size = self.n_way * self.k_shot
        self.query_size = self.n_way * self.q_query
        self.active_subjects_array = np.array([1, 2, 3], dtype=np.int32)
        self.data_split = "train"
        self.calls = 0

    def get_task(self):
        self.calls += 1
        return {
            "support_X": np.full(
                (self.support_size, 5, 2),
                fill_value=self.calls,
                dtype=np.float32,
            ),
            "support_y": np.repeat(np.arange(self.n_way), self.k_shot).astype(np.int32),
            "query_X": np.full(
                (self.query_size, 5, 2),
                fill_value=self.calls + 100,
                dtype=np.float32,
            ),
            "query_y": np.repeat(np.arange(self.n_way), self.q_query).astype(np.int32),
        }


class _FakeEngine:
    def __init__(self, fail_after: int | None = None):
        self.calls = 0
        self.fail_after = fail_after

    def train_step(self, support_x, support_y, query_x, query_y):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("forced adaptation failure")
        return tf.constant(float(self.calls), dtype=tf.float32), tf.constant(1.0)


class _FakeEvaluationEngine:
    def __init__(self):
        self.model = SimpleNamespace(can_support_mode="learned_prototype_memory")
        self.seen_support_modes = []

    def forward_task(
        self,
        support_x,
        support_y,
        query_x,
        query_y,
        training=False,
        return_similarity_scores=True,
    ):
        del support_x, support_y, training, return_similarity_scores
        self.seen_support_modes.append(self.model.can_support_mode)
        num_query = int(query_x.shape[0])
        logits = tf.one_hot(query_y, depth=2, dtype=tf.float32) * 2.0
        similarity_scores = tf.one_hot(query_y, depth=2, dtype=tf.float32)
        return {
            "loss": tf.constant(0.25, dtype=tf.float32),
            "task_loss": tf.constant(0.2, dtype=tf.float32),
            "contrastive_loss": tf.constant(0.0, dtype=tf.float32),
            "triplet_loss": tf.constant(0.0, dtype=tf.float32),
            "can_local_loss": tf.constant(0.01, dtype=tf.float32),
            "can_global_loss": tf.constant(0.02, dtype=tf.float32),
            "can_margin_loss": tf.constant(0.03, dtype=tf.float32),
            "logits": tf.reshape(logits, (num_query, 2)),
            "similarity_scores": tf.reshape(similarity_scores, (num_query, 2)),
        }


class _FakeFeatureExportEngine:
    def __init__(self):
        self.model = SimpleNamespace(can_support_mode="learned_prototype_memory")
        self.triplet_loss_weight = 1.0

    def forward_task(
        self,
        support_x,
        support_y,
        query_x,
        query_y,
        training=False,
        return_similarity_scores=True,
    ):
        del support_x, training, return_similarity_scores
        support_size = int(support_y.shape[0])
        query_size = int(query_y.shape[0])
        logits = tf.one_hot(query_y, depth=2, dtype=tf.float32) * 2.0
        return {
            "logits": logits,
            "loss": tf.constant(0.2, dtype=tf.float32),
            "task_loss": tf.constant(0.2, dtype=tf.float32),
            "contrastive_loss": tf.constant(0.0, dtype=tf.float32),
            "triplet_loss": tf.constant(0.0, dtype=tf.float32),
            "can_local_loss": tf.constant(0.0, dtype=tf.float32),
            "can_global_loss": tf.constant(0.0, dtype=tf.float32),
            "can_margin_loss": tf.constant(0.0, dtype=tf.float32),
            "model_aux_loss": tf.constant(0.0, dtype=tf.float32),
            "support_feature_maps": tf.reshape(
                tf.range(support_size * 3 * 2, dtype=tf.float32),
                (support_size, 3, 2),
            ),
            "query_feature_maps": tf.reshape(
                tf.range(query_size * 3 * 2, dtype=tf.float32),
                (query_size, 3, 2),
            ),
            "prototype_feature_maps": tf.reshape(
                tf.range(2 * 3 * 2, dtype=tf.float32),
                (2, 3, 2),
            ),
            "similarity_scores": tf.one_hot(query_y, depth=2, dtype=tf.float32),
        }


class _FakeDatasetForSampler:
    def __init__(self):
        self.config = SimpleNamespace(
            k_shot=30,
            q_query=5,
            n_way=2,
            val_tasks=1,
            tasks_per_epoch=1,
            heldout_eval_tasks=1,
            task_construction_mode="single_subject",
            task_normalize_mode="support",
            sequence_length=3,
            num_sensors=1,
        )
        self.last_use_base_index = None

    def _get_sampling_index_for_split(self, split, use_base_index=False):
        self.last_use_base_index = use_base_index
        return {
            7: {
                0: np.arange(20),
                1: np.arange(20, 40),
            }
        }

    def sample_task(
        self,
        subject,
        k_shot,
        q_query,
        normalize_mode,
        rng,
        allow_partial_query,
        split_normalization_stats,
        split,
        use_base_index,
    ):
        self.last_use_base_index = use_base_index
        support_size = self.config.n_way * int(k_shot)
        query_size = self.config.n_way * int(q_query)
        return {
            "support_X": np.zeros((support_size, 3, 1), dtype=np.float32),
            "support_y": np.repeat(np.arange(self.config.n_way), int(k_shot)).astype(
                np.int32
            ),
            "query_X": np.zeros((query_size, 3, 1), dtype=np.float32),
            "query_y": np.repeat(np.arange(self.config.n_way), int(q_query)).astype(
                np.int32
            ),
        }


class _FakePrototypeDataset:
    def build_all_query_task(
        self,
        subject,
        split,
        use_base_index,
        normalize_with_query_subject_stats,
    ):
        del subject, split, use_base_index, normalize_with_query_subject_stats
        return {
            "support_X": np.zeros((0, 3, 1), dtype=np.float32),
            "support_y": np.zeros((0,), dtype=np.int32),
            "query_X": np.zeros((4, 3, 1), dtype=np.float32),
            "query_y": np.array([0, 1, 0, 1], dtype=np.int32),
        }


class _InitializerPrototypeMemory:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.assigned = np.zeros(self.shape, dtype=np.float32)

    def assign_prototype_maps(self, maps):
        maps = np.asarray(maps, dtype=np.float32)
        if maps.shape != self.shape:
            raise ValueError(f"expected {self.shape}, got {maps.shape}")
        self.assigned = maps


class _InitializerModel:
    def __init__(self, num_classes=2, slots_per_class=2, feature_time=2, feature_dim=1):
        self.prototype_memory = _InitializerPrototypeMemory(
            (num_classes, slots_per_class, feature_time, feature_dim)
        )
        self.encoded_batches = []

    def encode_feature_map(self, x, training=False):
        self.encoded_batches.append(np.asarray(x.numpy(), dtype=np.float32))
        return x


class _FakeSourceVoteModel:
    def forward_source_subject_prototype_vote_can(
        self,
        *,
        prototype_maps,
        prototype_y,
        query_x,
        training=False,
    ):
        del prototype_maps, prototype_y, training
        query_count = int(query_x.shape[0])
        class_probabilities = tf.constant(
            [[0.8, 0.2], [0.35, 0.65]],
            dtype=tf.float32,
        )[:query_count]
        prototype_scores = tf.constant(
            [[3.0, 1.0, 0.5, 0.0], [0.1, 0.2, 2.0, 2.1]],
            dtype=tf.float32,
        )[:query_count]
        vote_weights = tf.nn.softmax(prototype_scores, axis=1)
        return {
            "logits": tf.math.log(class_probabilities),
            "similarity_scores": class_probabilities,
            "query_feature_maps": tf.ones((query_count, 2, 1), dtype=tf.float32),
            "prototype_feature_maps": tf.ones((4, 2, 1), dtype=tf.float32),
            "prototype_similarity_scores": prototype_scores,
            "prototype_vote_weights": vote_weights,
        }


class _FakeSourceVoteEngine:
    def __init__(self):
        self.model = _FakeSourceVoteModel()


class _InitializerDataset:
    def __init__(self, samples_per_subject_class=6, num_subjects=4, n_way=2):
        self.config = SimpleNamespace(n_way=n_way, task_normalize_mode="none")
        self.X = []
        self.subjects = []
        self.index_by_split = {"train": {}}
        row = 0
        for subject in range(num_subjects):
            self.index_by_split["train"][subject] = {}
            for class_id in range(n_way):
                refs = []
                for rep in range(samples_per_subject_class):
                    value = float(subject * 100 + class_id * 10 + rep)
                    self.X.append(np.full((2, 1), value, dtype=np.float32))
                    self.subjects.append(subject)
                    refs.append(row)
                    row += 1
                self.index_by_split["train"][subject][class_id] = np.asarray(
                    refs,
                    dtype=np.int64,
                )
        self.X = np.stack(self.X, axis=0)
        self.subjects = np.asarray(self.subjects, dtype=np.int32)

    def _get_sampling_index_for_split(self, split, use_base_index=False):
        self.last_use_base_index = bool(use_base_index)
        return self.index_by_split[split]

    def _gather_samples(self, refs):
        return self.X[np.asarray(refs, dtype=np.int64)]

    def _normalize_data_by_subjects(self, data, subjects):
        del subjects
        return data

    @staticmethod
    def _compute_batch_stats(data):
        return {
            "mean": np.mean(data, axis=(0, 1), keepdims=True),
            "std": np.std(data, axis=(0, 1), keepdims=True) + 1e-8,
        }

    @staticmethod
    def _apply_stats(data, stats):
        return (data - stats["mean"]) / stats["std"]

    def compute_split_normalization_stats(self, subjects, split="train"):
        refs = []
        for subject in subjects:
            for class_id in range(self.config.n_way):
                refs.extend(self.index_by_split[split][int(subject)][class_id])
        return self._compute_batch_stats(self._gather_samples(np.asarray(refs)))


def _make_initializer_learner(
    *,
    samples_per_slot=2,
    slots_per_class=2,
    samples_per_subject_class=6,
    normalize_mode="none",
):
    learner = FewShotPainLearner.__new__(FewShotPainLearner)
    learner.seed = 17
    learner.config = SimpleNamespace(
        n_way=2,
        task_normalize_mode=normalize_mode,
        can_support_mode="learned_prototype_memory",
        learned_prototype_slots_per_class=slots_per_class,
        prototype_bank_init_samples_per_class=samples_per_slot,
        source_subject_prototype_vote_use_base_index=True,
        source_subject_prototype_vote_query_normalize_with_subject_stats=True,
        source_subject_prototype_vote_softmax_scope="global",
    )
    learner.dataset = _InitializerDataset(
        samples_per_subject_class=samples_per_subject_class,
        n_way=2,
    )
    learner.dataset.config.task_normalize_mode = normalize_mode
    learner.model = _InitializerModel(num_classes=2, slots_per_class=slots_per_class)
    learner.logger = logging.getLogger("test_prototype_bank_initializer")
    train_sampler = SimpleNamespace(
        active_subjects_array=np.asarray([0, 1, 2, 3], dtype=np.int32),
        data_split="train",
        split_normalization_stats=None,
        rng=np.random.default_rng(99),
    )
    return learner, train_sampler


class LearnerRefactorServiceTests(unittest.TestCase):
    def _make_recorder(self):
        tmp = tempfile.TemporaryDirectory()
        recorder = CrossValidationResultRecorder(
            heldout_eval_pairs=[(2, 3)],
            training_progress_output_dir=tmp.name,
            csv_flush_every_events=1,
            validation_checkpoint_metric="f1",
            validation_checkpoint_mode="max",
            logger=logging.getLogger("test_cv_result_recorder"),
        )
        self.addCleanup(tmp.cleanup)
        return recorder

    def test_cv_result_recorder_initial_payload_contains_extended_keys(self):
        recorder = self._make_recorder()
        results = recorder.results
        size_bucket = results["heldout_eval_by_task_size"]["k2_q3"]

        for key in (
            "can_alignment_summary_files",
            "can_sample_statistics_files",
            "can_feature_export_files",
            "validation_checkpoint_values",
            "validation_checkpoint_metrics",
            "source_subject_prototype_vote_accuracies",
            "source_subject_prototype_vote_weight_files",
        ):
            self.assertIn(key, results)
        self.assertIn("zero_shot_accuracies", size_bucket)
        self.assertIn("k_shot_accuracies", size_bucket)
        self.assertEqual(results["validation_checkpoint_metric"], "f1")
        self.assertEqual(results["validation_checkpoint_mode"], "max")

    def test_prototype_bank_initializer_uses_training_subjects_stratified_nonoverlap(self):
        learner, train_sampler = _make_initializer_learner(samples_per_slot=2)
        rng_state_before = train_sampler.rng.bit_generator.state

        metadata = learner._initialize_prototype_bank_from_training_samples(
            fold=0,
            test_subject=3,
            train_sampler=train_sampler,
        )

        self.assertTrue(metadata["enabled"])
        self.assertEqual(train_sampler.rng.bit_generator.state, rng_state_before)
        self.assertEqual(metadata["train_subjects"], [0, 1, 2])
        assigned = learner.model.prototype_memory.assigned
        for class_id, class_metadata in metadata["classes"].items():
            refs_by_slot = [
                slot["refs"].astype(np.int64).tolist()
                for slot in class_metadata["slots"]
            ]
            flattened_refs = [ref for refs in refs_by_slot for ref in refs]
            self.assertEqual(len(flattened_refs), len(set(flattened_refs)))
            subjects = np.concatenate(
                [slot["subjects"] for slot in class_metadata["slots"]], axis=0
            )
            self.assertNotIn(3, subjects.tolist())
            counts = [int(np.sum(subjects == subject)) for subject in (0, 1, 2)]
            self.assertLessEqual(max(counts) - min(counts), 1)
            for slot_id, refs in enumerate(refs_by_slot):
                expected = np.mean(learner.dataset.X[np.asarray(refs)], axis=0)
                np.testing.assert_allclose(
                    assigned[int(class_id), slot_id],
                    expected,
                    atol=1e-6,
                )

    def test_source_subject_prototype_builder_uses_all_base_train_samples(self):
        learner, train_sampler = _make_initializer_learner(
            samples_per_slot=0,
            slots_per_class=1,
            samples_per_subject_class=3,
        )
        learner.config.attention_mode = "can"

        prototypes = learner._build_source_subject_class_prototypes(
            test_subject=3,
            train_sampler=train_sampler,
        )

        self.assertTrue(learner.dataset.last_use_base_index)
        self.assertEqual(prototypes["train_subjects"].tolist(), [0, 1, 2])
        self.assertEqual(prototypes["prototype_y"].tolist(), [0, 1, 0, 1, 0, 1])
        self.assertEqual(
            prototypes["prototype_subjects"].tolist(),
            [0, 0, 1, 1, 2, 2],
        )
        self.assertTrue(np.all(prototypes["prototype_sample_counts"] == 3))
        for idx, (subject, class_id) in enumerate(
            zip(prototypes["prototype_subjects"], prototypes["prototype_y"])
        ):
            refs = learner.dataset.index_by_split["train"][int(subject)][
                int(class_id)
            ]
            expected = np.mean(learner.dataset.X[refs], axis=0)
            np.testing.assert_allclose(
                prototypes["prototype_maps"][idx],
                expected,
                atol=1e-6,
            )

    def test_source_subject_prototype_vote_evaluator_returns_vote_metrics(self):
        evaluator = EpisodeEvaluationService(
            config=SimpleNamespace(n_way=2, attention_mode="can"),
            engine=_FakeSourceVoteEngine(),
            task_pipeline=None,
        )
        task = {
            "query_X": np.zeros((2, 3, 1), dtype=np.float32),
            "query_y": np.array([0, 1], dtype=np.int32),
        }
        prototype_maps = np.zeros((4, 2, 1), dtype=np.float32)
        prototype_y = np.array([0, 1, 0, 1], dtype=np.int32)

        loss, metrics, diagnostics = (
            evaluator.evaluate_source_subject_prototype_vote_task_metrics(
                task_dict=task,
                prototype_maps=prototype_maps,
                prototype_y=prototype_y,
            )
        )

        self.assertAlmostEqual(loss, float(-np.mean(np.log([0.8, 0.65]))), places=6)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["can_true_class_score"], 0.725, places=6)
        self.assertEqual(diagnostics["query_pred"].tolist(), [0, 1])
        self.assertEqual(diagnostics["prototype_y"].tolist(), [0, 1, 0, 1])
        np.testing.assert_allclose(
            np.sum(diagnostics["prototype_vote_weights"], axis=1),
            np.ones(2),
            atol=1e-6,
        )

    def test_source_subject_prototype_vote_weights_aggregate_to_subject_row(self):
        diagnostics = {
            "prototype_vote_weights": np.asarray(
                [
                    [0.10, 0.20, 0.30, 0.40],
                    [0.25, 0.25, 0.10, 0.40],
                ],
                dtype=np.float32,
            ),
            "prototype_subjects": np.asarray([1, 1, 3, 3], dtype=np.int32),
            "train_subjects": np.asarray([1, 3], dtype=np.int32),
        }

        row = FewShotPainLearner._aggregate_source_subject_prototype_vote_weights(
            diagnostics=diagnostics,
            test_subject=2,
        )

        self.assertEqual(list(row), [1, 2, 3])
        self.assertAlmostEqual(row[1], 0.4)
        self.assertEqual(row[2], 0.0)
        self.assertAlmostEqual(row[3], 0.6)
        self.assertAlmostEqual(sum(row.values()), 1.0)

    def test_source_subject_prototype_vote_weight_writer_records_one_fold_matrix(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        recorder = CrossValidationResultRecorder(
            heldout_eval_pairs=[(1, 1)],
            training_progress_output_dir=tmp.name,
            csv_flush_every_events=1,
            validation_checkpoint_metric="accuracy",
            validation_checkpoint_mode="max",
            logger=logging.getLogger("test_source_subject_vote_weights"),
        )
        progress_file = recorder.start_fold(fold_idx=1, test_subject=2)
        learner = FewShotPainLearner.__new__(FewShotPainLearner)
        diagnostics = {
            "prototype_vote_weights": np.asarray(
                [
                    [0.10, 0.20, 0.30, 0.40],
                    [0.25, 0.25, 0.10, 0.40],
                ],
                dtype=np.float32,
            ),
            "prototype_subjects": np.asarray([1, 1, 3, 3], dtype=np.int32),
            "train_subjects": np.asarray([1, 3], dtype=np.int32),
        }

        path = learner._write_source_subject_prototype_vote_weights(
            progress_file=progress_file,
            test_subject=2,
            diagnostics=diagnostics,
        )
        recorder.record_source_subject_prototype_vote_weight_file(path)
        recorder.close_fold()

        self.assertTrue(path.endswith("_source_subject_prototype_vote_weights.csv"))
        self.assertEqual(
            recorder.results["source_subject_prototype_vote_weight_files"],
            [path],
        )
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(list(rows[0]), ["subject_1", "subject_2", "subject_3"])
        self.assertAlmostEqual(float(rows[0]["subject_1"]), 0.4)
        self.assertEqual(float(rows[0]["subject_2"]), 0.0)
        self.assertAlmostEqual(float(rows[0]["subject_3"]), 0.6)

    def test_prototype_bank_initializer_zero_samples_leaves_memory_unchanged(self):
        learner, train_sampler = _make_initializer_learner(samples_per_slot=0)
        before = learner.model.prototype_memory.assigned.copy()

        metadata = learner._initialize_prototype_bank_from_training_samples(
            fold=0,
            test_subject=3,
            train_sampler=train_sampler,
        )

        self.assertFalse(metadata["enabled"])
        np.testing.assert_allclose(learner.model.prototype_memory.assigned, before)

    def test_prototype_bank_initializer_rejects_insufficient_nonoverlap_samples(self):
        learner, train_sampler = _make_initializer_learner(
            samples_per_slot=5,
            slots_per_class=2,
            samples_per_subject_class=3,
        )

        with self.assertRaisesRegex(ValueError, "Insufficient training samples"):
            learner._initialize_prototype_bank_from_training_samples(
                fold=0,
                test_subject=3,
                train_sampler=train_sampler,
            )

    def test_cv_result_recorder_records_standard_diagnostics(self):
        recorder = self._make_recorder()
        metrics = {
            "task_loss": 0.4,
            "contrastive_loss": 0.1,
            "triplet_loss": 0.2,
            "can_local_loss": 0.05,
            "can_global_loss": 0.03,
            "accuracy": 0.75,
            "precision": 0.7,
            "recall": 0.8,
            "f1": 0.74,
            "intra_class_similarity": 0.9,
            "inter_class_similarity": 0.2,
        }

        recorder.record_heldout_size_result(
            size_key="k2_q3",
            zero_shot_loss=0.5,
            zero_shot_metrics=metrics,
            adaptation_losses=[0.3, 0.2],
            k_shot_loss=0.4,
            k_shot_metrics=metrics,
            zero_shot_task_batch=[{"id": "zero"}],
            k_shot_task_batch=[{"id": "k"}],
        )

        bucket = recorder.results["heldout_eval_by_task_size"]["k2_q3"]
        self.assertEqual(bucket["zero_shot_accuracies"], [0.75])
        self.assertEqual(bucket["zero_shot_intra_class_similarities"], [0.9])
        self.assertEqual(bucket["k_shot_f1s"], [0.74])

    def test_cv_result_recorder_metric_kwargs_leave_non_can_margin_blank(self):
        kwargs = CrossValidationResultRecorder._metric_event_kwargs(
            {
                "task_loss": 0.4,
                "contrastive_loss": 0.1,
                "triplet_loss": 0.2,
                "can_local_loss": 0.05,
                "can_global_loss": 0.03,
                "can_margin_loss": 0.99,
                "accuracy": 0.75,
                "precision": 0.7,
                "recall": 0.8,
                "f1": 0.74,
                "intra_class_similarity": 0.9,
                "inter_class_similarity": 0.2,
                "similarity_margin": 0.7,
            },
            include_similarity_margin=True,
        )

        self.assertEqual(kwargs["contrastive_loss"], 0.1)
        self.assertEqual(kwargs["can_global_loss"], 0.03)
        self.assertIsNone(kwargs["can_margin_loss"])
        self.assertEqual(kwargs["similarity_margin"], 0.7)

    def test_cv_result_recorder_writes_standard_metrics_to_progress_csv(self):
        recorder = self._make_recorder()
        progress_file = recorder.start_fold(fold_idx=1, test_subject=2)
        recorder.write_metric_event(
            fold_idx=1,
            test_subject=2,
            event_type="k_shot_summary_k2_q3",
            loss=0.5,
            metrics={
                "task_loss": 0.4,
                "contrastive_loss": 0.1,
                "triplet_loss": 0.2,
                "can_local_loss": 0.05,
                "can_global_loss": 0.03,
                "accuracy": 0.75,
                "precision": 0.7,
                "recall": 0.8,
                "f1": 0.74,
                "intra_class_similarity": 0.9,
                "inter_class_similarity": 0.2,
            },
        )
        recorder.close_fold()

        with open(progress_file, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["event_type"], "k_shot_summary_k2_q3")
        self.assertEqual(rows[0]["accuracy"], "0.75")
        self.assertEqual(rows[0]["precision"], "0.7")
        self.assertEqual(rows[0]["recall"], "0.8")
        self.assertEqual(rows[0]["f1"], "0.74")

    def test_task_batch_pipeline_stacks_and_prefetches_batch_sizes(self):
        tasks = [
            {
                "support_X": np.full((2, 3, 1), fill_value=idx, dtype=np.float32),
                "support_y": np.array([0, 1], dtype=np.int32),
                "query_X": np.full((2, 3, 1), fill_value=idx + 10, dtype=np.float32),
                "query_y": np.array([0, 1], dtype=np.int32),
            }
            for idx in range(3)
        ]
        support_x, support_y, query_x, query_y = TaskBatchPipeline.stack_task_batch(
            tasks
        )
        self.assertEqual(tuple(support_x.shape), (3, 2, 3, 1))
        self.assertEqual(tuple(support_y.shape), (3, 2))
        self.assertEqual(tuple(query_x.shape), (3, 2, 3, 1))
        self.assertEqual(tuple(query_y.shape), (3, 2))

        class Sampler:
            def __init__(self):
                self.next_idx = 0

            def get_task(self):
                task = tasks[self.next_idx % len(tasks)]
                self.next_idx += 1
                return task

        pipeline = TaskBatchPipeline(
            train_batch_size=2,
            embedding_batch_size=1,
            train_prefetch_batches=2,
        )
        batches = list(pipeline.iter_prefetched_task_batches(Sampler(), 5))
        self.assertEqual([batch_size for batch_size, _ in batches], [2, 2, 1])
        self.assertTrue(
            all(arrays[0].shape[0] == batch_size for batch_size, arrays in batches)
        )

    def test_phase2_prototype_updates_use_configured_task_batches(self):
        learner = FewShotPainLearner.__new__(FewShotPainLearner)
        learner.train_batch_size = 3
        learner.task_pipeline = TaskBatchPipeline(
            train_batch_size=3,
            embedding_batch_size=1,
            train_prefetch_batches=1,
        )
        sampler = _CountingPhase2Sampler()

        batches = list(
            learner._iter_prototype_finetune_task_batches(
                sampler,
                prototype_updates_per_epoch=2,
            )
        )

        self.assertEqual(sampler.calls, 6)
        self.assertEqual([batch_size for batch_size, _ in batches], [3, 3])
        for batch_size, (support_x, support_y, query_x, query_y) in batches:
            self.assertEqual(batch_size, 3)
            self.assertEqual(support_x.shape, (3, 4, 5, 2))
            self.assertEqual(support_y.shape, (3, 4))
            self.assertEqual(query_x.shape, (3, 6, 5, 2))
            self.assertEqual(query_y.shape, (3, 6))

    def test_episode_evaluation_macro_and_similarity_metrics(self):
        evaluator = EpisodeEvaluationService(
            config=SimpleNamespace(n_way=3),
            engine=None,
            task_pipeline=None,
        )
        macro = evaluator.compute_macro_metrics(
            np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
            np.array([0, 1, 1, 1, 2, 0], dtype=np.int32),
        )
        self.assertAlmostEqual(macro["accuracy"], 4 / 6)
        self.assertTrue({"precision", "recall", "f1"}.issubset(macro))

        similarity = evaluator.compute_similarity_metrics(
            np.array([0.8, 0.7], dtype=np.float32),
            np.array([0.2, 0.4], dtype=np.float32),
        )
        self.assertAlmostEqual(similarity["similarity_margin"], 0.45, places=6)

    def test_episode_evaluation_can_support_mode_override_restores_original(self):
        engine = _FakeEvaluationEngine()
        evaluator = EpisodeEvaluationService(
            config=SimpleNamespace(
                n_way=2,
                attention_mode="can",
            ),
            engine=engine,
            task_pipeline=SimpleNamespace(
                task_batch_has_uniform_shapes=lambda task_batch: False
            ),
        )
        task = {
            "support_X": np.zeros((2, 3, 1), dtype=np.float32),
            "support_y": np.array([0, 1], dtype=np.int32),
            "query_X": np.zeros((2, 3, 1), dtype=np.float32),
            "query_y": np.array([0, 1], dtype=np.int32),
        }

        loss, metrics = evaluator.evaluate_task_batch_loss_and_metrics(
            [task],
            can_support_mode="sampled",
        )

        self.assertEqual(engine.seen_support_modes, ["sampled"])
        self.assertEqual(engine.model.can_support_mode, "learned_prototype_memory")
        self.assertEqual(loss, 0.25)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_episode_evaluation_collects_compact_can_feature_export(self):
        engine = _FakeFeatureExportEngine()
        evaluator = EpisodeEvaluationService(
            config=SimpleNamespace(n_way=2, attention_mode="can"),
            engine=engine,
            task_pipeline=None,
        )
        task = {
            "support_X": np.zeros((2, 3, 1), dtype=np.float32),
            "support_y": np.array([0, 1], dtype=np.int32),
            "query_X": np.zeros((2, 3, 1), dtype=np.float32),
            "query_y": np.array([0, 1], dtype=np.int32),
        }

        export = evaluator.collect_can_feature_export(
            [task],
            phase="k_shot",
            can_support_mode="sampled",
        )

        self.assertEqual(engine.model.can_support_mode, "learned_prototype_memory")
        self.assertEqual(engine.triplet_loss_weight, 1.0)
        self.assertEqual(export["query_features"].shape, (2, 2))
        self.assertEqual(export["support_features"].shape, (2, 2))
        self.assertEqual(export["prototype_features"].shape, (2, 2))
        self.assertTrue(np.array_equal(export["query_y"], np.array([0, 1])))
        self.assertTrue(np.array_equal(export["query_pred"], np.array([0, 1])))
        self.assertEqual(str(export["phase"]), "k_shot")
        self.assertEqual(str(export["can_support_mode"]), "sampled")

    def test_learned_prototype_holdout_sweep_writes_sampled_support_rows(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pairs = [(1, 1), (5, 5), (10, 10)]
        recorder = CrossValidationResultRecorder(
            heldout_eval_pairs=pairs,
            training_progress_output_dir=tmp.name,
            csv_flush_every_events=1,
            validation_checkpoint_metric="f1",
            validation_checkpoint_mode="max",
            logger=logging.getLogger("test_learned_prototype_sweep"),
        )
        progress_file = recorder.start_fold(fold_idx=1, test_subject=7)

        learner = FewShotPainLearner.__new__(FewShotPainLearner)
        learner.train_batch_size = 2
        learner.logger = logging.getLogger("test_learned_prototype_sweep")
        reference_calls = []

        def fake_reference(**kwargs):
            reference_calls.append(kwargs["label"])
            return (
                _FakePrototypeDataset().build_all_query_task(
                    subject=kwargs["test_subject"],
                    split=kwargs["test_sampler"].data_split,
                    use_base_index=True,
                    normalize_with_query_subject_stats=True,
                ),
                0.6,
                {
                    "task_loss": 0.6,
                    "contrastive_loss": 0.0,
                    "triplet_loss": 0.0,
                    "can_local_loss": 0.1,
                    "can_global_loss": 0.2,
                    "accuracy": 0.5,
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1": 0.5,
                    "intra_class_similarity": 0.4,
                    "inter_class_similarity": 0.2,
                },
            )

        learner._evaluate_learned_prototype_bank_reference = fake_reference
        support_modes = []

        def fake_eval(task_batch, forward_batch_size=None, can_support_mode=None):
            del forward_batch_size
            support_modes.append(can_support_mode)
            q_query = int(len(task_batch[0]["query_y"]) / 2)
            return (
                float(q_query),
                {
                    "task_loss": float(q_query),
                    "contrastive_loss": 0.0,
                    "triplet_loss": 0.0,
                    "can_local_loss": 0.1,
                    "can_global_loss": 0.2,
                    "accuracy": 0.8,
                    "precision": 0.8,
                    "recall": 0.8,
                    "f1": 0.8,
                    "intra_class_similarity": 0.6,
                    "inter_class_similarity": 0.1,
                },
            )

        learner._evaluate_task_batch_loss_and_metrics = fake_eval
        learner._set_sampler_task_size = EpisodeEvaluationService.set_sampler_task_size
        sampler = _FakeSampler()
        sampler.data_split = "test"

        sweep = learner._evaluate_learned_prototype_holdout_sweep(
            fold=0,
            num_subjects=1,
            test_subject=7,
            test_sampler=sampler,
            heldout_eval_pairs=pairs,
            configured_eval_pair=(1, 1),
            heldout_eval_tasks=1,
            result_recorder=recorder,
        )
        recorder.close_fold()

        with open(progress_file, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        events = [row["event_type"] for row in rows]

        self.assertEqual(
            events,
            ["k_shot_summary_k1_q1", "k_shot_summary_k5_q5", "k_shot_summary_k10_q10"],
        )
        self.assertEqual(reference_calls, ["Post-phase-2"])
        self.assertEqual(support_modes, ["sampled", "sampled", "sampled"])
        for row in rows:
            self.assertEqual(row["accuracy"], "0.8")
            self.assertEqual(row["f1"], "0.8")
        self.assertEqual(set(sweep), {"k1_q1", "k5_q5", "k10_q10"})
        self.assertEqual(
            recorder.results["heldout_eval_by_task_size"]["k5_q5"][
                "zero_shot_accuracies"
            ],
            [0.5],
        )
        self.assertEqual(
            recorder.results["heldout_eval_by_task_size"]["k5_q5"][
                "k_shot_accuracies"
            ],
            [0.8],
        )

    def test_phase2_initial_prototype_bank_evaluation_writes_progress_row(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        recorder = CrossValidationResultRecorder(
            heldout_eval_pairs=[(1, 1)],
            training_progress_output_dir=tmp.name,
            csv_flush_every_events=1,
            validation_checkpoint_metric="f1",
            validation_checkpoint_mode="max",
            logger=logging.getLogger("test_phase2_initial_prototype_bank"),
        )
        progress_file = recorder.start_fold(fold_idx=1, test_subject=7)

        learner = FewShotPainLearner.__new__(FewShotPainLearner)
        learner.dataset = _FakePrototypeDataset()
        learner.logger = logging.getLogger("test_phase2_initial_prototype_bank")
        reference_calls = []
        learner.evaluator = SimpleNamespace(
            evaluate_prototype_memory_task_metrics=lambda task: (
                reference_calls.append(len(task["query_y"]))
                or (
                    0.7,
                    {
                        "task_loss": 0.7,
                        "contrastive_loss": 0.0,
                        "triplet_loss": 0.0,
                        "can_local_loss": 0.1,
                        "can_global_loss": 0.2,
                        "accuracy": 0.45,
                        "precision": 0.46,
                        "recall": 0.44,
                        "f1": 0.45,
                        "intra_class_similarity": 0.3,
                        "inter_class_similarity": 0.2,
                    },
                )
            )
        )
        sampler = _FakeSampler()
        sampler.data_split = "test"

        learner._write_phase2_initial_prototype_bank_evaluation(
            fold=0,
            num_subjects=1,
            test_subject=7,
            test_sampler=sampler,
            result_recorder=recorder,
        )
        recorder.close_fold()

        with open(progress_file, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(reference_calls, [4])
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["event_type"],
            "prototype_bank_phase2_initial_summary",
        )
        self.assertEqual(rows[0]["loss"], "0.7")
        self.assertEqual(rows[0]["accuracy"], "0.45")
        self.assertEqual(rows[0]["f1"], "0.45")

    def test_phase2_initial_sampled_support_evaluation_writes_progress_row_and_restores_rng(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        recorder = CrossValidationResultRecorder(
            heldout_eval_pairs=[(10, 10)],
            training_progress_output_dir=tmp.name,
            csv_flush_every_events=1,
            validation_checkpoint_metric="f1",
            validation_checkpoint_mode="max",
            logger=logging.getLogger("test_phase2_initial_sampled_support"),
        )
        progress_file = recorder.start_fold(fold_idx=1, test_subject=7)

        learner = FewShotPainLearner.__new__(FewShotPainLearner)
        learner.train_batch_size = 2
        learner.logger = logging.getLogger("test_phase2_initial_sampled_support")
        learner._set_sampler_task_size = EpisodeEvaluationService.set_sampler_task_size
        support_modes = []
        query_sizes = []

        def fake_eval(task_batch, forward_batch_size=None, can_support_mode=None):
            del forward_batch_size
            support_modes.append(can_support_mode)
            query_sizes.append(len(task_batch[0]["query_y"]))
            return (
                0.4,
                {
                    "task_loss": 0.4,
                    "contrastive_loss": 0.0,
                    "triplet_loss": 0.0,
                    "can_local_loss": 0.1,
                    "can_global_loss": 0.2,
                    "accuracy": 0.83,
                    "precision": 0.84,
                    "recall": 0.82,
                    "f1": 0.83,
                    "intra_class_similarity": 0.6,
                    "inter_class_similarity": 0.1,
                },
            )

        learner._evaluate_task_batch_loss_and_metrics = fake_eval
        sampler = _FakeSampler()
        sampler.data_split = "test"
        initial_rng_state = sampler.rng.bit_generator.state

        learner._write_phase2_initial_sampled_support_evaluation(
            fold=0,
            num_subjects=1,
            test_subject=7,
            test_sampler=sampler,
            configured_eval_pair=(10, 10),
            heldout_eval_tasks=1,
            result_recorder=recorder,
        )
        recorder.close_fold()

        with open(progress_file, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(support_modes, ["sampled"])
        self.assertEqual(query_sizes, [20])
        self.assertEqual(sampler.rng.bit_generator.state, initial_rng_state)
        self.assertEqual((sampler.k_shot, sampler.q_query), (1, 1))
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["event_type"],
            "support_samples_phase2_initial_summary_k10_q10",
        )
        self.assertEqual(rows[0]["accuracy"], "0.83")
        self.assertEqual(rows[0]["f1"], "0.83")

    def test_heldout_adaptation_restores_task_size_after_success_and_failure(self):
        evaluator = EpisodeEvaluationService(
            config=SimpleNamespace(n_way=2),
            engine=None,
            task_pipeline=None,
        )
        sampler = _FakeSampler()
        service = HeldoutAdaptationService(
            engine=_FakeEngine(),
            evaluator=evaluator,
        )
        losses = service.adapt_on_sampler_at_task_size(
            sampler,
            adaptation_steps=2,
            k_shot=3,
            q_query=4,
        )
        self.assertEqual(losses, [1.0, 2.0])
        self.assertEqual((sampler.k_shot, sampler.q_query), (1, 1))
        self.assertEqual((sampler.support_size, sampler.query_size), (2, 2))

        failing_service = HeldoutAdaptationService(
            engine=_FakeEngine(fail_after=0),
            evaluator=evaluator,
        )
        with self.assertRaises(RuntimeError):
            failing_service.adapt_on_sampler_at_task_size(
                sampler,
                adaptation_steps=1,
                k_shot=5,
                q_query=6,
            )
        self.assertEqual((sampler.k_shot, sampler.q_query), (1, 1))
        self.assertEqual((sampler.support_size, sampler.query_size), (2, 2))

    def test_validation_sampler_maps_oversized_raw_tasks_to_ten_shot_ten_query(self):
        dataset = _FakeDatasetForSampler()
        sampler = SixWayKShotSampler(
            dataset=dataset,
            mode="val",
            train_subjects=[7],
            seed=123,
        )

        self.assertEqual((sampler.k_shot, sampler.q_query), (10, 10))
        self.assertEqual((sampler.support_size, sampler.query_size), (20, 20))

        task = sampler.get_task()

        self.assertTrue(dataset.last_use_base_index)
        self.assertEqual(task["support_X"].shape[0], 20)
        self.assertEqual(task["query_X"].shape[0], 20)

        test_sampler = SixWayKShotSampler(
            dataset=dataset,
            mode="test",
            train_subjects=[7],
            test_subject=7,
            seed=123,
        )
        self.assertEqual((test_sampler.k_shot, test_sampler.q_query), (10, 10))
        self.assertEqual((test_sampler.support_size, test_sampler.query_size), (20, 20))


if __name__ == "__main__":
    unittest.main()
