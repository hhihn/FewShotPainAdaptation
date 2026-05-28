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
        ):
            self.assertIn(key, results)
        self.assertIn("zero_shot_accuracies", size_bucket)
        self.assertIn("k_shot_accuracies", size_bucket)
        self.assertEqual(results["validation_checkpoint_metric"], "f1")
        self.assertEqual(results["validation_checkpoint_mode"], "max")

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
