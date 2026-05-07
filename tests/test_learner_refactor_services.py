import unittest
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from learner.episode_evaluation_service import EpisodeEvaluationService
from learner.heldout_adaptation_service import HeldoutAdaptationService
from learner.task_batch_pipeline import TaskBatchPipeline
from data_loaders.meta_ds_sampler import SixWayKShotSampler


class _FakeSampler:
    def __init__(self):
        self.n_way = 2
        self.k_shot = 1
        self.q_query = 1
        self.support_size = 2
        self.query_size = 2

    def get_task(self):
        return {
            "support_X": np.zeros((self.support_size, 3, 1), dtype=np.float32),
            "support_y": np.repeat(np.arange(self.n_way), self.k_shot).astype(
                np.int32
            ),
            "query_X": np.zeros((self.query_size, 3, 1), dtype=np.float32),
            "query_y": np.repeat(np.arange(self.n_way), self.q_query).astype(
                np.int32
            ),
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


class LearnerRefactorServiceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
