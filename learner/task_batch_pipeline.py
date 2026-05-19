from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf


class TaskBatchPipeline:
    """Task sampling, stacking, prefetching, and tensor chunking utilities."""

    def __init__(
        self,
        *,
        train_batch_size: int,
        embedding_batch_size: int,
        train_prefetch_batches: int,
    ):
        self.train_batch_size = max(1, int(train_batch_size))
        self.embedding_batch_size = max(1, int(embedding_batch_size))
        self.train_prefetch_batches = max(1, int(train_prefetch_batches))

    @staticmethod
    def stack_task_batch_numpy(
        task_batch: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pack a Python task list into dense NumPy arrays once per update."""
        support_x_np = np.stack(
            [task_dict["support_X"] for task_dict in task_batch], axis=0
        )
        support_y_np = np.stack(
            [task_dict["support_y"] for task_dict in task_batch], axis=0
        )
        query_x_np = np.stack(
            [task_dict["query_X"] for task_dict in task_batch], axis=0
        )
        query_y_np = np.stack(
            [task_dict["query_y"] for task_dict in task_batch], axis=0
        )
        return support_x_np, support_y_np, query_x_np, query_y_np

    @classmethod
    def stack_task_batch(
        cls,
        task_batch: list[dict],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Pack a Python task list into dense batch tensors once per update."""
        support_x_np, support_y_np, query_x_np, query_y_np = cls.stack_task_batch_numpy(
            task_batch
        )
        return (
            tf.convert_to_tensor(support_x_np, dtype=tf.float32),
            tf.convert_to_tensor(support_y_np, dtype=tf.int32),
            tf.convert_to_tensor(query_x_np, dtype=tf.float32),
            tf.convert_to_tensor(query_y_np, dtype=tf.int32),
        )

    @classmethod
    def sample_and_stack_task_batch_numpy(
        cls,
        sampler,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample one task batch and pack it into dense NumPy arrays."""
        task_batch = [sampler.get_task() for _ in range(max(1, int(batch_size)))]
        return cls.stack_task_batch_numpy(task_batch)

    def iter_prefetched_task_batches(
        self,
        sampler,
        tasks_per_epoch: int,
    ):
        """Yield `(batch_size, stacked_numpy_batch)` with async CPU prefetch."""
        batch_sizes = [
            min(self.train_batch_size, tasks_per_epoch - task_start)
            for task_start in range(0, tasks_per_epoch, self.train_batch_size)
        ]
        if not batch_sizes:
            return

        prefetch_batches = max(1, int(self.train_prefetch_batches))
        if prefetch_batches <= 1:
            for batch_size in batch_sizes:
                yield (
                    batch_size,
                    self.sample_and_stack_task_batch_numpy(
                        sampler,
                        batch_size,
                    ),
                )
            return

        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = deque()
            next_batch_idx = 0

            while next_batch_idx < len(batch_sizes) and len(pending) < prefetch_batches:
                batch_size = batch_sizes[next_batch_idx]
                pending.append(
                    (
                        batch_size,
                        executor.submit(
                            self.sample_and_stack_task_batch_numpy,
                            sampler,
                            batch_size,
                        ),
                    )
                )
                next_batch_idx += 1

            while pending:
                batch_size, batch_future = pending.popleft()
                batch_arrays = batch_future.result()

                if next_batch_idx < len(batch_sizes):
                    next_size = batch_sizes[next_batch_idx]
                    pending.append(
                        (
                            next_size,
                            executor.submit(
                                self.sample_and_stack_task_batch_numpy,
                                sampler,
                                next_size,
                            ),
                        )
                    )
                    next_batch_idx += 1

                yield batch_size, batch_arrays

    def iter_task_tensor_chunks(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Yield task tensor chunks sized by embedding_batch_size in eager mode."""
        total_tasks = int(tf.shape(support_x_batch)[0].numpy())
        if total_tasks <= 0:
            raise ValueError("task tensor batch must contain at least one task")
        chunk_size = min(max(1, int(self.embedding_batch_size)), total_tasks)
        for task_start in range(0, total_tasks, chunk_size):
            task_end = min(total_tasks, task_start + chunk_size)
            yield (
                support_x_batch[task_start:task_end],
                support_y_batch[task_start:task_end],
                query_x_batch[task_start:task_end],
                query_y_batch[task_start:task_end],
            )

    @staticmethod
    def task_batch_has_uniform_shapes(task_batch: list[dict]) -> bool:
        """Return True when support/query tensors share identical shapes across tasks."""
        if not task_batch:
            return False
        keys = ("support_X", "support_y", "query_X", "query_y")
        reference_shapes = {
            key: tuple(np.asarray(task_batch[0][key]).shape) for key in keys
        }
        for task_dict in task_batch[1:]:
            for key in keys:
                if tuple(np.asarray(task_dict[key]).shape) != reference_shapes[key]:
                    return False
        return True
