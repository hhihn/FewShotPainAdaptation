import tensorflow as tf


class HeldoutAdaptationService:
    """Adapt model weights on held-out tasks with temporary task sizes.

    The service owns the sampler k/q override lifecycle used during held-out
    adaptation sweeps.
    """

    def __init__(self, *, engine, evaluator):
        """Initialize the held-out adaptation service.

        Args:
            engine: Episodic learning engine that performs train steps.
            evaluator: Evaluation service that can modify sampler task sizes.
        """
        self.engine = engine
        self.evaluator = evaluator

    def adapt_on_sampler_at_task_size(
        self,
        sampler,
        *,
        adaptation_steps: int,
        k_shot: int,
        q_query: int,
    ) -> list[float]:
        """Run adaptation with a temporary k-shot/q-query override.

        Args:
            sampler: Held-out episodic sampler.
            adaptation_steps: Number of adaptation train steps to run.
            k_shot: Temporary support samples per class.
            q_query: Temporary query samples per class.
        """
        original_k = int(sampler.k_shot)
        original_q = int(sampler.q_query)
        self.evaluator.set_sampler_task_size(sampler, k_shot=k_shot, q_query=q_query)
        adaptation_losses = []
        try:
            for _ in range(max(0, int(adaptation_steps))):
                adapt_task = sampler.get_task()
                support_x = tf.constant(adapt_task["support_X"], dtype=tf.float32)
                support_y = tf.constant(adapt_task["support_y"], dtype=tf.int32)
                query_x = tf.constant(adapt_task["query_X"], dtype=tf.float32)
                query_y = tf.constant(adapt_task["query_y"], dtype=tf.int32)
                adapt_loss, _ = self.engine.train_step(
                    support_x,
                    support_y,
                    query_x,
                    query_y,
                )
                adaptation_losses.append(float(adapt_loss))
            return adaptation_losses
        finally:
            self.evaluator.set_sampler_task_size(
                sampler,
                k_shot=original_k,
                q_query=original_q,
            )
