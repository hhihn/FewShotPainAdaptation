import tensorflow as tf
from data_loaders.pain_ds_config import PainDatasetConfig
from learner.few_shot_pain_learner import FewShotPainLearner
from utils.logger import setup_logger

logger = setup_logger("FewShotPainLearner")


def main():
    """Example usage of the few-shot pain learner."""
    config = PainDatasetConfig()
    print(tf.config.list_physical_devices())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("MPS")))

    logger.info("=" * 60)
    logger.info("Multimodal Few-Shot Learning for Personalized Pain Assessment")
    logger.info("=" * 60)

    # Try different fusion methods
    fusion_methods = ["attention"]

    for fusion_method in fusion_methods:
        logger.info(f"\nTraining with fusion method: {fusion_method}")

        learner = FewShotPainLearner(
            config=config,
            data_dir="../data",
            learning_rate=1e-3,
            fusion_method=fusion_method,
            seed=config.seed,
            deterministic_ops=config.deterministic_ops,
        )

        cv_results = learner.train(
            num_epochs=100, tasks_per_epoch=50, val_tasks=10
        )
        logger.info(cv_results)
        logger.info(f"Training with {fusion_method} complete!")


if __name__ == "__main__":
    main()
