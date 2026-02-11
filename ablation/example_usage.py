import numpy as np
from pathlib import Path

import logging

from utils.logger import setup_logger

from data_loaders.pain_meta_dataset import PainMetaDataset
from data_loaders.pain_meta_dataset import PainDatasetConfig
from data_loaders.loso_cross_validator import LOSOCrossValidator

logger = setup_logger("PainMetaExampleUsage", level=logging.DEBUG)


def example_usage():
    """Demonstrate the sampler usage."""

    logger.info("=" * 60)
    logger.info("6-Way-K-Shot Sampler for Pain Dataset")
    logger.info("=" * 60)

    # Configuration
    config = PainDatasetConfig()

    np.random.seed(42)

    # Create synthetic data
    data_dir = Path("../data")

    # Create dataset
    logger.info("Loading dataset...")
    dataset = PainMetaDataset(
        data_dir=str(data_dir),
        config=config,
        normalize=True,
        normalize_per_subject=True,
    )

    # Test basic episode sampling
    logger.info("-" * 40)
    logger.info("Testing episode sampling...")
    logger.info("-" * 40)

    episode = dataset.sample_episode(subject=0, k_shot=3, q_query=3)
    logger.info(f"Support X shape: {episode['support_X'].shape}")
    logger.info(f"Support y shape: {episode['support_y'].shape}")
    logger.info(f"Query X shape: {episode['query_X'].shape}")
    logger.info(f"Query y shape: {episode['query_y'].shape}")
    logger.info(
        f"Support y distribution: {np.bincount(episode['support_y'], minlength=6)}"
    )
    logger.info(f"Query y distribution: {np.bincount(episode['query_y'], minlength=6)}")

    # Test LOSO cross-validation
    logger.info("-" * 40)
    logger.info("Testing LOSO Cross-Validation...")
    logger.info("-" * 40)

    cv = LOSOCrossValidator(
        dataset=dataset,
        k_shot=config.k_shot,
        q_query=config.q_query,
        episodes_per_epoch=10,
    )

    logger.info(f"Number of folds: {len(cv)}")

    # Get first fold
    fold = cv.get_fold(test_subject=0)
    logger.info(f"Fold for test subject 0:")
    logger.info(f"Training subjects: {len(fold['train_subjects'])}")
    logger.info(f"Validation subjects: {len(fold['val_subjects'])}")
    logger.info(f"Test subject: {fold['test_subject']}")

    # Test the sampler
    logger.info("-" * 40)
    logger.info("Testing Sampler iteration...")
    logger.info("-" * 40)

    train_sampler = fold["train_sampler"]
    for i, episode in enumerate(train_sampler):
        if i >= 3:
            break
        logger.info(
            f"Episode {i}: "
            f"support={episode['support_X'].shape}, "
            f"query={episode['query_X'].shape}"
        )

    # Test few-shot split
    logger.info("-" * 40)
    logger.info("Testing Few-Shot Split...")
    logger.info("-" * 40)

    for k in [1, 3, 5]:
        support, eval_set = dataset.get_few_shot_split(subject=0, k_shot=k)
        logger.info(
            f"{k}-shot: Support={support['X'].shape}, Eval={eval_set['X'].shape}"
        )

    # Clean up

    logger.info("=" * 60)
    logger.info("All tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    example_usage()
