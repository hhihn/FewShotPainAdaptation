import numpy as np
from pathlib import Path


from utils.logger import setup_logger

from data_loaders.pain_meta_dataset import PainMetaDataset
from data_loaders.pain_meta_dataset import PainDatasetConfig
from data_loaders.loso_cross_validator import LOSOCrossValidator

logger = setup_logger("PainMetaExampleUsage")


def example_usage():
    """Demonstrate the sampler usage."""

    logger.debug("=" * 60)
    logger.debug("6-Way-K-Shot Sampler for Pain Dataset")
    logger.debug("=" * 60)

    # Configuration
    config = PainDatasetConfig()

    np.random.seed(42)

    # Create synthetic data
    data_dir = Path("../data")

    # Create dataset
    logger.debug("Loading dataset...")
    dataset = PainMetaDataset(
        data_dir=str(data_dir),
        config=config,
        normalize=True,
        normalize_per_subject=True,
    )

    # Test basic episode sampling
    logger.debug("-" * 40)
    logger.debug("Testing episode sampling...")
    logger.debug("-" * 40)

    episode = dataset.sample_episode(subject=0, k_shot=3, q_query=3)
    logger.debug(f"Support X shape: {episode['support_X'].shape}")
    logger.debug(f"Support y shape: {episode['support_y'].shape}")
    logger.debug(f"Query X shape: {episode['query_X'].shape}")
    logger.debug(f"Query y shape: {episode['query_y'].shape}")
    logger.debug(
        f"Support y distribution: {np.bincount(episode['support_y'], minlength=6)}"
    )
    logger.debug(
        f"Query y distribution: {np.bincount(episode['query_y'], minlength=6)}"
    )

    # Test LOSO cross-validation
    logger.debug("-" * 40)
    logger.debug("Testing LOSO Cross-Validation...")
    logger.debug("-" * 40)

    cv = LOSOCrossValidator(
        dataset=dataset,
        k_shot=config.k_shot,
        q_query=config.q_query,
        episodes_per_epoch=10,
    )

    logger.debug(f"Number of folds: {len(cv)}")

    # Get first fold
    fold = cv.get_fold(test_subject=0)
    logger.debug("Fold for test subject 0:")
    logger.debug(f"Training subjects: {len(fold['train_subjects'])}")
    logger.debug(f"Validation subjects: {len(fold['val_subjects'])}")
    logger.debug(f"Test subject: {fold['test_subject']}")

    # Test the sampler
    logger.debug("-" * 40)
    logger.debug("Testing Sampler iteration...")
    logger.debug("-" * 40)

    train_sampler = fold["train_sampler"]
    for i, episode in enumerate(train_sampler):
        if i >= 3:
            break
        logger.debug(
            f"Episode {i}: "
            f"support={episode['support_X'].shape}, "
            f"query={episode['query_X'].shape}"
        )

    # Test few-shot split
    logger.debug("-" * 40)
    logger.debug("Testing Few-Shot Split...")
    logger.debug("-" * 40)

    for k in [1, 3, 5]:
        support, eval_set = dataset.get_few_shot_split(subject=0, k_shot=k)
        logger.debug(
            f"{k}-shot: Support={support['X'].shape}, Eval={eval_set['X'].shape}"
        )

    # Clean up

    logger.debug("=" * 60)
    logger.debug("All tests passed!")
    logger.debug("=" * 60)


if __name__ == "__main__":
    example_usage()
