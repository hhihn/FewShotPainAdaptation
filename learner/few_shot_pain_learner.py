import numpy as np
import tensorflow as tf
from tensorflow import keras
from data_loaders.pain_meta_dataset import PainMetaDataset
from data_loaders.loso_cross_validator import LOSOCrossValidator
from data_loaders.pain_ds_config import PainDatasetConfig
from architecture.mulitmodal_proto_net import MultimodalPrototypicalNetwork


class FewShotPainLearner:
    """Meta-learning trainer for personalized pain assessment."""

    def __init__(
        self,
        config: PainDatasetConfig,
        data_dir: str = "./dataset/np-dataset",
        learning_rate: float = 1e-3,
        fusion_method: str = "attention",
    ):
        """
        Args:
            config: PainDatasetConfig instance
            data_dir: Directory containing numpy files
            learning_rate: Outer loop learning rate
            fusion_method: 'concat', 'mean', or 'attention'
        """
        self.config = config
        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.fusion_method = fusion_method

        # Initialize dataset and cross-validator
        self.dataset = PainMetaDataset(
            data_dir=data_dir, config=config, normalize=True, normalize_per_subject=True
        )

        self.cv = LOSOCrossValidator(
            dataset=self.dataset, k_shot=config.k_shot, q_query=config.q_query
        )

        # Initialize model
        num_sensors = len(config.sensor_idx)
        self.model = MultimodalPrototypicalNetwork(
            sequence_length=config.sequence_length,
            num_sensors=num_sensors,
            num_classes=config.num_stimuli_levels,
            embedding_dim=64,
            modality_names=config.modality_names,
            fusion_method=fusion_method,
            distance_metric="euclidean",
        )

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        logger.info(
            f"Initialized FewShotPainLearner with {len(self.cv.subjects)} subjects"
        )
        logger.info(
            f"Data shape: (sequence_length={config.sequence_length}, num_sensors={num_sensors})"
        )
        logger.info(f"Modalities: {config.modality_names}")
        logger.info(f"Fusion method: {fusion_method}")

    def train_step(self, support_x, support_y, query_x, query_y):
        """Single training step on one episode."""
        with tf.GradientTape() as tape:
            logits = self.model(support_x, support_y, query_x, training=True)
            loss = self.loss_fn(query_y, logits)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Compute accuracy
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, tf.cast(query_y, tf.int64)), tf.float32)
        )

        return loss, accuracy

    def evaluate_episode(self, support_x, support_y, query_x, query_y):
        """Evaluate on one episode without updating weights."""
        logits = self.model(support_x, support_y, query_x, training=False)
        loss = self.loss_fn(query_y, logits)

        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, tf.cast(query_y, tf.int64)), tf.float32)
        )

        return loss, accuracy

    def train(
        self,
        num_epochs: int = 10,
        episodes_per_epoch: int = 100,
        val_episodes: int = 20,
    ):
        """
        Train on all subjects using leave-one-subject-out cross-validation.
        """
        cv_results = {
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
            "test_losses": [],
            "test_accuracies": [],
        }

        num_subjects = len(self.cv.subjects)

        for fold, test_subject in enumerate(self.cv.subjects):
            logger.info(f"\n{'=' * 60}")
            logger.info(
                f"Fold {fold + 1}/{num_subjects}: Test subject = {test_subject}"
            )
            logger.info(f"{'=' * 60}")

            # Reset model for each fold
            num_sensors = len(self.config.sensor_idx)
            self.model = MultimodalPrototypicalNetwork(
                sequence_length=self.config.sequence_length,
                num_sensors=num_sensors,
                num_classes=self.config.num_stimuli_levels,
                embedding_dim=64,
                modality_names=self.config.modality_names,
                fusion_method=self.fusion_method,
                distance_metric="euclidean",
            )
            self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

            # Get fold dictionary with samplers
            fold_dict = self.cv.get_fold(test_subject)

            train_sampler = fold_dict["train_sampler"]
            val_sampler = fold_dict["val_sampler"]
            test_sampler = fold_dict["test_sampler"]

            fold_results = {
                "train_losses": [],
                "train_accuracies": [],
                "val_losses": [],
                "val_accuracies": [],
            }

            for epoch in range(num_epochs):
                # Training
                epoch_train_losses = []
                epoch_train_accs = []

                for episode in range(episodes_per_epoch):
                    episode_dict = train_sampler.get_episode()

                    support_x = episode_dict["support_X"]  # [6*k_shot, 2500, 3]
                    support_y = episode_dict["support_y"]  # [6*k_shot]
                    query_x = episode_dict["query_X"]  # [6*q_query, 2500, 3]
                    query_y = episode_dict["query_y"]  # [6*q_query]

                    loss, acc = self.train_step(
                        tf.constant(support_x, dtype=tf.float32),
                        tf.constant(support_y, dtype=tf.int32),
                        tf.constant(query_x, dtype=tf.float32),
                        tf.constant(query_y, dtype=tf.int32),
                    )

                    epoch_train_losses.append(float(loss))
                    epoch_train_accs.append(float(acc))

                # Validation
                epoch_val_losses = []
                epoch_val_accs = []

                for _ in range(val_episodes):
                    episode_dict = val_sampler.get_episode()

                    support_x = episode_dict["support_X"]
                    support_y = episode_dict["support_y"]
                    query_x = episode_dict["query_X"]
                    query_y = episode_dict["query_y"]

                    loss, acc = self.evaluate_episode(
                        tf.constant(support_x, dtype=tf.float32),
                        tf.constant(support_y, dtype=tf.int32),
                        tf.constant(query_x, dtype=tf.float32),
                        tf.constant(query_y, dtype=tf.int32),
                    )

                    epoch_val_losses.append(float(loss))
                    epoch_val_accs.append(float(acc))

                avg_train_loss = np.mean(epoch_train_losses)
                avg_train_acc = np.mean(epoch_train_accs)
                avg_val_loss = np.mean(epoch_val_losses)
                avg_val_acc = np.mean(epoch_val_accs)

                fold_results["train_losses"].append(avg_train_loss)
                fold_results["train_accuracies"].append(avg_train_acc)
                fold_results["val_losses"].append(avg_val_loss)
                fold_results["val_accuracies"].append(avg_val_acc)

                if (epoch + 1) % 2 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{num_epochs} | "
                        f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | "
                        f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}"
                    )

            # Test on held-out subject
            test_losses = []
            test_accs = []

            for _ in range(val_episodes):
                episode_dict = test_sampler.get_episode()

                support_x = episode_dict["support_X"]
                support_y = episode_dict["support_y"]
                query_x = episode_dict["query_X"]
                query_y = episode_dict["query_y"]

                loss, acc = self.evaluate_episode(
                    tf.constant(support_x, dtype=tf.float32),
                    tf.constant(support_y, dtype=tf.int32),
                    tf.constant(query_x, dtype=tf.float32),
                    tf.constant(query_y, dtype=tf.int32),
                )

                test_losses.append(float(loss))
                test_accs.append(float(acc))

            avg_test_loss = np.mean(test_losses)
            avg_test_acc = np.mean(test_accs)

            logger.info(
                f"\nTest Subject {test_subject}: "
                f"Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_acc:.4f}"
            )

            cv_results["train_losses"].append(np.mean(fold_results["train_losses"]))
            cv_results["train_accuracies"].append(
                np.mean(fold_results["train_accuracies"])
            )
            cv_results["val_losses"].append(np.mean(fold_results["val_losses"]))
            cv_results["val_accuracies"].append(np.mean(fold_results["val_accuracies"]))
            cv_results["test_losses"].append(avg_test_loss)
            cv_results["test_accuracies"].append(avg_test_acc)

        logger.info(f"\n{'=' * 60}")
        logger.info("CROSS-VALIDATION RESULTS")
        logger.info(f"{'=' * 60}")
        logger.info(
            f"Average Test Accuracy: {np.mean(cv_results['test_accuracies']):.4f} "
            f"(±{np.std(cv_results['test_accuracies']):.4f})"
        )
        logger.info(f"Average Test Loss: {np.mean(cv_results['test_losses']):.4f}")
        logger.info(f"{'=' * 60}\n")

        return cv_results
