import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Dict, List
import logging
from pathlib import Path
from data_loaders.pain_meta_dataset import PainMetaDataset
from data_loaders.loso_cross_validator import LOSOCrossValidator
from data_loaders.meta_ds_sampler import SixWayKShotSampler
from data_loaders.pain_ds_config import PainDatasetConfig
from utils.logger import setup_logger

logger = setup_logger("FewShotPainLearner", level=logging.INFO)


class MultimodalPrototypicalNetwork(keras.Model):
    """Multimodal Prototypical Networks for few-shot learning on pain data."""

    def __init__(self,
                 sequence_length: int = 2500,
                 num_sensors: int = 3,
                 num_classes: int = 6,
                 embedding_dim: int = 64,
                 modality_names: Tuple[str, ...] = ('EDA', 'ECG', 'EMG'),
                 fusion_method: str = 'concat',
                 distance_metric: str = 'euclidean'):
        """
        Args:
            sequence_length: Length of temporal sequence
            num_sensors: Number of sensor channels
            num_classes: Number of pain levels (6-way)
            embedding_dim: Dimension of embedding space per modality
            modality_names: Names of modalities (EDA, ECG, EMG)
            fusion_method: 'concat', 'mean', 'attention'
            distance_metric: 'euclidean' or 'cosine'
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.num_sensors = num_sensors
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.modality_names = modality_names
        self.fusion_method = fusion_method
        self.distance_metric = distance_metric

        # Create separate encoder for each modality
        self.modality_encoders = {}
        for modality_name in modality_names:
            self.modality_encoders[modality_name] = self._build_encoder(
                modality_name, embedding_dim
            )

        # Fusion layer based on fusion method
        if fusion_method == 'concat':
            self.fused_embedding_dim = embedding_dim * len(modality_names)
            self.fusion_layer = keras.layers.Dense(embedding_dim, activation='relu')
        elif fusion_method == 'mean':
            self.fused_embedding_dim = embedding_dim
            self.fusion_layer = None
        elif fusion_method == 'attention':
            self.fused_embedding_dim = embedding_dim * len(modality_names)
            self.attention_weights = keras.layers.Dense(len(modality_names),
                                                        activation='softmax')
            self.fusion_layer = keras.layers.Dense(embedding_dim, activation='relu')
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        logger.info(f"Initialized MultimodalPrototypicalNetwork with {len(modality_names)} modalities")
        logger.info(f"Fusion method: {fusion_method}, Final embedding dim: {self.fused_embedding_dim}")

    def _build_encoder(self, modality_name: str, embedding_dim: int) -> keras.Sequential:
        """Build 1D CNN encoder for a single modality."""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.sequence_length, 1)),

            # Block 1
            keras.layers.Conv1D(32, kernel_size=7, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv1D(32, kernel_size=7, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(pool_size=4),
            keras.layers.Dropout(0.3),

            # Block 2
            keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(pool_size=4),
            keras.layers.Dropout(0.3),

            # Block 3
            keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling1D(),

            # Embedding layer
            keras.layers.Dense(embedding_dim, activation='relu'),
            keras.layers.LayerNormalization()],
            name=f"encoder_{modality_name}"
        )
        logger.info(f"Built CNN encoder with {modality_name}")
        logger.info(model.summary())

    def encode(self, x, training=False):
        """
        Map input to combined embedding space.

        Args:
            x: [batch_size, sequence_length, num_sensors]
            training: Whether in training mode

        Returns:
            embeddings: [batch_size, fused_embedding_dim] or [batch_size, embedding_dim]
        """
        # Split input by modality
        modality_embeddings = []

        for i, modality_name in enumerate(self.modality_names):
            # Extract single modality: [batch_size, sequence_length, 1]
            modality_data = x[:, :, i:i + 1]

            # Encode modality
            encoder = self.modality_encoders[modality_name]
            embedding = encoder(modality_data, training=training)
            modality_embeddings.append(embedding)

        # Fuse embeddings
        if self.fusion_method == 'concat':
            # Concatenate all embeddings
            fused = tf.concat(modality_embeddings, axis=1)  # [batch, embedding_dim * num_modalities]
            fused = self.fusion_layer(fused)  # [batch, embedding_dim]

        elif self.fusion_method == 'mean':
            # Simple mean of embeddings
            fused = tf.stack(modality_embeddings, axis=1)  # [batch, num_modalities, embedding_dim]
            fused = tf.reduce_mean(fused, axis=1)  # [batch, embedding_dim]

        elif self.fusion_method == 'attention':
            # Attention-based fusion
            stacked = tf.stack(modality_embeddings, axis=1)  # [batch, num_modalities, embedding_dim]

            # Compute attention weights
            # Use mean of modality embeddings as query
            query = tf.reduce_mean(stacked, axis=1, keepdims=True)  # [batch, 1, embedding_dim]
            scores = tf.reduce_sum(stacked * query, axis=2)  # [batch, num_modalities]
            weights = tf.nn.softmax(scores, axis=1)  # [batch, num_modalities]

            # Apply weights
            weighted = stacked * tf.expand_dims(weights, axis=2)  # [batch, num_modalities, embedding_dim]
            fused_weighted = tf.reduce_sum(weighted, axis=1)  # [batch, embedding_dim]

            # Concatenate original embeddings with weighted sum for richer representation
            fused = tf.concat(modality_embeddings + [fused_weighted], axis=1)
            fused = self.fusion_layer(fused)  # [batch, embedding_dim]

        return fused

    def compute_distances(self, query_embeddings, prototype_embeddings):
        """
        Compute distances between query and class prototypes.

        Args:
            query_embeddings: [num_queries, embedding_dim]
            prototype_embeddings: [num_classes, embedding_dim]

        Returns:
            distances: [num_queries, num_classes]
        """
        if self.distance_metric == 'euclidean':
            distances = tf.sqrt(
                tf.reduce_sum(
                    (tf.expand_dims(query_embeddings, 1) -
                     tf.expand_dims(prototype_embeddings, 0)) ** 2,
                    axis=2
                ) + 1e-8
            )
        elif self.distance_metric == 'cosine':
            query_norm = tf.nn.l2_normalize(query_embeddings, axis=1)
            prototype_norm = tf.nn.l2_normalize(prototype_embeddings, axis=1)
            distances = 1 - tf.matmul(query_norm, tf.transpose(prototype_norm))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def call(self, support_x, support_y, query_x, training=False):
        """
        Forward pass for few-shot learning.

        Args:
            support_x: [n_way * k_shot, sequence_length, num_sensors]
            support_y: [n_way * k_shot] (class labels 0-5)
            query_x: [n_way * q_query, sequence_length, num_sensors]
            training: Whether in training mode

        Returns:
            logits: [n_way * q_query, n_way]
        """
        # Encode all samples
        support_embeddings = self.encode(support_x, training=training)
        query_embeddings = self.encode(query_x, training=training)

        logger.debug(f"Support embeddings shape: {tf.shape(support_embeddings)}")
        logger.debug(f"Query embeddings shape: {tf.shape(query_embeddings)}")

        # Compute class prototypes as mean of support embeddings per class
        prototypes = []

        for class_id in range(self.num_classes):
            # Create mask for samples belonging to this class
            mask = tf.cast(tf.equal(support_y, class_id), tf.float32)
            count = tf.reduce_sum(mask)

            # Compute mean embedding for this class
            class_embeddings = support_embeddings * tf.expand_dims(mask, 1)
            prototype = tf.reduce_sum(class_embeddings, axis=0) / (count + 1e-8)
            prototypes.append(prototype)

        prototypes = tf.stack(prototypes, axis=0)  # [num_classes, embedding_dim]

        logger.debug(f"Prototypes shape: {tf.shape(prototypes)}")

        # Compute distances to prototypes
        distances = self.compute_distances(query_embeddings, prototypes)

        # Convert distances to logits (lower distance = higher probability)
        logits = -distances

        return logits

class FewShotPainLearner:
    """Meta-learning trainer for personalized pain assessment."""

    def __init__(self,
                 config: PainDatasetConfig,
                 data_dir: str = "./dataset/np-dataset",
                 learning_rate: float = 1e-3,
                 fusion_method: str = 'attention'):
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
            data_dir=data_dir,
            config=config,
            normalize=True,
            normalize_per_subject=True
        )

        self.cv = LOSOCrossValidator(
            dataset=self.dataset,
            k_shot=config.k_shot,
            q_query=config.q_query
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
            distance_metric='euclidean'
        )

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        logger.info(f"Initialized FewShotPainLearner with {len(self.cv.subjects)} subjects")
        logger.info(f"Data shape: (sequence_length={config.sequence_length}, num_sensors={num_sensors})")
        logger.info(f"Modalities: {config.modality_names}")
        logger.info(f"Fusion method: {fusion_method}")

    def train_step(self, support_x, support_y, query_x, query_y):
        """Single training step on one episode."""
        with tf.GradientTape() as tape:
            logits = self.model(support_x, support_y, query_x, training=True)
            loss = self.loss_fn(query_y, logits)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

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

    def train(self,
              num_epochs: int = 10,
              episodes_per_epoch: int = 100,
              val_episodes: int = 20):
        """
        Train on all subjects using leave-one-subject-out cross-validation.
        """
        cv_results = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'test_losses': [],
            'test_accuracies': []
        }

        num_subjects = len(self.cv.subjects)

        for fold, test_subject in enumerate(self.cv.subjects):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Fold {fold + 1}/{num_subjects}: Test subject = {test_subject}")
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
                distance_metric='euclidean'
            )
            self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

            # Get fold dictionary with samplers
            fold_dict = self.cv.get_fold(test_subject)

            train_sampler = fold_dict['train_sampler']
            val_sampler = fold_dict['val_sampler']
            test_sampler = fold_dict['test_sampler']

            fold_results = {
                'train_losses': [],
                'train_accuracies': [],
                'val_losses': [],
                'val_accuracies': [],
            }

            for epoch in range(num_epochs):
                # Training
                epoch_train_losses = []
                epoch_train_accs = []

                for episode in range(episodes_per_epoch):
                    episode_dict = train_sampler.get_episode()

                    support_x = episode_dict['support_X']  # [6*k_shot, 2500, 3]
                    support_y = episode_dict['support_y']  # [6*k_shot]
                    query_x = episode_dict['query_X']  # [6*q_query, 2500, 3]
                    query_y = episode_dict['query_y']  # [6*q_query]

                    loss, acc = self.train_step(
                        tf.constant(support_x, dtype=tf.float32),
                        tf.constant(support_y, dtype=tf.int32),
                        tf.constant(query_x, dtype=tf.float32),
                        tf.constant(query_y, dtype=tf.int32)
                    )

                    epoch_train_losses.append(float(loss))
                    epoch_train_accs.append(float(acc))

                # Validation
                epoch_val_losses = []
                epoch_val_accs = []

                for _ in range(val_episodes):
                    episode_dict = val_sampler.get_episode()

                    support_x = episode_dict['support_X']
                    support_y = episode_dict['support_y']
                    query_x = episode_dict['query_X']
                    query_y = episode_dict['query_y']

                    loss, acc = self.evaluate_episode(
                        tf.constant(support_x, dtype=tf.float32),
                        tf.constant(support_y, dtype=tf.int32),
                        tf.constant(query_x, dtype=tf.float32),
                        tf.constant(query_y, dtype=tf.int32)
                    )

                    epoch_val_losses.append(float(loss))
                    epoch_val_accs.append(float(acc))

                avg_train_loss = np.mean(epoch_train_losses)
                avg_train_acc = np.mean(epoch_train_accs)
                avg_val_loss = np.mean(epoch_val_losses)
                avg_val_acc = np.mean(epoch_val_accs)

                fold_results['train_losses'].append(avg_train_loss)
                fold_results['train_accuracies'].append(avg_train_acc)
                fold_results['val_losses'].append(avg_val_loss)
                fold_results['val_accuracies'].append(avg_val_acc)

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

                support_x = episode_dict['support_X']
                support_y = episode_dict['support_y']
                query_x = episode_dict['query_X']
                query_y = episode_dict['query_y']

                loss, acc = self.evaluate_episode(
                    tf.constant(support_x, dtype=tf.float32),
                    tf.constant(support_y, dtype=tf.int32),
                    tf.constant(query_x, dtype=tf.float32),
                    tf.constant(query_y, dtype=tf.int32)
                )

                test_losses.append(float(loss))
                test_accs.append(float(acc))

            avg_test_loss = np.mean(test_losses)
            avg_test_acc = np.mean(test_accs)

            logger.info(
                f"\nTest Subject {test_subject}: "
                f"Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_acc:.4f}"
            )

            cv_results['train_losses'].append(np.mean(fold_results['train_losses']))
            cv_results['train_accuracies'].append(np.mean(fold_results['train_accuracies']))
            cv_results['val_losses'].append(np.mean(fold_results['val_losses']))
            cv_results['val_accuracies'].append(np.mean(fold_results['val_accuracies']))
            cv_results['test_losses'].append(avg_test_loss)
            cv_results['test_accuracies'].append(avg_test_acc)

        logger.info(f"\n{'=' * 60}")
        logger.info("CROSS-VALIDATION RESULTS")
        logger.info(f"{'=' * 60}")
        logger.info(f"Average Test Accuracy: {np.mean(cv_results['test_accuracies']):.4f} "
                    f"(±{np.std(cv_results['test_accuracies']):.4f})")
        logger.info(f"Average Test Loss: {np.mean(cv_results['test_losses']):.4f}")
        logger.info(f"{'=' * 60}\n")

        return cv_results

def main():
    """Example usage of the few-shot pain learner."""
    config = PainDatasetConfig()

    logger.info("=" * 60)
    logger.info("Multimodal Few-Shot Learning for Personalized Pain Assessment")
    logger.info("=" * 60)

    # Try different fusion methods
    fusion_methods = ['concat', 'mean', 'attention']

    for fusion_method in fusion_methods:
        logger.info(f"\nTraining with fusion method: {fusion_method}")

        learner = FewShotPainLearner(
            config=config,
            data_dir="../data",
            learning_rate=1e-3,
            fusion_method=fusion_method
        )

        cv_results = learner.train(
            num_epochs=100,
            episodes_per_epoch=50,
            val_episodes=10
        )

        logger.info(f"Training with {fusion_method} complete!")

if __name__ == "__main__":
    main()