import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from architecture.crossmod_feature_map_encoder import CrossModFeatureMapEncoder
from architecture.eegnet_style_encoder import EEGNetStyleEncoder
from architecture.multimodal_proto_net import MultimodalPrototypicalNetwork
from data_loaders.loso_cross_validator import LOSOCrossValidator
from data_loaders.meta_ds_sampler import SixWayKShotSampler
from data_loaders.pain_ds_config import PainDatasetConfig
from data_loaders.pain_meta_dataset import PainMetaDataset
from learner.episodic_learning_engine import EpisodicLearningEngine
from learner.few_shot_pain_learner import FewShotPainLearner


def _write_synthetic_dataset(
    data_dir: Path,
    num_subjects: int = 3,
    n_way: int = 6,
    reps_per_class: int = 4,
    seq_len: int = 2500,
    full_sensor_count: int = 6,
    compressed: bool = False,
) -> None:
    """Create a small deterministic dataset compatible with sensor_idx=[1,4,5]."""
    rng = np.random.default_rng(123)

    X_rows = []
    y_rows = []
    subject_rows = []

    for subject in range(num_subjects):
        for class_id in range(n_way):
            for _ in range(reps_per_class):
                x = rng.normal(size=(seq_len, full_sensor_count, 1)).astype(np.float32)
                y = np.zeros(n_way, dtype=np.float32)
                y[class_id] = 1.0

                X_rows.append(x)
                y_rows.append(y)
                subject_rows.append(subject)

    X = np.stack(X_rows, axis=0)
    y = np.stack(y_rows, axis=0)
    subjects = np.array(subject_rows, dtype=np.int32)

    if compressed:
        np.savez_compressed(data_dir / "X_pre.npz", data=X)
        np.savez_compressed(data_dir / "y_heater.npz", data=y)
        np.savez_compressed(data_dir / "subjects.npz", data=subjects)
    else:
        np.save(data_dir / "X_pre.npy", X)
        np.save(data_dir / "y_heater.npy", y)
        np.save(data_dir / "subjects.npy", subjects)


def _write_synthetic_biovid_dataset(
    data_dir: Path,
    num_subjects: int = 4,
    num_classes: int = 5,
    train_samples_per_class: int = 6,
    test_samples_per_class: int = 2,
    seq_len: int = 1152,
    compressed: bool = False,
) -> None:
    """Create a tiny BioVid-like PartA train/test tree for split-contract tests."""
    rng = np.random.default_rng(456)
    root = data_dir / "BioVid" / "PartA"
    modalities = ("GSR", "ECG", "EMG")
    splits = {
        "Train": train_samples_per_class,
        "Test": test_samples_per_class,
    }
    subject_keys = [f"subject_{idx:02d}" for idx in range(num_subjects)]

    for split_name, samples_per_class in splits.items():
        for modality in modalities:
            (root / split_name / modality).mkdir(parents=True, exist_ok=True)

        for subject_idx, subject_key in enumerate(subject_keys):
            labels = np.concatenate(
                [
                    np.full((samples_per_class, 1), class_id, dtype=np.uint8)
                    for class_id in range(num_classes)
                ],
                axis=0,
            )
            for modality_idx, modality in enumerate(modalities):
                # Inject small modality-specific shifts so the channels are not identical.
                mean_shift = float(subject_idx * 0.25 + modality_idx * 0.1)
                data = np.concatenate(
                    [
                        rng.normal(
                            loc=class_id + mean_shift,
                            scale=0.5,
                            size=(samples_per_class, seq_len, 1),
                        ).astype(np.float32)
                        for class_id in range(num_classes)
                    ],
                    axis=0,
                )
                if compressed:
                    np.savez_compressed(
                        root / split_name / modality / f"{subject_key}_data.npz",
                        data=data,
                    )
                    np.savez_compressed(
                        root / split_name / modality / f"{subject_key}_label.npz",
                        data=labels,
                    )
                else:
                    np.save(
                        root / split_name / modality / f"{subject_key}_data.npy",
                        data,
                    )
                    np.save(
                        root / split_name / modality / f"{subject_key}_label.npy",
                        labels,
                    )


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        _write_synthetic_dataset(self.data_dir)

        self.config = PainDatasetConfig(
            dataset_source="painmonit",
            sequence_length=2500,
            n_way=6,
            num_stimuli_levels=6,
            k_shot=1,
            q_query=1,
            num_epochs=1,
            tasks_per_epoch=1,
            val_tasks=1,
            heldout_eval_tasks=1,
            single_loso_fold=False,
            seed=7,
            deterministic_ops=True,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_triplet_mining_strategy_validation(self):
        self.assertEqual(PainDatasetConfig().triplet_mining_strategy, "batch_hard")
        self.assertEqual(
            PainDatasetConfig(
                triplet_mining_strategy="triplet_center"
            ).triplet_mining_strategy,
            "triplet_center",
        )
        with self.assertRaises(ValueError):
            PainDatasetConfig(triplet_mining_strategy="semi_hard")

    def test_eegnet_defaults_are_configured_for_joint_sensor_encoding(self):
        config = PainDatasetConfig(dataset_source="painmonit")

        self.assertEqual(config.eegnet_temporal_filters, 8)
        self.assertEqual(config.eegnet_depth_multiplier, 2)
        self.assertEqual(config.eegnet_separable_filters, 16)
        self.assertEqual(config.eegnet_temporal_kernel_size, 64)
        self.assertEqual(config.eegnet_separable_kernel_size, 16)
        self.assertEqual(config.eegnet_pool_size_1, 4)
        self.assertEqual(config.eegnet_pool_size_2, 8)
        self.assertEqual(config.eegnet_dropout_rate, 0.25)

    def test_eegnet_hyperparameters_are_validated(self):
        with self.assertRaisesRegex(ValueError, "eegnet_temporal_filters must be > 0"):
            PainDatasetConfig(dataset_source="painmonit", eegnet_temporal_filters=0)
        with self.assertRaisesRegex(ValueError, "eegnet_dropout_rate must be in"):
            PainDatasetConfig(dataset_source="painmonit", eegnet_dropout_rate=1.0)
        with self.assertRaisesRegex(
            ValueError, "eegnet_l2_weight must be non-negative"
        ):
            PainDatasetConfig(dataset_source="painmonit", eegnet_l2_weight=-1e-4)

    def test_can_config_validation(self):
        config = PainDatasetConfig(
            dataset_source="painmonit",
            task_class_ids=(0, 5),
            attention_mode="can",
            classifier_mode="prototype",
        )

        self.assertEqual(config.attention_mode, "can")
        self.assertEqual(config.can_attention_temperature, 1.0)
        self.assertEqual(config.can_meta_hidden_dim, 32)
        self.assertEqual(config.can_local_loss_weight, 1.0)
        self.assertEqual(config.can_global_loss_weight, 0.1)
        self.assertEqual(config.can_margin_loss_weight, 0.2)
        self.assertEqual(config.can_margin_target, 0.3)

        with self.assertRaisesRegex(ValueError, "requires classifier_mode"):
            PainDatasetConfig(
                dataset_source="painmonit",
                task_class_ids=(0, 5),
                attention_mode="can",
                classifier_mode="soft_knn",
            )
        with self.assertRaisesRegex(ValueError, "at least two task classes"):
            PainDatasetConfig(
                dataset_source="painmonit",
                task_class_ids=(0,),
                attention_mode="can",
            )
        with self.assertRaisesRegex(ValueError, "can_attention_temperature"):
            PainDatasetConfig(
                dataset_source="painmonit",
                task_class_ids=(0, 5),
                attention_mode="can",
                can_attention_temperature=0.0,
            )
        with self.assertRaisesRegex(ValueError, "can_margin_loss_weight"):
            PainDatasetConfig(
                dataset_source="painmonit",
                task_class_ids=(0, 5),
                attention_mode="can",
                can_margin_loss_weight=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "can_margin_target"):
            PainDatasetConfig(
                dataset_source="painmonit",
                task_class_ids=(0, 5),
                attention_mode="can",
                can_margin_target=-0.1,
            )

    def test_crossmod_config_selects_eda_ecg_modalities(self):
        painmonit_config = PainDatasetConfig(
            dataset_source="painmonit",
            task_class_ids=(0, 5),
            attention_mode="can",
            encoder_backend="crossmod",
        )

        self.assertEqual(painmonit_config.encoder_backend, "crossmod")
        self.assertEqual(painmonit_config.num_sensors, 2)
        self.assertEqual(painmonit_config.sensor_idx, (1, 4))
        self.assertEqual(painmonit_config.modality_names, ("EDA", "ECG"))

        biovid_config = PainDatasetConfig(
            dataset_source="biovid_part_a",
            task_class_ids=(0, 4),
            attention_mode="can",
            encoder_backend="crossmod",
        )
        self.assertEqual(biovid_config.biovid_modalities, ("GSR", "ECG"))
        self.assertEqual(biovid_config.num_sensors, 2)

    def test_crossmod_hyperparameters_are_validated(self):
        with self.assertRaisesRegex(ValueError, "requires attention_mode"):
            PainDatasetConfig(
                dataset_source="painmonit",
                task_class_ids=(0, 5),
                encoder_backend="crossmod",
                attention_mode="none",
            )
        with self.assertRaisesRegex(ValueError, "crossmod_num_heads must be > 0"):
            PainDatasetConfig(
                dataset_source="painmonit",
                task_class_ids=(0, 5),
                attention_mode="can",
                encoder_backend="crossmod",
                crossmod_num_heads=0,
            )
        with self.assertRaisesRegex(ValueError, "dropout_rate must be in"):
            PainDatasetConfig(
                dataset_source="painmonit",
                task_class_ids=(0, 5),
                attention_mode="can",
                encoder_backend="crossmod",
                eegnet_dropout_rate=1.0,
            )
        with self.assertRaisesRegex(ValueError, "divisible"):
            PainDatasetConfig(
                dataset_source="painmonit",
                task_class_ids=(0, 5),
                attention_mode="can",
                encoder_backend="crossmod",
                eegnet_separable_filters=10,
                crossmod_num_heads=8,
            )

    def test_learned_prototype_memory_config_validation(self):
        config = PainDatasetConfig(
            dataset_source="painmonit",
            task_class_ids=(0, 5),
            attention_mode="can",
            classifier_mode="prototype",
            can_support_mode="learned_prototype_memory",
            learned_prototype_slots_per_class=3,
        )

        self.assertEqual(config.can_support_mode, "learned_prototype_memory")
        self.assertEqual(config.learned_prototype_slots_per_class, 3)

        with self.assertRaisesRegex(ValueError, "requires attention_mode"):
            PainDatasetConfig(
                dataset_source="painmonit",
                task_class_ids=(0, 5),
                attention_mode="none",
                can_support_mode="learned_prototype_memory",
            )
        with self.assertRaisesRegex(ValueError, "slots_per_class"):
            PainDatasetConfig(
                dataset_source="painmonit",
                task_class_ids=(0, 5),
                attention_mode="can",
                can_support_mode="learned_prototype_memory",
                learned_prototype_slots_per_class=0,
            )

    def test_eegnet_encoder_outputs_compact_joint_embeddings(self):
        encoder = EEGNetStyleEncoder(
            sequence_length=1152,
            num_sensors=3,
            embedding_dim=64,
        )

        embeddings = encoder(tf.zeros((2, 1152, 3)), training=False)

        self.assertEqual(embeddings.shape, (2, 64))
        self.assertLess(encoder.count_params(), 100_000)

    def test_eegnet_encoder_extracts_temporal_feature_map(self):
        encoder = EEGNetStyleEncoder(
            sequence_length=32,
            num_sensors=3,
            embedding_dim=8,
            temporal_filters=2,
            depth_multiplier=1,
            separable_filters=4,
            temporal_kernel_size=8,
            separable_kernel_size=4,
            pool_size_1=2,
            pool_size_2=2,
            dropout_rate=0.0,
        )

        feature_map = encoder.extract_feature_map(
            tf.zeros((2, 32, 3)),
            training=False,
        )
        embeddings = encoder.embed_feature_map(feature_map, training=False)

        self.assertEqual(feature_map.shape, (2, 8, 4))
        self.assertEqual(embeddings.shape, (2, 8))

    def test_crossmod_encoder_returns_fused_feature_maps_without_logits(self):
        encoder = CrossModFeatureMapEncoder(
            sequence_length=32,
            num_sensors=2,
            temporal_filters=2,
            depth_multiplier=1,
            separable_filters=4,
            temporal_kernel_size=8,
            separable_kernel_size=4,
            pool_size_1=2,
            pool_size_2=2,
            dropout_rate=0.0,
            num_heads=2,
            hidden_dim=8,
            num_layers=1,
        )

        eda_features, ecg_features = encoder.extract_modality_feature_maps(
            tf.zeros((2, 32, 2)),
            training=False,
        )
        feature_map = encoder.extract_feature_map(tf.zeros((2, 32, 2)), training=False)

        self.assertEqual(eda_features.shape, (2, 8, 4))
        self.assertEqual(ecg_features.shape, (2, 8, 4))
        self.assertEqual(feature_map.shape, (2, 8, 8))
        self.assertFalse(hasattr(encoder, "fc4"))
        with self.assertRaisesRegex(RuntimeError, "does not produce embeddings"):
            encoder.embed_feature_map(feature_map, training=False)

    def _small_can_model(
        self,
        *,
        can_support_mode: str = "sampled",
        learned_prototype_slots_per_class: int = 1,
        encoder_backend: str = "eegnet",
        num_sensors: int = 3,
    ) -> MultimodalPrototypicalNetwork:
        return MultimodalPrototypicalNetwork(
            sequence_length=32,
            num_sensors=num_sensors,
            num_classes=2,
            embedding_dim=4,
            eegnet_temporal_filters=2,
            eegnet_depth_multiplier=1,
            eegnet_separable_filters=4,
            eegnet_temporal_kernel_size=8,
            eegnet_separable_kernel_size=4,
            eegnet_pool_size_1=2,
            eegnet_pool_size_2=2,
            eegnet_dropout_rate=0.0,
            attention_mode="can",
            can_meta_hidden_dim=4,
            can_support_mode=can_support_mode,
            learned_prototype_slots_per_class=learned_prototype_slots_per_class,
            encoder_backend=encoder_backend,
            crossmod_num_heads=2,
            crossmod_hidden_dim=8,
            crossmod_num_layers=1,
        )

    def _small_episode_batch(self):
        rng = np.random.default_rng(2718)
        support_x = tf.constant(rng.normal(size=(2, 4, 32, 3)), dtype=tf.float32)
        query_x = tf.constant(rng.normal(size=(2, 4, 32, 3)), dtype=tf.float32)
        support_y = tf.constant([[0, 0, 1, 1], [1, 0, 1, 0]], dtype=tf.int32)
        query_y = tf.constant([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=tf.int32)
        return support_x, support_y, query_x, query_y

    def test_can_forward_outputs_attention_and_auxiliary_logits(self):
        model = self._small_can_model()
        support_x, support_y, query_x, _ = self._small_episode_batch()

        outputs = model.forward_episode_batch(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=False,
        )

        self.assertEqual(outputs["logits"].shape, (2, 4, 2))
        self.assertEqual(outputs["can_local_logits"].shape, (2, 4, 8, 2))
        self.assertEqual(outputs["can_proto_attention"].shape, (2, 4, 2, 8))
        self.assertEqual(outputs["can_query_attention"].shape, (2, 4, 2, 8))
        self.assertFalse(model.encoder.enable_embedding_projection)
        self.assertIsNone(model.encoder.embedding_dense)
        self.assertIsNone(model.encoder.embedding_norm)
        self.assertIsNone(model.triplet_centers)
        self.assertNotIn("support_embeddings", outputs)
        self.assertNotIn("query_embeddings", outputs)
        self.assertNotIn("prototypes", outputs)
        self.assertNotIn("can_global_logits", outputs)
        np.testing.assert_allclose(
            tf.reduce_sum(outputs["can_proto_attention"], axis=-1).numpy(),
            np.ones((2, 4, 2)),
            atol=1e-5,
        )

    def test_crossmod_can_forward_outputs_attention_and_auxiliary_logits(self):
        model = self._small_can_model(encoder_backend="crossmod", num_sensors=2)
        rng = np.random.default_rng(314)
        support_x = tf.constant(rng.normal(size=(2, 4, 32, 2)), dtype=tf.float32)
        query_x = tf.constant(rng.normal(size=(2, 4, 32, 2)), dtype=tf.float32)
        support_y = tf.constant([[0, 0, 1, 1], [1, 0, 1, 0]], dtype=tf.int32)

        outputs = model.forward_episode_batch(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=False,
        )

        self.assertEqual(outputs["logits"].shape, (2, 4, 2))
        self.assertEqual(outputs["can_local_logits"].shape, (2, 4, 8, 2))
        self.assertEqual(outputs["can_proto_attention"].shape, (2, 4, 2, 8))
        self.assertEqual(outputs["can_query_attention"].shape, (2, 4, 2, 8))
        self.assertNotIn("support_embeddings", outputs)
        self.assertNotIn("query_embeddings", outputs)
        np.testing.assert_allclose(
            tf.reduce_sum(outputs["can_query_attention"], axis=-1).numpy(),
            np.ones((2, 4, 2)),
            atol=1e-5,
        )

    def test_crossmod_can_loss_reaches_frontend_and_attention_variables(self):
        model = self._small_can_model(encoder_backend="crossmod", num_sensors=2)
        rng = np.random.default_rng(2027)
        support_x = tf.constant(rng.normal(size=(1, 4, 32, 2)), dtype=tf.float32)
        query_x = tf.constant(rng.normal(size=(1, 4, 32, 2)), dtype=tf.float32)
        support_y = tf.constant([[0, 0, 1, 1]], dtype=tf.int32)
        query_y = tf.constant([[0, 0, 1, 1]], dtype=tf.int32)

        with tf.GradientTape() as tape:
            logits = model.forward_episode_batch(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                training=True,
            )["logits"]
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    query_y,
                    logits,
                    from_logits=True,
                )
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        gradients_by_name = {
            getattr(variable, "path", variable.name): gradient
            for variable, gradient in zip(model.trainable_variables, gradients)
            if gradient is not None
        }
        gradient_names = [
            getattr(variable, "path", variable.name)
            for variable, gradient in zip(model.trainable_variables, gradients)
            if gradient is not None
        ]
        self.assertGreaterEqual(
            sum("temporal_conv/kernel" in name for name in gradient_names),
            2,
        )
        self.assertTrue(
            any("crossmod_eda_to_ecg_attention" in name for name in gradients_by_name)
        )
        self.assertTrue(
            any("crossmod_ecg_to_eda_attention" in name for name in gradients_by_name)
        )

    def test_learned_prototype_memory_can_forward_uses_configurable_slots(self):
        model = self._small_can_model(
            can_support_mode="learned_prototype_memory",
            learned_prototype_slots_per_class=3,
        )
        support_x, support_y, query_x, _ = self._small_episode_batch()

        outputs = model.forward_episode_batch(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=False,
        )

        self.assertEqual(outputs["logits"].shape, (2, 4, 2))
        self.assertEqual(outputs["slot_similarity_scores"].shape, (2, 4, 6))
        self.assertEqual(outputs["support_feature_maps"].shape[1], 6)
        self.assertEqual(outputs["prototype_support_y"].shape, (2, 6))
        self.assertNotIn("support_embeddings", outputs)
        self.assertNotIn("query_embeddings", outputs)
        self.assertNotIn("prototypes", outputs)

    def test_learned_prototype_memory_gradients_reach_memory_and_can(self):
        model = self._small_can_model(
            can_support_mode="learned_prototype_memory",
            learned_prototype_slots_per_class=2,
        )
        support_x, support_y, query_x, query_y = self._small_episode_batch()

        with tf.GradientTape() as tape:
            outputs = model.forward_episode_batch(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                training=True,
            )
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    query_y,
                    outputs["logits"],
                    from_logits=True,
                )
            )

        variables = (
            model.prototype_memory.trainable_variables
            + model.cross_attention.trainable_variables
        )
        gradients = tape.gradient(loss, variables)
        self.assertTrue(any(gradient is not None for gradient in gradients))

    def test_can_batched_forward_matches_per_task_forward(self):
        model = self._small_can_model()
        support_x, support_y, query_x, _ = self._small_episode_batch()

        batched = model.forward_episode_batch(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=False,
        )
        per_task_logits = []
        per_task_local_logits = []
        for task_idx in range(int(support_x.shape[0])):
            single = model.forward_episode(
                support_x=support_x[task_idx],
                support_y=support_y[task_idx],
                query_x=query_x[task_idx],
                training=False,
            )
            per_task_logits.append(single["logits"])
            per_task_local_logits.append(single["can_local_logits"])

        np.testing.assert_allclose(
            batched["logits"].numpy(),
            tf.stack(per_task_logits, axis=0).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            batched["can_local_logits"].numpy(),
            tf.stack(per_task_local_logits, axis=0).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_can_gradients_reach_cam_variables(self):
        model = self._small_can_model()
        support_x, support_y, query_x, query_y = self._small_episode_batch()

        with tf.GradientTape() as tape:
            outputs = model.forward_episode_batch(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                training=True,
            )
            task_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    query_y,
                    outputs["logits"],
                    from_logits=True,
                )
            )
            local_labels = tf.tile(
                query_y[:, :, tf.newaxis],
                [1, 1, int(outputs["can_local_logits"].shape[2])],
            )
            local_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    local_labels,
                    outputs["can_local_logits"],
                    from_logits=True,
                )
            )
            loss = task_loss + local_loss

        variables = model.cross_attention.trainable_variables
        gradients = tape.gradient(loss, variables)

        self.assertTrue(any(gradient is not None for gradient in gradients))

    def test_can_objective_does_not_require_embedding_outputs(self):
        model = self._small_can_model()
        support_x, support_y, query_x, query_y = self._small_episode_batch()
        learner = FewShotPainLearner.__new__(FewShotPainLearner)
        learner.config = PainDatasetConfig(
            dataset_source="painmonit",
            task_class_ids=(0, 5),
            attention_mode="can",
            classifier_mode="prototype",
            can_local_loss_weight=1.0,
        )
        learner.model = model
        learner.triplet_loss_weight = 1.0
        learner.triplet_margin = 0.1
        learner.triplet_mining_strategy = "triplet_center"
        learner.engine = EpisodicLearningEngine(learner)

        episode_outputs = model.forward_episode_batch(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=False,
        )
        self.assertNotIn("support_embeddings", episode_outputs)
        self.assertNotIn("query_embeddings", episode_outputs)

        objective = learner.engine.compute_task_batch_objective(
            episode_outputs,
            support_y,
            query_y,
        )

        np.testing.assert_allclose(objective["triplet_losses"].numpy(), np.zeros(2))
        np.testing.assert_allclose(objective["can_global_losses"].numpy(), np.zeros(2))
        self.assertTrue(np.all(np.isfinite(objective["can_margin_losses"].numpy())))
        self.assertTrue(np.all(np.isfinite(objective["losses"].numpy())))

    def test_can_margin_loss_uses_true_minus_best_other_scores(self):
        learner = FewShotPainLearner.__new__(FewShotPainLearner)
        learner.config = PainDatasetConfig(
            dataset_source="painmonit",
            task_class_ids=(0, 5),
            attention_mode="can",
            classifier_mode="prototype",
            can_local_loss_weight=0.0,
            can_margin_loss_weight=0.2,
            can_margin_target=0.3,
        )
        learner.triplet_loss_weight = 1.0
        learner.model = type("DummyModel", (), {"losses": []})()
        learner.engine = EpisodicLearningEngine(learner)
        logits = tf.zeros((2, 3, 2), dtype=tf.float32)
        query_y = tf.constant([[0, 1, 0], [1, 0, 1]], dtype=tf.int32)
        support_y = tf.constant([[0, 1], [0, 1]], dtype=tf.int32)
        similarity_scores = tf.constant(
            [
                [[0.8, 0.4], [0.6, 0.7], [0.2, 0.4]],
                [[0.1, 0.6], [0.4, 0.3], [0.5, 0.55]],
            ],
            dtype=tf.float32,
        )

        objective = learner.engine.compute_task_batch_objective(
            {
                "logits": logits,
                "similarity_scores": similarity_scores,
            },
            support_y,
            query_y,
        )

        expected_unweighted = np.array(
            [
                np.mean([0.0, 0.2, 0.5]),
                np.mean([0.0, 0.2, 0.25]),
            ],
            dtype=np.float32,
        )
        expected_weighted = 0.2 * expected_unweighted
        np.testing.assert_allclose(
            objective["can_margin_losses"].numpy(),
            expected_weighted,
            rtol=1e-6,
            atol=1e-6,
        )
        expected_task_loss = np.log(2.0)
        np.testing.assert_allclose(
            objective["losses"].numpy(),
            expected_task_loss + expected_weighted,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_can_margin_loss_is_zero_when_target_is_satisfied(self):
        learner = FewShotPainLearner.__new__(FewShotPainLearner)
        learner.config = PainDatasetConfig(
            dataset_source="painmonit",
            task_class_ids=(0, 5),
            attention_mode="can",
            classifier_mode="prototype",
            can_local_loss_weight=0.0,
            can_margin_loss_weight=0.2,
            can_margin_target=0.3,
        )
        learner.triplet_loss_weight = 1.0
        learner.model = type("DummyModel", (), {"losses": []})()
        learner.engine = EpisodicLearningEngine(learner)
        logits = tf.zeros((1, 2, 2), dtype=tf.float32)
        query_y = tf.constant([[0, 1]], dtype=tf.int32)
        support_y = tf.constant([[0, 1]], dtype=tf.int32)
        similarity_scores = tf.constant(
            [[[0.9, 0.1], [0.2, 0.8]]],
            dtype=tf.float32,
        )

        objective = learner.engine.compute_task_batch_objective(
            {
                "logits": logits,
                "similarity_scores": similarity_scores,
            },
            support_y,
            query_y,
        )

        np.testing.assert_allclose(objective["can_margin_losses"].numpy(), [0.0])

    def test_can_batch_forward_outputs_standard_logits(self):
        model = self._small_can_model()
        support_x, support_y, query_x, _ = self._small_episode_batch()

        outputs = model.forward_episode_batch(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=False,
        )

        self.assertEqual(outputs["logits"].shape, (2, 4, 2))
        self.assertEqual(outputs["similarity_scores"].shape, (2, 4, 2))

    def test_batch_hard_triplet_loss_uses_hardest_positive_and_negative(self):
        learner = FewShotPainLearner.__new__(FewShotPainLearner)
        learner.triplet_loss_weight = 1.0
        learner.triplet_margin = 0.5
        learner.engine = EpisodicLearningEngine(learner)

        embeddings = tf.constant(
            [
                [1.0, 0.0],
                [0.5, np.sqrt(3.0) / 2.0],
                [-0.5, np.sqrt(3.0) / 2.0],
                [-1.0, 0.0],
            ],
            dtype=tf.float32,
        )
        labels = tf.constant([0, 0, 1, 1], dtype=tf.int32)

        loss = learner._compute_batch_hard_triplet_loss(embeddings, labels)

        self.assertAlmostEqual(float(loss.numpy()), 0.25, places=5)

    def test_batch_hard_triplet_loss_ignores_invalid_anchors(self):
        learner = FewShotPainLearner.__new__(FewShotPainLearner)
        learner.triplet_loss_weight = 1.0
        learner.triplet_margin = 0.5
        learner.engine = EpisodicLearningEngine(learner)

        embeddings = tf.eye(3, dtype=tf.float32)
        labels = tf.constant([0, 1, 2], dtype=tf.int32)

        loss = learner._compute_batch_hard_triplet_loss(embeddings, labels)

        self.assertEqual(float(loss.numpy()), 0.0)

    def test_triplet_center_loss_uses_nearest_negative_center(self):
        learner = FewShotPainLearner.__new__(FewShotPainLearner)
        learner.triplet_loss_weight = 1.0
        learner.triplet_margin = 1.0
        learner.triplet_mining_strategy = "triplet_center"
        learner.triplet_center_gradient_clip_norm = 0.01
        learner.engine = EpisodicLearningEngine(learner)
        learner.model = MultimodalPrototypicalNetwork(
            sequence_length=32,
            num_sensors=3,
            num_classes=3,
            embedding_dim=2,
            eegnet_temporal_filters=2,
            eegnet_depth_multiplier=1,
            eegnet_separable_filters=4,
            eegnet_temporal_kernel_size=8,
            eegnet_separable_kernel_size=4,
            eegnet_pool_size_1=2,
            eegnet_pool_size_2=2,
            eegnet_dropout_rate=0.0,
        )
        learner.model.triplet_centers.assign(
            tf.constant(
                [
                    [0.0, 0.0],
                    [3.0, 0.0],
                    [0.0, 3.0],
                ],
                dtype=tf.float32,
            )
        )

        embeddings = tf.constant(
            [
                [2.0, 0.0],
                [3.0, 1.0],
            ],
            dtype=tf.float32,
        )
        labels = tf.constant([0, 1], dtype=tf.int32)

        with tf.GradientTape() as tape:
            loss = learner._compute_triplet_center_loss(embeddings, labels)

        self.assertAlmostEqual(float(loss.numpy()), 1.25, places=5)
        self.assertAlmostEqual(
            float(learner._compute_triplet_loss(embeddings, labels).numpy()),
            1.25,
            places=5,
        )
        center_gradient = tape.gradient(loss, learner.model.triplet_centers)
        self.assertIsNotNone(center_gradient)

    def test_batched_triplet_center_loss_matches_per_task_loss(self):
        learner = FewShotPainLearner.__new__(FewShotPainLearner)
        learner.triplet_loss_weight = 1.0
        learner.triplet_margin = 1.0
        learner.triplet_mining_strategy = "triplet_center"
        learner.triplet_center_gradient_clip_norm = 0.01
        learner.engine = EpisodicLearningEngine(learner)
        learner.model = MultimodalPrototypicalNetwork(
            sequence_length=32,
            num_sensors=3,
            num_classes=3,
            embedding_dim=2,
            eegnet_temporal_filters=2,
            eegnet_depth_multiplier=1,
            eegnet_separable_filters=4,
            eegnet_temporal_kernel_size=8,
            eegnet_separable_kernel_size=4,
            eegnet_pool_size_1=2,
            eegnet_pool_size_2=2,
            eegnet_dropout_rate=0.0,
        )
        learner.model.triplet_centers.assign(
            tf.constant(
                [
                    [0.0, 0.0],
                    [3.0, 0.0],
                    [0.0, 3.0],
                ],
                dtype=tf.float32,
            )
        )
        embeddings_batch = tf.constant(
            [
                [
                    [2.0, 0.0],
                    [3.0, 1.0],
                ],
                [
                    [0.0, 2.0],
                    [2.0, 0.0],
                ],
            ],
            dtype=tf.float32,
        )
        labels_batch = tf.constant([[0, 1], [2, 1]], dtype=tf.int32)

        batched = learner.engine.compute_triplet_center_loss_batch(
            embeddings_batch,
            labels_batch,
        )
        per_task = tf.stack(
            [
                learner._compute_triplet_center_loss(
                    embeddings_batch[task_idx],
                    labels_batch[task_idx],
                )
                for task_idx in range(2)
            ],
            axis=0,
        )

        np.testing.assert_allclose(batched.numpy(), per_task.numpy(), atol=1e-6)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                learner.engine.compute_triplet_center_loss_batch(
                    embeddings_batch,
                    labels_batch,
                )
            )
        center_gradient = tape.gradient(loss, learner.model.triplet_centers)
        self.assertIsNotNone(center_gradient)

    def test_embedding_batch_size_must_be_positive(self):
        self.assertEqual(
            PainDatasetConfig(embedding_batch_size=2).embedding_batch_size,
            2,
        )
        with self.assertRaisesRegex(ValueError, "embedding_batch_size must be > 0"):
            PainDatasetConfig(embedding_batch_size=0)

    def test_validation_checkpoint_config_validation(self):
        self.assertEqual(PainDatasetConfig().validation_checkpoint_metric, "accuracy")
        self.assertEqual(PainDatasetConfig().validation_checkpoint_mode, "auto")
        self.assertEqual(
            PainDatasetConfig(
                validation_checkpoint_metric="f1",
                validation_checkpoint_mode="max",
            ).validation_checkpoint_metric,
            "f1",
        )
        with self.assertRaisesRegex(ValueError, "validation_checkpoint_metric"):
            PainDatasetConfig(validation_checkpoint_metric="unknown")
        with self.assertRaisesRegex(ValueError, "validation_checkpoint_mode"):
            PainDatasetConfig(validation_checkpoint_mode="sideways")

    def test_forward_episode_batch_matches_per_task_eval_order(self):
        model = MultimodalPrototypicalNetwork(
            sequence_length=32,
            num_sensors=3,
            num_classes=2,
            embedding_dim=4,
            eegnet_temporal_filters=2,
            eegnet_depth_multiplier=1,
            eegnet_separable_filters=4,
            eegnet_temporal_kernel_size=8,
            eegnet_separable_kernel_size=4,
            eegnet_pool_size_1=2,
            eegnet_pool_size_2=2,
            eegnet_dropout_rate=0.0,
        )
        rng = np.random.default_rng(2025)
        support_x = tf.constant(rng.normal(size=(3, 4, 32, 3)), dtype=tf.float32)
        query_x = tf.constant(rng.normal(size=(3, 4, 32, 3)), dtype=tf.float32)
        support_y = tf.constant(
            [
                [0, 0, 1, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            dtype=tf.int32,
        )

        batched = model.forward_episode_batch(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=False,
        )
        per_task_logits = []
        per_task_prototypes = []
        for task_idx in range(int(support_x.shape[0])):
            single = model.forward_episode(
                support_x=support_x[task_idx],
                support_y=support_y[task_idx],
                query_x=query_x[task_idx],
                training=False,
            )
            per_task_logits.append(single["logits"])
            per_task_prototypes.append(single["prototypes"])

        np.testing.assert_allclose(
            batched["logits"].numpy(),
            tf.stack(per_task_logits, axis=0).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            batched["prototypes"].numpy(),
            tf.stack(per_task_prototypes, axis=0).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_encode_returns_joint_embedding_without_modality_stack(self):
        model = MultimodalPrototypicalNetwork(
            sequence_length=32,
            num_sensors=3,
            num_classes=2,
            embedding_dim=4,
            eegnet_temporal_filters=2,
            eegnet_depth_multiplier=1,
            eegnet_separable_filters=4,
            eegnet_temporal_kernel_size=8,
            eegnet_separable_kernel_size=4,
            eegnet_pool_size_1=2,
            eegnet_pool_size_2=2,
            eegnet_dropout_rate=0.0,
        )

        embeddings = model.encode(tf.random.normal((2, 32, 3)), training=False)

        self.assertEqual(embeddings.shape, (2, 4))
        self.assertIsInstance(model.encoder, EEGNetStyleEncoder)
        self.assertFalse(hasattr(model, "modality_encoders"))

    def test_logit_scale_is_trainable_and_initialized_by_metric(self):
        cosine_model = MultimodalPrototypicalNetwork(
            sequence_length=32,
            num_sensors=3,
            num_classes=2,
            embedding_dim=4,
            eegnet_temporal_filters=2,
            eegnet_depth_multiplier=1,
            eegnet_separable_filters=4,
            eegnet_temporal_kernel_size=8,
            eegnet_separable_kernel_size=4,
            eegnet_pool_size_1=2,
            eegnet_pool_size_2=2,
            eegnet_dropout_rate=0.0,
            distance_metric="cosine",
        )
        euclidean_model = MultimodalPrototypicalNetwork(
            sequence_length=32,
            num_sensors=3,
            num_classes=2,
            embedding_dim=4,
            eegnet_temporal_filters=2,
            eegnet_depth_multiplier=1,
            eegnet_separable_filters=4,
            eegnet_temporal_kernel_size=8,
            eegnet_separable_kernel_size=4,
            eegnet_pool_size_1=2,
            eegnet_pool_size_2=2,
            eegnet_dropout_rate=0.0,
            distance_metric="euclidean",
        )

        self.assertAlmostEqual(float(cosine_model.logit_scale.numpy()), 10.0)
        self.assertAlmostEqual(float(euclidean_model.logit_scale.numpy()), 1.0)
        self.assertTrue(
            any(
                variable is cosine_model.logit_scale
                for variable in cosine_model.trainable_variables
            )
        )

        rng = np.random.default_rng(2026)
        support_x = tf.constant(rng.normal(size=(1, 4, 32, 3)), dtype=tf.float32)
        query_x = tf.constant(rng.normal(size=(1, 4, 32, 3)), dtype=tf.float32)
        support_y = tf.constant([[0, 0, 1, 1]], dtype=tf.int32)
        query_y = tf.constant([[0, 0, 1, 1]], dtype=tf.int32)

        with tf.GradientTape() as tape:
            logits = cosine_model.forward_episode_batch(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                training=True,
            )["logits"]
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    query_y,
                    logits,
                    from_logits=True,
                )
            )

        gradient = tape.gradient(loss, cosine_model.logit_scale)
        self.assertIsNotNone(gradient)

    def test_eegnet_encoder_layers_are_present(self):
        model = MultimodalPrototypicalNetwork(
            sequence_length=32,
            num_sensors=3,
            num_classes=2,
            embedding_dim=8,
            eegnet_temporal_filters=2,
            eegnet_depth_multiplier=1,
            eegnet_separable_filters=4,
            eegnet_temporal_kernel_size=8,
            eegnet_separable_kernel_size=4,
            eegnet_pool_size_1=2,
            eegnet_pool_size_2=2,
            eegnet_dropout_rate=0.0,
        )

        embeddings = model.encode(tf.random.normal((2, 32, 3)), training=False)
        self.assertEqual(embeddings.shape, (2, 8))

        layer_names = {layer.name for layer in model.encoder.layers}
        self.assertIn("temporal_conv", layer_names)
        self.assertIn("sensor_depthwise_conv", layer_names)
        self.assertIn("separable_temporal_conv", layer_names)
        self.assertIn("global_pooling", layer_names)

    def test_loso_split_contracts(self):
        dataset = PainMetaDataset(data_dir=str(self.data_dir), config=self.config)
        cv = LOSOCrossValidator(
            dataset=dataset,
            seed=self.config.seed,
        )

        all_subjects = set(dataset.unique_subjects.tolist())

        for test_subject in dataset.unique_subjects:
            fold = cv.get_fold(int(test_subject))
            train_subjects = set(fold["train_subjects"])
            val_subjects = set(fold["val_subjects"])

            self.assertNotIn(test_subject, train_subjects)
            self.assertNotIn(test_subject, val_subjects)
            self.assertTrue(train_subjects.isdisjoint(val_subjects))
            self.assertEqual(
                train_subjects | val_subjects, all_subjects - {int(test_subject)}
            )
            self.assertEqual(fold["n_train_subjects"], len(train_subjects))
            self.assertEqual(fold["n_val_subjects"], len(val_subjects))

    def test_sampler_task_contracts(self):
        dataset = PainMetaDataset(data_dir=str(self.data_dir), config=self.config)
        cv = LOSOCrossValidator(
            dataset=dataset,
            seed=self.config.seed,
        )

        fold = cv.get_fold(test_subject=int(dataset.unique_subjects[0]))
        samplers = (
            fold["train_sampler"],
            fold["val_sampler"],
            fold["test_sampler"],
        )

        for sampler in samplers:
            task = sampler.get_task()
            self.assertEqual(
                task["support_X"].shape,
                (
                    self.config.n_way * self.config.k_shot,
                    self.config.sequence_length,
                    len(self.config.sensor_idx),
                ),
            )
            self.assertEqual(
                task["query_X"].shape,
                (
                    self.config.n_way * self.config.q_query,
                    self.config.sequence_length,
                    len(self.config.sensor_idx),
                ),
            )
            self.assertEqual(
                task["support_y"].shape, (self.config.n_way * self.config.k_shot,)
            )
            self.assertEqual(
                task["query_y"].shape, (self.config.n_way * self.config.q_query,)
            )
            self.assertIn("subject", task)

            support_counts = np.bincount(task["support_y"], minlength=self.config.n_way)
            query_counts = np.bincount(task["query_y"], minlength=self.config.n_way)
            self.assertTrue(np.all(support_counts == self.config.k_shot))
            self.assertTrue(np.all(query_counts == self.config.q_query))

        expected_train_subject = (
            -1 if len(fold["train_subjects"]) > 1 else fold["train_subjects"][0]
        )
        expected_val_subject = (
            -1 if len(fold["val_subjects"]) > 1 else fold["val_subjects"][0]
        )
        self.assertEqual(
            fold["train_sampler"].get_task()["subject"], expected_train_subject
        )
        self.assertEqual(
            fold["val_sampler"].get_task()["subject"], expected_val_subject
        )
        self.assertEqual(
            fold["test_sampler"].get_task()["subject"],
            int(dataset.unique_subjects[0]),
        )

    def test_test_sampler_uses_available_query_budget(self):
        tmp = tempfile.TemporaryDirectory()
        data_dir = Path(tmp.name)
        _write_synthetic_dataset(data_dir, num_subjects=4)
        config = PainDatasetConfig(
            dataset_source="painmonit",
            sequence_length=2500,
            n_way=6,
            num_stimuli_levels=6,
            k_shot=3,
            q_query=2,
            num_epochs=1,
            tasks_per_epoch=1,
            val_tasks=1,
            heldout_eval_tasks=1,
            k_shot_adaptation_steps=3,
            single_loso_fold=False,
            seed=7,
            deterministic_ops=True,
        )
        original_fallback_k = SixWayKShotSampler.VALIDATION_FALLBACK_K_SHOT
        original_fallback_q = SixWayKShotSampler.VALIDATION_FALLBACK_Q_QUERY
        SixWayKShotSampler.VALIDATION_FALLBACK_K_SHOT = config.k_shot
        SixWayKShotSampler.VALIDATION_FALLBACK_Q_QUERY = config.q_query
        try:
            dataset = PainMetaDataset(data_dir=str(data_dir), config=config)
            cv = LOSOCrossValidator(dataset=dataset, seed=config.seed)
            fold = cv.get_fold(test_subject=int(dataset.unique_subjects[0]))

            train_task = fold["train_sampler"].get_task()
            train_support_counts = np.bincount(
                train_task["support_y"], minlength=config.n_way
            )
            train_query_counts = np.bincount(
                train_task["query_y"], minlength=config.n_way
            )
            self.assertIn(train_task["subject"], fold["train_subjects"])
            self.assertTrue(np.all(train_support_counts == config.k_shot))
            self.assertTrue(np.all(train_query_counts == config.q_query))

            test_task = fold["test_sampler"].get_task()
            test_support_counts = np.bincount(
                test_task["support_y"], minlength=config.n_way
            )
            test_query_counts = np.bincount(
                test_task["query_y"], minlength=config.n_way
            )
            self.assertTrue(np.all(test_support_counts == config.k_shot))
            self.assertTrue(np.all(test_query_counts == 1))
        finally:
            SixWayKShotSampler.VALIDATION_FALLBACK_K_SHOT = original_fallback_k
            SixWayKShotSampler.VALIDATION_FALLBACK_Q_QUERY = original_fallback_q
            tmp.cleanup()

    def test_training_contracts_smoke(self):
        learner = FewShotPainLearner(
            config=self.config,
            data_dir=str(self.data_dir),
            learning_rate=1e-3,
        )

        results = learner.train()

        required_keys = {
            "train_losses",
            "train_accuracies",
            "val_losses",
            "val_accuracies",
            "test_losses",
            "test_accuracies",
            "zero_shot_intra_class_similarities",
            "zero_shot_inter_class_similarities",
            "k_shot_intra_class_similarities",
            "k_shot_inter_class_similarities",
        }
        self.assertTrue(required_keys.issubset(results.keys()))

        n_folds = len(learner.cv.subjects)
        for key in required_keys:
            self.assertEqual(len(results[key]), n_folds)
            self.assertTrue(np.all(np.isfinite(results[key])))

        self.assertEqual(results["validation_checkpoint_metric"], "accuracy")
        self.assertEqual(results["validation_checkpoint_mode"], "auto")
        for key in (
            "validation_checkpoint_values",
            "validation_checkpoint_epochs",
            "validation_checkpoint_steps",
        ):
            self.assertEqual(len(results[key]), n_folds)
            self.assertTrue(np.all(np.isfinite(results[key])))
        self.assertEqual(len(results["validation_checkpoint_metrics"]), n_folds)

        with open(
            results["training_progress_files"][0], newline="", encoding="utf-8"
        ) as handle:
            header = next(csv.reader(handle))
        self.assertIn("checkpoint_metric", header)
        self.assertIn("checkpoint_value", header)
        self.assertIn("checkpoint_is_best", header)

    def test_two_way_task_class_mapping(self):
        config = PainDatasetConfig(
            dataset_source="painmonit",
            sequence_length=2500,
            num_stimuli_levels=6,
            task_class_ids=(0, 5),
            k_shot=1,
            q_query=1,
            num_epochs=1,
            tasks_per_epoch=1,
            val_tasks=1,
            heldout_eval_tasks=1,
            single_loso_fold=False,
            seed=7,
            deterministic_ops=True,
        )
        self.assertEqual(config.n_way, 2)

        dataset = PainMetaDataset(data_dir=str(self.data_dir), config=config)
        cv = LOSOCrossValidator(dataset=dataset, seed=config.seed)
        fold = cv.get_fold(test_subject=int(dataset.unique_subjects[0]))
        task = fold["test_sampler"].get_task()

        self.assertEqual(task["support_X"].shape[0], 2 * config.k_shot)
        self.assertEqual(task["query_X"].shape[0], 2 * config.q_query)
        self.assertTrue(np.array_equal(np.unique(task["support_y"]), np.array([0, 1])))
        self.assertTrue(np.array_equal(np.unique(task["query_y"]), np.array([0, 1])))

    def test_biovid_predefined_split_contracts(self):
        tmp = tempfile.TemporaryDirectory()
        data_dir = Path(tmp.name)
        _write_synthetic_biovid_dataset(data_dir)

        config = PainDatasetConfig(
            dataset_source="biovid_part_a",
            k_shot=1,
            q_query=1,
            num_epochs=1,
            tasks_per_epoch=1,
            val_tasks=1,
            heldout_eval_tasks=1,
            single_loso_fold=False,
            seed=13,
            deterministic_ops=True,
        )
        try:
            # BioVid default binary classes should adapt to (0, 4).
            self.assertEqual(config.task_class_ids, (0, 4))
            self.assertEqual(config.split_strategy, "predefined")
            self.assertFalse(config.enable_window_shift_augmentation)

            dataset = PainMetaDataset(data_dir=str(data_dir), config=config)
            self.assertTrue(dataset.has_predefined_split)
            self.assertEqual(
                sorted(dataset.get_split_subjects("train")),
                sorted(dataset.get_split_subjects("test")),
            )

            train_index = dataset._get_index_for_split("train")
            test_index = dataset._get_index_for_split("test")
            for subject in dataset.get_split_subjects("train"):
                for class_id in range(config.n_way):
                    self.assertGreater(
                        len(train_index[int(subject)][class_id]),
                        len(test_index[int(subject)][class_id]),
                    )

            cv = LOSOCrossValidator(dataset=dataset, seed=config.seed)
            test_subjects = sorted(dataset.get_split_subjects("test"))
            self.assertEqual(sorted(cv.subjects), test_subjects)

            held_out_subject = int(test_subjects[1])
            fold = cv.get_fold(test_subject=held_out_subject)

            self.assertEqual(fold["test_subject"], held_out_subject)
            self.assertEqual(fold["test_subjects"], [held_out_subject])
            self.assertNotIn(held_out_subject, fold["train_subjects"])
            self.assertNotIn(held_out_subject, fold["val_subjects"])
            self.assertEqual(
                sorted(fold["val_subjects"]),
                [subject for subject in test_subjects if subject != held_out_subject],
            )

            train_task = fold["train_sampler"].get_task()
            val_task = fold["val_sampler"].get_task()
            test_task = fold["test_sampler"].get_task()
            expected_support = config.n_way * config.k_shot
            expected_query = config.n_way * config.q_query
            self.assertEqual(train_task["support_X"].shape, (expected_support, 1152, 3))
            self.assertEqual(train_task["query_X"].shape, (expected_query, 1152, 3))
            self.assertEqual(val_task["support_X"].shape, (expected_support, 1152, 3))
            self.assertEqual(val_task["query_X"].shape, (expected_query, 1152, 3))
            self.assertEqual(test_task["support_X"].shape, (expected_support, 1152, 3))
            self.assertEqual(test_task["query_X"].shape, (expected_query, 1152, 3))
            self.assertNotEqual(train_task["subject"], held_out_subject)
            self.assertNotEqual(val_task["subject"], held_out_subject)
            self.assertEqual(test_task["subject"], held_out_subject)
        finally:
            tmp.cleanup()

    def test_painmonit_loader_accepts_compressed_npz_arrays(self):
        tmp = tempfile.TemporaryDirectory()
        data_dir = Path(tmp.name)
        _write_synthetic_dataset(data_dir, compressed=True)

        config = PainDatasetConfig(
            dataset_source="painmonit",
            sequence_length=2500,
            n_way=6,
            num_stimuli_levels=6,
            k_shot=1,
            q_query=1,
            num_epochs=1,
            tasks_per_epoch=1,
            val_tasks=1,
            heldout_eval_tasks=1,
            single_loso_fold=False,
            seed=7,
            deterministic_ops=True,
        )
        try:
            dataset = PainMetaDataset(data_dir=str(data_dir), config=config)
            self.assertEqual(dataset.X.shape, (72, 2500, 3))
            self.assertEqual(dataset.y.shape, (72,))
            self.assertEqual(dataset.subjects.shape, (72,))
        finally:
            tmp.cleanup()

    def test_biovid_loader_accepts_compressed_npz_tree(self):
        tmp = tempfile.TemporaryDirectory()
        data_dir = Path(tmp.name)
        _write_synthetic_biovid_dataset(data_dir, compressed=True)

        config = PainDatasetConfig(
            dataset_source="biovid_part_a",
            k_shot=1,
            q_query=1,
            num_epochs=1,
            tasks_per_epoch=1,
            val_tasks=1,
            heldout_eval_tasks=1,
            single_loso_fold=False,
            seed=13,
            deterministic_ops=True,
        )
        try:
            dataset = PainMetaDataset(data_dir=str(data_dir), config=config)
            self.assertTrue(dataset.has_predefined_split)
            self.assertEqual(dataset.X.shape[1:], (1152, 3))
            self.assertEqual(sorted(np.unique(dataset.y).tolist()), [0, 1, 2, 3, 4])
            self.assertEqual(
                sorted(dataset.get_split_subjects("train")),
                sorted(dataset.get_split_subjects("test")),
            )
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
