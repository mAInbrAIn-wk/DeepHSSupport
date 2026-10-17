"""
Unit tests for Data Splits, 3D Sequence Scaling, and Masking Propagation (AP 0 / AP 3 / AP 5).
Compatible with both `unittest` (`python -m unittest discover -s tests -v`) and `pytest`.

Tests:
1. `fb.split_student_ids` guarantees zero student overlap across Train/Val/Test (70/15/15).
2. `fb.scale_3d_sequences` actually scales valid (non-padded) entries to zero-mean/unit-variance on Train while keeping `-99.0` padding untouched (unlike the broken `arr[split][mask] = ...` copy bug).
3. `SinCosPositionalEncoding` in `autoregressive_transformer.py` propagates Keras masks (`_keras_mask`).
4. `build_exam_transformer` includes `Masking` and `PositionalEncoding` layers.
"""

import unittest
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import deepsupport.data_engine.feature_builder as fb
from deepsupport.models.autoregressive_transformer import SinCosPositionalEncoding
from deepsupport.models.exam_transformer import build_exam_transformer, PositionalEncoding


class TestDataSplitsAndScaling(unittest.TestCase):

    def test_split_student_ids_zero_leakage_and_proportions(self):
        """Verify 70/15/15 student-level split has zero overlap between Train, Val, and Test."""
        student_ids = np.repeat(np.arange(1, 1001), 4)
        tr_ids, va_ids, te_ids = fb.split_student_ids(student_ids, seed=42)

        self.assertEqual(len(tr_ids), 700)
        self.assertEqual(len(va_ids), 150)
        self.assertEqual(len(te_ids), 150)

        self.assertEqual(len(set(tr_ids).intersection(set(va_ids))), 0)
        self.assertEqual(len(set(tr_ids).intersection(set(te_ids))), 0)
        self.assertEqual(len(set(va_ids).intersection(set(te_ids))), 0)
        self.assertEqual(len(set(tr_ids) | set(va_ids) | set(te_ids)), 1000)

    def test_scale_3d_sequences_scales_valid_and_preserves_padding(self):
        """
        Verify fb.scale_3d_sequences scales valid entries (mean ~ 0, std ~ 1 on train)
        and leaves PADDING_VALUE (-99.0) untouched.
        """
        rng = np.random.default_rng(42)
        X_tr = rng.normal(loc=50.0, scale=12.0, size=(80, 10, 4)).astype(np.float32)
        X_va = rng.normal(loc=50.0, scale=12.0, size=(20, 10, 4)).astype(np.float32)
        X_te = rng.normal(loc=50.0, scale=12.0, size=(20, 10, 4)).astype(np.float32)

        X_tr[:40, 7:, :] = fb.PADDING_VALUE
        X_va[:10, 7:, :] = fb.PADDING_VALUE
        X_te[:10, 7:, :] = fb.PADDING_VALUE

        X_tr_s, X_va_s, X_te_s, scaler = fb.scale_3d_sequences(X_tr, X_va, X_te)

        self.assertFalse(np.allclose(X_tr_s[40:, :7, :], X_tr[40:, :7, :]))

        valid_mask_tr = (X_tr_s != fb.PADDING_VALUE).all(axis=-1)
        valid_vals_tr = X_tr_s[valid_mask_tr]
        self.assertTrue(np.allclose(valid_vals_tr.mean(axis=0), 0.0, atol=1e-5))
        self.assertTrue(np.allclose(valid_vals_tr.std(axis=0), 1.0, atol=1e-5))

        self.assertTrue(np.all(X_tr_s[:40, 7:, :] == fb.PADDING_VALUE))
        self.assertTrue(np.all(X_va_s[:10, 7:, :] == fb.PADDING_VALUE))
        self.assertTrue(np.all(X_te_s[:10, 7:, :] == fb.PADDING_VALUE))

    def test_sincos_positional_encoding_propagates_keras_mask(self):
        """Verify SinCosPositionalEncoding has supports_masking=True and preserves _keras_mask."""
        inp = layers.Input(shape=(12, 6))
        masked = layers.Masking(mask_value=fb.PADDING_VALUE)(inp)
        pos = SinCosPositionalEncoding()(masked)

        self.assertIsNotNone(pos._keras_mask)

        pooled = layers.GlobalAveragePooling1D()(pos)
        model = tf.keras.Model(inputs=inp, outputs=pooled)

        x1 = np.ones((2, 12, 6), dtype=np.float32)
        x1[:, 5:, :] = fb.PADDING_VALUE
        out1 = model.predict(x1, verbose=0)
        self.assertFalse(np.isnan(out1).any())

    def test_exam_transformer_contains_masking_and_positional_encoding(self):
        """Verify build_exam_transformer includes both Masking and PositionalEncoding layers."""
        model = build_exam_transformer(seq_len=20, n_features=8)
        layer_types = [type(layer) for layer in model.layers]
        self.assertIn(layers.Masking, layer_types, "Exam Transformer is missing Masking layer!")
        self.assertIn(PositionalEncoding, layer_types, "Exam Transformer is missing PositionalEncoding layer!")


if __name__ == "__main__":
    unittest.main()
