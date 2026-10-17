import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import numpy as np
from pathlib import Path
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization, 
                                     MultiHeadAttention, Concatenate, Masking, Add, GlobalAveragePooling1D)
from tensorflow.keras.models import Model

# Import data preparation from feature_builder
sys.path.insert(0, str(Path('src').absolute()))
import deepsupport.data_engine.feature_builder as fb
from deepsupport.models.autoregressive_gru import prepare_next_exam_dataset
from deepsupport.evaluation.metrics_logger import DualHeadEvaluator

PADDING_VALUE = -99.0

@tf.keras.utils.register_keras_serializable()
class SinCosPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        
        positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        
        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * (i // 2.0)) / tf.cast(d_model, tf.float32))
        angle_rads = positions * angle_rates
        
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.reshape(tf.stack([sines, cosines], axis=-1), [seq_len, d_model])
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return inputs + tf.cast(pos_encoding, inputs.dtype)

def build_deep_transformer_dual_head(seq_timesteps: int, seq_features: int, context_features: int, 
                                     d_model=64, num_heads=4, num_blocks=3, dropout_rate=0.2) -> Model:
    # === ZWEIG A: Historie (Deep Transformer) ===
    seq_input = Input(shape=(seq_timesteps, seq_features), name='exam_history')
    x = Masking(mask_value=PADDING_VALUE)(seq_input)
    
    # Eingangs-Projektion
    x = Dense(d_model, activation='relu')(x)
    
    # NEU: Sin/Cos Positional Encoding mit Mask-Propagation
    x = SinCosPositionalEncoding()(x)
    
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Transformer Blöcke
    for _ in range(num_blocks):
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate)(x, x)
        x = Add()([x, attn_out])
        x = LayerNormalization()(x)
        
        ffn = Dense(d_model * 2, activation='relu')(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(d_model)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)
        
    history_embedding = GlobalAveragePooling1D()(x)
    history_embedding = Dense(64, activation='relu')(history_embedding)
    history_embedding = LayerNormalization()(history_embedding)
    
    # === ZWEIG B: Ziel-Klausur Kontext ===
    ctx_input = Input(shape=(context_features,), name='next_exam_context')
    ctx_dense = Dense(32, activation='relu')(ctx_input)
    ctx_dense = LayerNormalization()(ctx_dense)
    ctx_dense = Dropout(dropout_rate)(ctx_dense)
    
    # === FUSION ===
    merged = Concatenate()([history_embedding, ctx_dense])
    
    shared = Dense(64, activation='relu')(merged)
    shared = LayerNormalization()(shared)
    shared = Dropout(dropout_rate)(shared)
    
    h_grade = Dense(32, activation='relu')(shared)
    h_grade = Dropout(0.1)(h_grade)
    out_grade = Dense(1, activation='linear', name='out_grade')(h_grade)
    
    h_pass = Dense(32, activation='relu')(shared)
    h_pass = Dropout(0.1)(h_pass)
    out_pass = Dense(1, activation='sigmoid', name='out_pass')(h_pass)
    
    model = Model(inputs=[seq_input, ctx_input], outputs=[out_grade, out_pass])
    return model

def train_autoregressive_deep_transformer(data_dir=None, output_dir=None, mode: str = 'standard', epochs: int = 12, batch_size: int = 512):
    if data_dir is None:
        data_dir = Path(os.environ.get('DATA_DIR', 'data_v4_grid/S01_baseline/universe_A'))
    else:
        data_dir = Path(data_dir)
        
    if output_dir is None:
        output_dir = Path(os.environ.get('OUTPUT_DIR', str(data_dir)))
    else:
        output_dir = Path(output_dir)
        
    print("="*70)
    print(f" DEEP TRANSFORMER AUTOREGRESSOR + POSITIONAL ENCODING (data={data_dir}, mode={mode}) ")
    print("="*70)
    
    X_hist, X_ctx, y_grade, y_pass, student_ids = fb.prepare_next_exam_dataset(data_dir, mode=mode)
    
    # Group-consistent Split on student IDs via feature_builder
    tr_students, va_students, te_students = fb.split_student_ids(student_ids, seed=42)
    
    tr_idx = np.where(np.isin(student_ids, tr_students))[0]
    va_idx = np.where(np.isin(student_ids, va_students))[0]
    te_idx = np.where(np.isin(student_ids, te_students))[0]
    
    # Zentralisierte 3D-Skalierung (behebt den NumPy Fancy-Indexing Bug X_hist_scaled[s_idx][vm] = ...)
    scaled_hist, _ = fb.scale_3d_sequences(
        {"train": X_hist[tr_idx], "val": X_hist[va_idx], "test": X_hist[te_idx]},
        fit_key="train",
        padding_value=PADDING_VALUE,
    )
    X_hist_tr, X_hist_va, X_hist_te = scaled_hist["train"], scaled_hist["val"], scaled_hist["test"]

    scaler_ctx = StandardScaler()
    X_ctx_tr = scaler_ctx.fit_transform(X_ctx[tr_idx]).astype(np.float32)
    X_ctx_va = scaler_ctx.transform(X_ctx[va_idx]).astype(np.float32)
    X_ctx_te = scaler_ctx.transform(X_ctx[te_idx]).astype(np.float32)
        
    model = build_deep_transformer_dual_head(
        seq_timesteps=X_hist.shape[1],
        seq_features=X_hist.shape[2],
        context_features=X_ctx.shape[1]
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss=['mse', 'binary_crossentropy'],
        loss_weights=[1.0, 0.8],
        metrics=[['mae'], ['accuracy']]
    )
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    lr_sched = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=0)
    
    # 1/K_i Sample-Gewichtung pro Studierendem (verhindert Dominanz von Langzeitstudierenden)
    unique_tr, counts_tr = np.unique(student_ids[tr_idx], return_counts=True)
    count_map_tr = dict(zip(unique_tr, counts_tr))
    sw_tr = np.array([1.0 / max(1, count_map_tr[s]) for s in student_ids[tr_idx]], dtype=np.float32)
    sw_tr *= len(sw_tr) / np.sum(sw_tr)

    print(f"Starte Training (Max {epochs} Epochen, Batch-Size {batch_size}, 1/K_i gewichtet)...")
    history = model.fit(
        [X_hist_tr, X_ctx_tr],
        [y_grade[tr_idx], y_pass[tr_idx]],
        sample_weight=sw_tr,
        validation_data=(
            [X_hist_va, X_ctx_va],
            [y_grade[va_idx], y_pass[va_idx]]
        ),
        epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es, lr_sched]
    )
    
    print("\nEvaluiere auf Test-Set...")
    preds = model.predict([X_hist_te, X_ctx_te], verbose=0)
    
    r2 = r2_score(y_grade[te_idx], preds[0].flatten())
    auc = roc_auc_score(y_pass[te_idx], preds[1].flatten())
    
    evaluator = DualHeadEvaluator(base_dir=output_dir, model_name='autoregressive_deep_transformer', mode=mode)
    evaluator.evaluate_and_log(
        y_grade_true=y_grade[te_idx], y_grade_pred=preds[0].flatten(),
        y_pass_true=y_pass[te_idx], y_pass_prob=preds[1].flatten(),
        history=history.history, model=model, mode=mode,
        extra_metrics={
            'Next_Exam_Grade_R2': float(r2),
            'Next_Exam_Pass_ROC_AUC': float(auc),
        },
    )
        
    print(f"\nERGEBNISSE DEEP TRANSFORMER AUTOREGRESSOR (mit PE):")
    print(f" -> R2 Score (Note): {r2:.4f}")
    print(f" -> ROC-AUC (Pass):  {auc:.4f}")

main = train_autoregressive_deep_transformer

if __name__ == '__main__':
    train_autoregressive_deep_transformer()
