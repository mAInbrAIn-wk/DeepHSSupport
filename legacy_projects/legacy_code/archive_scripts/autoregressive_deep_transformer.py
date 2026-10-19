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

# Import data preparation from the old script
sys.path.insert(0, str(Path('src').absolute()))
from autoregressive_next_exam import prepare_next_exam_dataset

PADDING_VALUE = -99.0

@tf.keras.utils.register_keras_serializable()
class SinCosPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
    
    # NEU: Sin/Cos Positional Encoding
    x = SinCosPositionalEncoding()(x)
    
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Transformer Blcke
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

def train_autoregressive_deep_transformer(data_dir=None):
    if data_dir is None:
        data_dir = Path(os.environ.get('DATA_DIR', 'output_v4_grid_v41/S01_baseline/universe_A'))
    else:
        data_dir = Path(data_dir)
        
    print("="*70)
    print(f" DEEP TRANSFORMER AUTOREGRESSOR + POSITIONAL ENCODING (dir={data_dir}) ")
    print("="*70)
    
    X_hist, X_ctx, y_grade, y_pass, student_ids = prepare_next_exam_dataset(data_dir)
    
    # Group-consistent Split on student IDs
    unique_students = np.unique(student_ids)
    tr_students, temp_students = train_test_split(unique_students, test_size=0.30, random_state=42)
    va_students, te_students = train_test_split(temp_students, test_size=0.50, random_state=42)
    
    tr_idx = np.where(np.isin(student_ids, tr_students))[0]
    va_idx = np.where(np.isin(student_ids, va_students))[0]
    te_idx = np.where(np.isin(student_ids, te_students))[0]
    
    scaler_seq = StandardScaler()
    scaler_ctx = StandardScaler()
    
    vm_tr = X_hist[tr_idx, :, 0] != PADDING_VALUE
    scaler_seq.fit(X_hist[tr_idx][vm_tr])
    scaler_ctx.fit(X_ctx[tr_idx])
    
    X_hist_scaled = X_hist.copy()
    X_ctx_scaled = X_ctx.copy()
    for s_idx in [tr_idx, va_idx, te_idx]:
        vm = X_hist[s_idx, :, 0] != PADDING_VALUE
        X_hist_scaled[s_idx][vm] = scaler_seq.transform(X_hist[s_idx][vm])
        X_ctx_scaled[s_idx] = scaler_ctx.transform(X_ctx[s_idx])
        
    model = build_deep_transformer_dual_head(
        seq_timesteps=X_hist.shape[1],
        seq_features=X_hist.shape[2],
        context_features=X_ctx.shape[1]
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'out_grade': 'mse', 'out_pass': 'binary_crossentropy'},
        loss_weights={'out_grade': 1.0, 'out_pass': 0.8},
        metrics={'out_grade': ['mae'], 'out_pass': ['accuracy']}
    )
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    
    print("Starte Training (Max 12 Epochen, Batch-Size 512)...")
    model.fit(
        {'exam_history': X_hist_scaled[tr_idx], 'next_exam_context': X_ctx_scaled[tr_idx]},
        {'out_grade': y_grade[tr_idx], 'out_pass': y_pass[tr_idx]},
        validation_data=(
            {'exam_history': X_hist_scaled[va_idx], 'next_exam_context': X_ctx_scaled[va_idx]},
            {'out_grade': y_grade[va_idx], 'out_pass': y_pass[va_idx]}
        ),
        epochs=12, batch_size=512, verbose=1, callbacks=[es]
    )
    
    print("\nEvaluiere auf Test-Set...")
    preds = model.predict({'exam_history': X_hist_scaled[te_idx], 'next_exam_context': X_ctx_scaled[te_idx]}, verbose=0)
    
    r2 = r2_score(y_grade[te_idx], preds[0].flatten())
    auc = roc_auc_score(y_pass[te_idx], preds[1].flatten())
    
    (data_dir / 'models').mkdir(exist_ok=True, parents=True)
    model.save(data_dir / 'models' / 'autoregressive_deep_transformer.keras')

    metrics_out = {
        'Next_Exam_Grade_R2': float(r2),
        'Next_Exam_Pass_ROC_AUC': float(auc)
    }
    (data_dir / 'metrics').mkdir(exist_ok=True, parents=True)
    with open(data_dir / 'metrics' / 'autoregressive_deep_transformer_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_out, f, indent=4)
        
    print(f"\nERGEBNISSE DEEP TRANSFORMER AUTOREGRESSOR (mit PE):")
    print(f" -> R2 Score (Note): {r2:.4f}")
    print(f" -> ROC-AUC (Pass):  {auc:.4f}")

main = train_autoregressive_deep_transformer

if __name__ == '__main__':
    train_autoregressive_deep_transformer()
