import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Concatenate, Masking, Add, GlobalAveragePooling1D, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from sklearn.metrics import r2_score, accuracy_score, classification_report, roc_auc_score

sys.path.insert(0, str(Path('src').absolute()))
import feature_builder as fb
from autoregressive_deep_transformer import SinCosPositionalEncoding
from autoregressive_next_exam import PADDING_VALUE

def build_v37_multi_task_model(seq_timesteps: int, seq_features: int, context_features: int, 
                               d_model=64, num_heads=4, num_blocks=3):
                               
    seq_input = Input(shape=(seq_timesteps, seq_features), name='exam_history')
    x = Masking(mask_value=PADDING_VALUE)(seq_input)
    
    x = Dense(d_model)(x)
    x = SinCosPositionalEncoding()(x)
    
    for _ in range(num_blocks):
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
        attn_out = Dropout(0.1)(attn_out)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, attn_out]))
        
        ffn_out = Dense(d_model, activation='relu')(x)
        ffn_out = Dropout(0.1)(ffn_out)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, ffn_out]))
        
    history_embedding = GlobalAveragePooling1D()(x)
    
    ctx_input = Input(shape=(context_features,), name='next_exam_context')
    
    merged = Concatenate()([history_embedding, ctx_input])
    shared = Dense(64, activation='relu')(merged)
    
    out_grade_local = Dense(1, activation='linear', name='out_grade_local')(shared)
    out_pass_local = Dense(1, activation='sigmoid', name='out_pass_local')(shared)
    
    shared_global = Dense(64, activation='relu')(history_embedding)
    
    out_status_global = Dense(4, activation='softmax', name='out_status_global')(shared_global)
    out_grade_global = Dense(1, activation='linear', name='out_grade_global')(shared_global)
    
    model = Model(inputs={'exam_history': seq_input, 'next_exam_context': ctx_input}, 
                  outputs={
                      'out_grade_local': out_grade_local,
                      'out_pass_local': out_pass_local,
                      'out_status_global': out_status_global,
                      'out_grade_global': out_grade_global
                  })
                  
    return model

def prepare_multi_task_dataset(data_dir: Path, max_history_len: int = 30):
    df_ab, df_pr = fb._load_raw_data(data_dir)
    
    df_ab['hzb_ord'] = df_ab['hzb_typ'].map(fb.HZB_ORDINAL_MAP).fillna(3.0)
    demo_dict = df_ab.set_index('studierenden_id')[['hzb_note', 'hzb_ord', 'erwerbstaetigkeit_std', 'erstakademiker']].to_dict('index')
    
    df_pr['bestanden_int'] = df_pr['bestanden'].astype(int)
    df_pr['note_clean'] = df_pr['note'].fillna(5.0)
    
    hist_feats = ['versuch', 'schwierigkeit', 'cp', 'bestanden_int', 'note_clean', 
                  'support_vorher_fachlich', 'support_vorher_ueberfachlich', 'support_vorher_psychosozial',
                  'support_glz_fachlich', 'support_glz_ueberfachlich', 'support_glz_psychosozial', 'fachsemester']
                  
    X_hist_list, X_ctx_list, y_grade_list, y_pass_list, studi_list = [], [], [], [], []
    
    for studi_id, group in df_pr.groupby('studierenden_id'):
        group = group.sort_values(['fachsemester', 'modul_id'])
        records = group[hist_feats].values
        
        demo_vec = list(demo_dict[studi_id].values())
        
        for k in range(1, len(records)):
            history = records[:k]
            next_exam = records[k]
            
            if len(history) > max_history_len:
                history = history[-max_history_len:]
                
            pad_len = max_history_len - len(history)
            if pad_len > 0:
                pad_mat = np.full((pad_len, len(hist_feats)), PADDING_VALUE, dtype=np.float32)
                pad_seq = np.vstack([pad_mat, history])
            else:
                pad_seq = history
                
            ctx = [next_exam[0], next_exam[1], next_exam[2], next_exam[5], next_exam[6], next_exam[7], next_exam[8], next_exam[9], next_exam[10], next_exam[11]] + demo_vec
            
            X_hist_list.append(pad_seq)
            X_ctx_list.append(ctx)
            y_grade_list.append(next_exam[4])
            y_pass_list.append(next_exam[3])
            studi_list.append(studi_id)
            
    X_hist = np.array(X_hist_list, dtype=np.float32)
    X_ctx = np.array(X_ctx_list, dtype=np.float32)
    y_grade = np.array(y_grade_list, dtype=np.float32)
    y_pass = np.array(y_pass_list, dtype=np.float32)
    
    return X_hist, X_ctx, y_grade, y_pass, studi_list, df_ab

def main():
    data_dir = Path("src/output_dl_seed99999")
    print("Baue Features fr Multi-Task Modell (aus V3.6 Daten)...")
    
    X_hist, X_ctx, y_grade, y_pass, student_ids, df_ab = prepare_multi_task_dataset(data_dir, max_history_len=30)
    
    df_ab = df_ab.set_index('studierenden_id')
    status_map = {'abgeschlossen': 0, 'abgebrochen': 1, 'exmatrikuliert': 2, 'zeitueberschreitung': 3}
    
    y_status_global = []
    y_grade_global = []
    sample_weights_grade = []
    
    for sid in student_ids:
        row = df_ab.loc[sid]
        stat = status_map[row['status']]
        y_status_global.append(stat)
        
        if stat == 0:
            y_grade_global.append(row['abschlussnote'])
            sample_weights_grade.append(1.0)
        else:
            y_grade_global.append(0.0)
            sample_weights_grade.append(0.0)
            
    y_status_global = np.array(y_status_global)
    y_grade_global = np.array(y_grade_global)
    sample_weights_grade = np.array(sample_weights_grade)
    
    idx = np.arange(len(y_grade))
    np.random.seed(42)
    np.random.shuffle(idx)
    
    split_idx = int(len(idx) * 0.8)
    tr_idx, te_idx = idx[:split_idx], idx[split_idx:]
    
    X_train = {'exam_history': X_hist[tr_idx], 'next_exam_context': X_ctx[tr_idx]}
    y_train = {
        'out_grade_local': y_grade[tr_idx],
        'out_pass_local': y_pass[tr_idx],
        'out_status_global': y_status_global[tr_idx],
        'out_grade_global': y_grade_global[tr_idx]
    }
    w_train = {
        'out_grade_local': np.ones(len(tr_idx)),
        'out_pass_local': np.ones(len(tr_idx)),
        'out_status_global': np.ones(len(tr_idx)),
        'out_grade_global': sample_weights_grade[tr_idx]
    }
    
    X_test = {'exam_history': X_hist[te_idx], 'next_exam_context': X_ctx[te_idx]}
    y_test = {
        'out_grade_local': y_grade[te_idx],
        'out_pass_local': y_pass[te_idx],
        'out_status_global': y_status_global[te_idx],
        'out_grade_global': y_grade_global[te_idx]
    }
    w_test = {
        'out_grade_local': np.ones(len(te_idx)),
        'out_pass_local': np.ones(len(te_idx)),
        'out_status_global': np.ones(len(te_idx)),
        'out_grade_global': sample_weights_grade[te_idx]
    }
    
    model = build_v37_multi_task_model(seq_timesteps=30, seq_features=X_hist.shape[2], context_features=X_ctx.shape[1])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'out_grade_local': 'mse',
            'out_pass_local': 'binary_crossentropy',
            'out_status_global': 'sparse_categorical_crossentropy',
            'out_grade_global': 'mse'
        },
        loss_weights={
            'out_grade_local': 1.0,
            'out_pass_local': 0.5,
            'out_status_global': 0.8,
            'out_grade_global': 1.0
        },
        metrics={
            'out_grade_local': ['mae'],
            'out_pass_local': ['accuracy'],
            'out_status_global': ['accuracy'],
            'out_grade_global': ['mae']
        }
    )
    
    model.fit(
        X_train, y_train, sample_weight=w_train,
        validation_data=(X_test, y_test, w_test),
        epochs=15, batch_size=256
    )

if __name__ == '__main__':
    main()
