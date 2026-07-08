import jax
import jax.numpy as jnp
import flax.linen as nn

class TransformerEncoder(nn.Module):
    num_heads: int = 4
    qkv_features: int = 64
    mlp_dim: int = 128
    
    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[1]
        pos_emb = self.param('pos_emb', nn.initializers.normal(stddev=0.02), (1, seq_len, x.shape[2]))
        x = x + pos_emb
        
        for _ in range(2):
            attn_out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.qkv_features)(x, x)
            x = x + attn_out
            x = nn.LayerNorm()(x)
            
            mlp_out = nn.Sequential([
                nn.Dense(self.mlp_dim), nn.relu,
                nn.Dense(x.shape[-1])
            ])(x)
            x = x + mlp_out
            x = nn.LayerNorm()(x)
            
        return jnp.mean(x, axis=1)

class TargetActorCritic(nn.Module):
    action_dim: int = 2
    
    @nn.compact
    def __call__(self, obs_dict):
        current = obs_dict["current"]
        history = obs_dict["history"]
        
        hist_feat = TransformerEncoder(num_heads=2, qkv_features=32, mlp_dim=64)(history)
        
        curr_feat = nn.Sequential([
            nn.Dense(64), nn.relu,
            nn.Dense(64), nn.relu
        ])(current)
        
        fused = jnp.concatenate([hist_feat, curr_feat], axis=-1)
        
        x = nn.Sequential([
            nn.Dense(128), nn.relu,
            nn.Dense(128), nn.relu
        ])(fused)
        
        action_mean = nn.Dense(self.action_dim)(x)
        action_logstd = self.param('logstd', nn.initializers.zeros, (self.action_dim,))
        value = nn.Dense(1)(x)
        return action_mean, action_logstd, jnp.squeeze(value, axis=-1)

import numpy as np
import torch
import torch.nn as pt_nn
import torch.nn.functional as F
import os

class PyTorchMHA(pt_nn.Module):
    def __init__(self, num_heads, qkv_features, in_features):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_features = qkv_features
        
        self.q_proj = pt_nn.Linear(in_features, num_heads * qkv_features)
        self.k_proj = pt_nn.Linear(in_features, num_heads * qkv_features)
        self.v_proj = pt_nn.Linear(in_features, num_heads * qkv_features)
        self.out_proj = pt_nn.Linear(num_heads * qkv_features, in_features)
        
    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.qkv_features).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.qkv_features).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.qkv_features).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.qkv_features ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, self.num_heads * self.qkv_features)
        return self.out_proj(out)

class PyTorchTransformerEncoder(pt_nn.Module):
    def __init__(self, seq_len=25, in_dim=3, num_heads=2, qkv_features=16, mlp_dim=64):
        super().__init__()
        self.pos_emb = pt_nn.Parameter(torch.zeros(1, seq_len, in_dim))
        
        self.mha_0 = PyTorchMHA(num_heads, qkv_features, in_dim)
        self.ln_0 = pt_nn.LayerNorm(in_dim)
        self.mlp_0 = pt_nn.Sequential(pt_nn.Linear(in_dim, mlp_dim), pt_nn.ReLU(), pt_nn.Linear(mlp_dim, in_dim))
        self.ln_1 = pt_nn.LayerNorm(in_dim)
        
        self.mha_1 = PyTorchMHA(num_heads, qkv_features, in_dim)
        self.ln_2 = pt_nn.LayerNorm(in_dim)
        self.mlp_1 = pt_nn.Sequential(pt_nn.Linear(in_dim, mlp_dim), pt_nn.ReLU(), pt_nn.Linear(mlp_dim, in_dim))
        self.ln_3 = pt_nn.LayerNorm(in_dim)
        
    def forward(self, x):
        x = x + self.pos_emb
        x = self.ln_0(x + self.mha_0(x))
        x = self.ln_1(x + self.mlp_0(x))
        x = self.ln_2(x + self.mha_1(x))
        x = self.ln_3(x + self.mlp_1(x))
        return torch.mean(x, dim=1)

class PyTorchTargetActorCritic(pt_nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_encoder = PyTorchTransformerEncoder()
        self.curr_feat = pt_nn.Sequential(pt_nn.Linear(6, 64), pt_nn.ReLU(), pt_nn.Linear(64, 64), pt_nn.ReLU())
        self.fused = pt_nn.Sequential(pt_nn.Linear(3 + 64, 128), pt_nn.ReLU(), pt_nn.Linear(128, 128), pt_nn.ReLU())
        self.action_mean = pt_nn.Linear(128, 2)
        
    def forward(self, current, history):
        hist_f = self.hist_encoder(history)
        curr_f = self.curr_feat(current)
        f = torch.cat([hist_f, curr_f], dim=-1)
        x = self.fused(f)
        return self.action_mean(x)

def copy_dense(pt_linear, flax_dict):
    pt_linear.weight.data.copy_(torch.from_numpy(np.array(flax_dict['kernel']).T))
    pt_linear.bias.data.copy_(torch.from_numpy(np.array(flax_dict['bias'])))

def copy_mha(pt_mha, flax_dict):
    for name, pt_layer in [('query', pt_mha.q_proj), ('key', pt_mha.k_proj), ('value', pt_mha.v_proj)]:
        w = np.array(flax_dict[name]['kernel']).reshape(pt_layer.in_features, -1).T
        b = np.array(flax_dict[name]['bias']).flatten()
        pt_layer.weight.data.copy_(torch.from_numpy(w))
        pt_layer.bias.data.copy_(torch.from_numpy(b))
    
    w_out = np.array(flax_dict['out']['kernel']).reshape(-1, pt_mha.out_proj.out_features).T
    b_out = np.array(flax_dict['out']['bias'])
    pt_mha.out_proj.weight.data.copy_(torch.from_numpy(w_out))
    pt_mha.out_proj.bias.data.copy_(torch.from_numpy(b_out))

def copy_ln(pt_ln, flax_dict):
    pt_ln.weight.data.copy_(torch.from_numpy(np.array(flax_dict['scale'])))
    pt_ln.bias.data.copy_(torch.from_numpy(np.array(flax_dict['bias'])))

def export_to_onnx(flax_params, save_dir):
    pt_model = PyTorchTargetActorCritic()
    
    p = flax_params['params']
    
    copy_dense(pt_model.curr_feat[0], p['Dense_0'])
    copy_dense(pt_model.curr_feat[2], p['Dense_1'])
    
    copy_dense(pt_model.fused[0], p['Dense_2'])
    copy_dense(pt_model.fused[2], p['Dense_3'])
    
    copy_dense(pt_model.action_mean, p['Dense_4'])
    
    tf_p = p['TransformerEncoder_0']
    pt_tf = pt_model.hist_encoder
    
    pt_tf.pos_emb.data.copy_(torch.from_numpy(np.array(tf_p['pos_emb'])))
    
    copy_mha(pt_tf.mha_0, tf_p['MultiHeadDotProductAttention_0'])
    copy_ln(pt_tf.ln_0, tf_p['LayerNorm_0'])
    copy_dense(pt_tf.mlp_0[0], tf_p['Dense_0'])
    copy_dense(pt_tf.mlp_0[2], tf_p['Dense_1'])
    copy_ln(pt_tf.ln_1, tf_p['LayerNorm_1'])
    
    copy_mha(pt_tf.mha_1, tf_p['MultiHeadDotProductAttention_1'])
    copy_ln(pt_tf.ln_2, tf_p['LayerNorm_2'])
    copy_dense(pt_tf.mlp_1[0], tf_p['Dense_2'])
    copy_dense(pt_tf.mlp_1[2], tf_p['Dense_3'])
    copy_ln(pt_tf.ln_3, tf_p['LayerNorm_3'])
    
    pt_model.eval()
    dummy_curr = torch.zeros(1, 6, dtype=torch.float32)
    dummy_hist = torch.zeros(1, 25, 3, dtype=torch.float32)
    
    onnx_path = os.path.join(save_dir, "target_model_3d.onnx")
    try:
        torch.onnx.export(
            pt_model, (dummy_curr, dummy_hist), onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['current', 'history'],
            output_names=['action_mean'],
            dynamic_axes={'current': {0: 'batch_size'}, 'history': {0: 'batch_size'}, 'action_mean': {0: 'batch_size'}}
        )
        print(f"✅ 已成功導出 ONNX 模型至: {onnx_path}")
    except Exception as e:
        print(f"⚠️ ONNX 模型導出失敗: {e}")
