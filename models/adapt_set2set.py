import torch
import torch.nn as nn


class SetToSetAdapter(nn.Module):
    """
    FEAT-style set-to-set adapter (tối giản):
    - Self-attn encoder cho toàn bộ support tokens để làm mượt biểu diễn (có thể coi như FEAT encoder).
    - Cross-attn cập nhật query từ support đã thích nghi.
    API: forward(support_emb, query_emb) -> (support_emb_adapt, query_emb_adapt)
    """

    def __init__(self, d_model: int, n_layers: int = 1, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=4 * d_model,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cross = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                           dropout=dropout, batch_first=True)

    def forward(self, support_emb: torch.Tensor, query_emb: torch.Tensor):
        """
        support_emb: [N_sup, T, D]
        query_emb:   [N_q,  T, D]
        """
        # Flatten theo mẫu cho encoder (không cần theo lớp ở đây)
        sup_flat = support_emb  # [N_sup, T, D]
        sup_adapt = self.encoder(sup_flat)  # [N_sup, T, D]

        # Dùng toàn bộ tokens support làm K,V (đơn giản hoá); có thể thay bằng theo-lớp nếu cần
        K = sup_adapt.reshape(-1, sup_adapt.shape[-1]).unsqueeze(0).expand(query_emb.shape[0], -1, -1)
        V = K
        Q = query_emb
        q_adapt, _ = self.cross(Q, K, V, need_weights=False)
        return sup_adapt, q_adapt


