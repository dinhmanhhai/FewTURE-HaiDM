import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnHead(nn.Module):
    """
    Cross-attention head cho few-shot:
    - Query tokens (mỗi ảnh query) làm Q
    - Support tokens theo từng lớp làm K,V
    - Pool theo patch để lấy 1 vector/score cho mỗi lớp
    - Trả về logits [B_query, n_way]
    Ghi chú: Module này khởi tạo 'lazy' theo d_model khi batch đầu tiên đi qua.
    """

    def __init__(self, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.mha = None  # lazy init với d_model
        self.proj_cls = None  # Linear(d_model -> 1)
        # log temperature để ổn định thang logits
        self.log_tau = nn.Parameter(torch.tensor([0.0]))

    def _lazy_build(self, d_model: int, device: torch.device):
        if self.mha is None:
            self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=self.n_heads,
                                             dropout=self.dropout, batch_first=True).to(device)
            self.proj_cls = nn.Linear(d_model, 1, bias=True).to(device)

    def forward(self,
                emb_support: torch.Tensor,
                emb_query: torch.Tensor,
                label_support: torch.Tensor,
                n_way: int) -> torch.Tensor:
        """
        emb_support: [N_sup, T, D]  (tokens patch-level của tất cả ảnh support)
        emb_query:   [N_q,  T, D]  (tokens patch-level của tất cả ảnh query)
        label_support: [N_sup] (0..n_way-1), theo thứ tự aabbcc...
        n_way: số lớp trong episode
        """
        Bq, Tq, D = emb_query.shape
        device = emb_query.device
        self._lazy_build(D, device)
        # Gom support theo lớp: danh sách độ dài n_way, mỗi phần tử [Nk*T, D]
        class_kv = []
        for c in range(n_way):
            idx = (label_support == c).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                # phòng vệ: lớp rỗng
                class_kv.append(torch.zeros(1, D, device=device).unsqueeze(0))
                continue
            tokens_c = emb_support[idx]  # [Nk, T, D]
            kv_c = tokens_c.reshape(-1, D)  # [Nk*T, D]
            class_kv.append(kv_c.unsqueeze(0))  # thêm batch dim 1 để broadcast theo query

        # Chuẩn bị Q: [Bq, Tq, D]
        Q = emb_query  # batch_first

        scores = []
        tau = torch.exp(self.log_tau) + 1e-8
        for c in range(n_way):
            KcVc = class_kv[c]  # [1, Tk, D]
            # Lặp lại theo batch query để batch hóa attention
            K = KcVc.expand(Bq, -1, -1)
            V = K
            # Multi-head attention: out [Bq, Tq, D]
            out, _ = self.mha(Q, K, V, need_weights=False)
            # Pool theo patch → [Bq, D]
            pooled = out.mean(dim=1)
            # Chiếu sang scalar rồi chuẩn hóa theo temperature
            score_c = self.proj_cls(pooled).squeeze(-1) / tau
            scores.append(score_c)

        logits = torch.stack(scores, dim=-1)  # [Bq, n_way]
        return logits


