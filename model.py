import torch
from torch import nn
import torch.nn.functional as F
# nn.TransformerEncoder switches to a fused "fast path" whenever autograd is off or no parameter requires grad: the frozen prior,
# and all sampling under torch.no_grad(). On this model that path differs from the standard one by up to 0.47 in the logits
# (about 0.02 nats per token), so identical weights gave different log-probs for the agent (standard path) and the prior
# (fast path), and sampling used a slightly different distribution from the one the loss evaluates. The standard path matches a
# float64 reference to 2e-4, so it is used everywhere. Set before any forward pass; importing this module is enough.
torch.backends.mha.set_fastpath_enabled(False)
class LigandGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        nhead=12,
        num_layers=12,
        dim_feedforward=3072,
        dropout=0.1,
        activation=nn.GELU(),
        layer_norm_eps=1e-05,
        batch_first=True,
        norm_first=True,
        transformer_bias=False,
        numhead_bias=True,
        context_length=2048,
    ):
        super().__init__()
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.encoder_stack = nn.TransformerEncoder(
            encoder_layer=encoder, num_layers=num_layers, enable_nested_tensor=False
        )
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(context_length, d_model)
        self.lm_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=transformer_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, vocab_size, bias=transformer_bias),
        )
        self.num_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias=numhead_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1, bias=numhead_bias),
        )
    def forward(self, x, x_num):
        x = self.token_embed(x) * x_num.unsqueeze(-1)
        x = x + self.position_embed.weight[: x.shape[1]].unsqueeze(0)
        seq_len = x.shape[1]
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()     
        x = self.encoder_stack(x, mask=causal_mask, is_causal=True)
        logit_preds = self.lm_head(x)
        num_preds = self.num_head(x)
        return logit_preds, num_preds