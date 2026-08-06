from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, output_dim: int, frequency_dim: int = 128):
        super().__init__()
        if frequency_dim % 2:
            raise ValueError("frequency_dim must be even")
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), frequency_dim // 2)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * time[:, None].float() * self.frequencies[None]
        return self.mlp(torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1))


class SelfAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, width = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        attention_mask = None
        if mask is not None:
            attention_mask = mask[:, None, None, :]
        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        result = result.transpose(1, 2).reshape(batch, tokens, width)
        return self.output(result)


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class FlowBlock(nn.Module):
    def __init__(self, width: int, heads: int, hidden: int):
        super().__init__()
        self.attention_norm = nn.RMSNorm(width, eps=1.0e-6)
        self.ffn_norm = nn.RMSNorm(width, eps=1.0e-6)
        self.attention_condition = nn.Linear(width, width, bias=False)
        self.ffn_condition = nn.Linear(width, width, bias=False)
        self.attention = SelfAttention(width, heads)
        self.ffn = SwiGLU(width, hidden)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_input = self.attention_norm(x) + self.attention_condition(condition)[:, None]
        x = x + self.attention(attention_input, mask)
        ffn_input = self.ffn_norm(x) + self.ffn_condition(condition)[:, None]
        x = x + self.ffn(ffn_input)
        if mask is not None:
            x = x * mask[..., None]
        return x


class CoilFlowTransformer(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int = 100,
        width: int = 512,
        layers: int = 8,
        heads: int = 8,
        hidden: int = 1408,
        max_nfp: int = 16,
    ):
        super().__init__()
        self.config = {
            "token_dim": token_dim,
            "width": width,
            "layers": layers,
            "heads": heads,
            "hidden": hidden,
            "max_nfp": max_nfp,
        }
        self.input = nn.Linear(token_dim, width)
        self.time_embedding = TimeEmbedding(width)
        self.nfp_embedding = nn.Embedding(max_nfp + 1, width)
        self.blocks = nn.ModuleList(
            [FlowBlock(width, heads, hidden) for _ in range(layers)]
        )
        self.final_norm = nn.RMSNorm(width, eps=1.0e-6)
        self.output = nn.Linear(width, token_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        time: torch.Tensor,
        nfp: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.validate_nfp(nfp)
        return self.forward_unchecked(tokens, time, nfp, mask)

    def validate_nfp(self, nfp: torch.Tensor) -> None:
        if torch.any((nfp < 1) | (nfp > self.config["max_nfp"])):
            raise ValueError("nfp is outside the model embedding range")

    def forward_unchecked(
        self,
        tokens: torch.Tensor,
        time: torch.Tensor,
        nfp: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self.time_embedding(time) + self.nfp_embedding(nfp)
        x = self.input(tokens)
        if mask is not None:
            x = x * mask[..., None]
        for block in self.blocks:
            x = block(x, condition, mask)
        output = self.output(self.final_norm(x))
        return output if mask is None else output * mask[..., None]

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class _UncheckedFlowTransformer(nn.Module):
    def __init__(self, model: CoilFlowTransformer):
        super().__init__()
        self.model = model

    def forward(
        self,
        tokens: torch.Tensor,
        time: torch.Tensor,
        nfp: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.forward_unchecked(tokens, time, nfp, mask)


class CompiledFlowTransformer(nn.Module):
    def __init__(self, model: CoilFlowTransformer):
        super().__init__()
        self.model = model
        self.compiled_unchecked = torch.compile(
            _UncheckedFlowTransformer(model), mode="default", fullgraph=True
        )

    def validate_nfp(self, nfp: torch.Tensor) -> None:
        self.model.validate_nfp(nfp)

    def forward_unchecked(
        self,
        tokens: torch.Tensor,
        time: torch.Tensor,
        nfp: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.compiled_unchecked(tokens, time, nfp, mask)

    def forward(
        self,
        tokens: torch.Tensor,
        time: torch.Tensor,
        nfp: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.validate_nfp(nfp)
        return self.forward_unchecked(tokens, time, nfp, mask)


def compile_flow_transformer(model: CoilFlowTransformer) -> CompiledFlowTransformer:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return CompiledFlowTransformer(model)
