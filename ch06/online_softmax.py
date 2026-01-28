
import torch


def standard_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]

    m = x[..., 0].clone()
    d = torch.ones_like(m)

    for i in range(1, n):
        m_new = torch.maximum(m, x[..., i])
        d = d * torch.exp(m - m_new) + torch.exp(x[..., i] - m_new)
        m = m_new

    result = torch.exp(x - m.unsqueeze(-1)) / d.unsqueeze(-1)
    return result


def online_softmax_with_output(
    x: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[-1]
    d_v = v.shape[-1]

    m = x[..., 0].clone()
    d = torch.ones_like(m)
    o = v[..., 0, :].clone()

    for i in range(1, n):
        m_new = torch.maximum(m, x[..., i])

        scale_old = torch.exp(m - m_new)
        scale_new = torch.exp(x[..., i] - m_new)

        d_new = d * scale_old + scale_new

        o = (o * d.unsqueeze(-1) * scale_old.unsqueeze(-1) +
             v[..., i, :] * scale_new.unsqueeze(-1)) / d_new.unsqueeze(-1)

        m = m_new
        d = d_new

    return o, d


def explain_online_softmax() -> str:
    return """
Online Softmax

Standard softmax requires two passes over data:
1. Find max for numerical stability
2. Compute exp and sum, then normalize

This is a problem for FlashAttention because we process tiles sequentially
and don't have access to the full row at once.

Online softmax maintains running statistics:
- m: running maximum
- d: running sum of exp(x - m)

Update rules when seeing new element x_i:
  m_new = max(m, x_i)
  d_new = d * exp(m - m_new) + exp(x_i - m_new)

The key insight: we can rescale previous sums when the max changes.

For attention output, we also maintain:
  o: running weighted sum

When max changes, we rescale both the denominator (d) and numerator (o).

This allows computing attention tile-by-tile:
1. Compute Q @ K^T for current tile
2. Update m, d, o with online softmax
3. No need to store full attention matrix
"""


def demonstrate_online_softmax():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    standard = standard_softmax(x)
    online = online_softmax(x)

    print("Standard softmax:", standard)
    print("Online softmax:  ", online)
    print("Difference:      ", (standard - online).abs().max().item())

    return standard, online


if __name__ == "__main__":
    print(explain_online_softmax())
    print("\n" + "=" * 60)
    print("Verification")
    print("-" * 60)
    demonstrate_online_softmax()

    print("\nBatch verification:")
    x = torch.randn(4, 8, 64)
    standard = standard_softmax(x)
    online = online_softmax(x)
    max_diff = (standard - online).abs().max().item()
    print(f"Max difference: {max_diff:.2e}")
