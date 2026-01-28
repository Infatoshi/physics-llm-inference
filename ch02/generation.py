import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '..')
from ch01.transformer import TransformerModel


def naive_generate(
    model: TransformerModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    model.eval()
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, indices = torch.topk(next_logits, top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, indices, values)

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

    return generated
