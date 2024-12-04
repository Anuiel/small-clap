import torch


class KNNMetric(torch.nn.Module):
    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.k = k

    def forward(self, query: list[tuple[int, torch.Tensor]], key: list[tuple[int, torch.Tensor]]) -> float:
        assert len(query) == len(key)
        query_embed = torch.stack([item[1] for item in query])
        query_embed = torch.nn.functional.normalize(query_embed, dim=-1)
        key_embed = torch.stack([item[1] for item in key])
        key_embed = torch.nn.functional.normalize(key_embed, dim=-1)

        simulairty_logits = query_embed @ key_embed.T

        macro_acc = 0.
        for item, logits in zip(query, simulairty_logits):
            query_id = item[0]
            indices = logits.argsort(descending=True)[1:self.k + 1]
            keys_id = torch.tensor([key[idx][0] for idx in indices])
            acc = (keys_id == query_id).float().mean().item()
            macro_acc += acc
        macro_acc /= len(query)
        return macro_acc
