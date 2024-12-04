import torch
from transformers import CLIPModel, CLIPTokenizer


class TextEncoder(torch.nn.Module):
    def __init__(
        self,
        pretrain_name: str = "openai/clip-vit-base-patch32"
    ) -> None:
        super().__init__()

        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(pretrain_name)
        self.model = CLIPModel.from_pretrained(pretrain_name)
    
    def forward(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            tokenized = torch.tensor(self.tokenizer(text).input_ids, device=self.device).unsqueeze(0)
            return self.model.get_text_features(tokenized)

    @property
    def device(self) -> str:
        return next(iter(self.parameters())).device
