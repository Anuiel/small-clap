import numpy as np
import torch
from torch import nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class AudioEncoder(nn.Module):
    def __init__(self, pretrain_name: str, sampling_rate: int = 16000) -> None:
        super().__init__()

        self.sampling_rate = sampling_rate
        self.normalizer: Wav2Vec2FeatureExtractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrain_name)
        self.model = Wav2Vec2Model.from_pretrained(pretrain_name)
        self.model.eval()
    
    def forward(self, audio: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            audio = torch.tensor(self.normalizer(audio, sampling_rate=self.sampling_rate).input_values, device=self.device)
            embeddings = self.model(audio).extract_features
            return embeddings
    
    @property
    def device(self) -> str:
        return next(iter(self.parameters())).device 


class AudioProjector(nn.Module):
    """
    Audio encoder based on Wav2Vec2 pretrained checkpoints

    Args:
        out_features: size of output embedding
        sample_rate: sample rate of input audio
        pretrain_name: hugging face name of Wav2Vec2 checkpoint
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features


        self.norm = nn.LayerNorm(in_features)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=1,
            bias=False,
            vdim=out_features,
            batch_first=True
        )
        self.relu = nn.PReLU()
        self.linear = nn.Linear(in_features, out_features)

    @staticmethod
    def padding_mask(audio_embedding_lenght: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_embedding_lenght: Tensor with shape [batch_size]
        Returns:
            mask: Tensor with shape [batch_size, 1]
        """
        max_len = audio_embedding_lenght.max()
        mask = torch.arange(max_len, device=audio_embedding_lenght.device).expand(len(audio_embedding_lenght), max_len)
        mask = mask < audio_embedding_lenght.unsqueeze(1)
        return mask

    def forward(self, audio_embedding: torch.Tensor, audio_embedding_lenght: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_embedding: Tensor with shape [batch_size, seq_len, in_features]
            audio_embedding_lenght: Tensor with shape [batch_size]
        Returns:
            out: Tensor with shape [batch_size, out_features]
        """
        mask = self.padding_mask(audio_embedding_lenght)
        audio_embedding = self.norm(audio_embedding)
        x = audio_embedding
        x, _ = self.mhsa(x, x, x, key_padding_mask=~mask)
        x = torch.mean(x, dim=1)
        x = self.linear(self.relu(x))
        return x
