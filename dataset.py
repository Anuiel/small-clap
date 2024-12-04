from pathlib import Path
import typing as tp
import hashlib

import torch
import pandas as pd 
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def calculate_hash(input_string: str) -> str:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_string.encode('utf-8'))
    return sha256_hash.hexdigest()


class AudiocapsDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        part: tp.Literal['train', 'val', 'val_new', 'test', 'test_new'],
        audio_encoder: tp.Callable[[np.ndarray], torch.Tensor],
        text_encoder: tp.Callable[[str], torch.Tensor],
        limit: int | None = None,
        max_duration: float = 11.
    ) -> None:
        super().__init__()

        self.dataset_dir: Path = Path(dataset_dir)
        self.part = part
        self.df = pd.read_csv(self.dataset_dir / f'audiocaps_{part}.tsv', delimiter='\t')
        if limit:
            self.df = self.df.iloc[:limit]
        
        self.df['audio'] = f'{self.dataset_dir.parent}/' + self.df['audio']
        self.df = self.df[self.df['duration'] <= max_duration]
        
        self._precompute_embeddings(
            dataset_dir=self.dataset_dir,
            part=part,
            audio_encoder=audio_encoder,
            text_encoder=text_encoder
        )

    def _precompute_embeddings(
        self,
        dataset_dir: Path,
        part: str,
        audio_encoder: tp.Callable[[np.ndarray], torch.Tensor],
        text_encoder: tp.Callable[[str], torch.Tensor]
    ) -> None:
        audio_embeddings_prefix = dataset_dir / 'audio_embedding'
        audio_embeddings_prefix.mkdir(exist_ok=True, parents=True)

        audio_embeddings_path = audio_embeddings_prefix / part
        self._precompute_audio_embeddings(audio_embeddings_path, audio_encoder)

        text_embeddings_prefix = dataset_dir / 'text_embedding'
        text_embeddings_prefix.mkdir(exist_ok=True, parents=True)
        text_embeddings_path = text_embeddings_prefix / part
        self._precompute_text_embeddings(text_embeddings_path, text_encoder)

    def _precompute_audio_embeddings(
        self,
        embeddings_dir: Path,
        encoder: tp.Callable[[np.ndarray], torch.Tensor]
    ) -> None:
        if embeddings_dir.exists():
            print('Audio embeddings exists.')
            return
        embeddings_dir.mkdir(parents=True)
        print('Precomputing audio embeddings...')
        for audio_path in tqdm(self.df['audio']):
            audio_path = Path(audio_path)
            save_path = embeddings_dir / (audio_path.stem + '.npy')
            if save_path.exists():
                continue
            audio = self.load_audio(audio_path)
            audio_embedding = encoder(audio)
            np.save(save_path, audio_embedding.cpu())

    def _precompute_text_embeddings(
        self,
        embeddings_dir: Path,
        encoder: tp.Callable[[str], torch.Tensor]
    ) -> None:
        if embeddings_dir.exists():
            print('Text embeddings exists.')
            return
        embeddings_dir.mkdir(parents=True)
        print('Precomputing text embeddings...')
        for text in tqdm(self.df['text']):
            text_embedding = encoder(text)
            text_hash = calculate_hash(text)
            save_path = embeddings_dir / text_hash
            np.save(save_path, text_embedding.cpu())

    @staticmethod
    def load_audio(path: Path) -> np.ndarray:
        audio, _ = torchaudio.load(path)
        audio = np.array(audio[0, :])
        return audio
    
    @staticmethod
    def load_npy(path: Path) -> np.ndarray:
        return np.load(path)[0, ...]

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> dict[str, tp.Any]:
        item = self.df.iloc[index]
        path = Path(item['audio'])
        audio, _ = torchaudio.load(path)
        audio = np.array(audio[0, :])

        stem = path.stem
        text_hash = calculate_hash(item['text'])
        audio_embedding = self.load_npy(self.dataset_dir / 'audio_embedding' / self.part / f"{stem}.npy")
        text_embedding = self.load_npy(self.dataset_dir / 'text_embedding' / self.part / f"{text_hash}.npy")

        out = dict(audio=audio, text=item['text'], path=item['audio'], audio_embedding=audio_embedding, text_embedding=text_embedding)
        return out
