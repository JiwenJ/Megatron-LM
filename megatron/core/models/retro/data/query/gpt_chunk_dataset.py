# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import torch

from megatron.core.models.retro.data.utils import get_num_chunks_per_sample

from .utils import get_neighbor_dir


class GPTChunkDataset(torch.utils.data.Dataset):
    '''Pretraining chunk dataset wraps a standard GPT dataset.

    This dataset conceptually divides each sample (e.g., length 2048)
    into chunks (e.g., length 64) and restructures them into a list of
    chunks (e.g., length num_samples * num_chunks_per_sample).
    '''

    def __init__(self, sample_dataset: GPTDataset, sample_length: int, chunk_length: int):

        super().__init__()

        self.sample_dataset = sample_dataset
        self.chunk_length = chunk_length
        self.n_chunks_per_sample = get_num_chunks_per_sample(sample_length, chunk_length)
        self.n_samples = len(sample_dataset)
        self.n_chunks = self.n_samples * self.n_chunks_per_sample

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> dict:

        # Convert global chunk index to global sample index & local chunk index.
        sample_idx = idx // self.n_chunks_per_sample
        chunk_idx = idx % self.n_chunks_per_sample

        # Extract sample data.
        sample = self.sample_dataset[sample_idx]
        sample_token_ids = sample["text"]
        sample_doc_ids = sample["document_ids"]

        # Chunk start/end token idxs.
        token_start_idx = chunk_idx * self.chunk_length
        token_end_idx = token_start_idx + self.chunk_length
        chunk_token_ids = sample_token_ids[token_start_idx:token_end_idx]

        # Sample.
        return {
            "doc_ids": sample_doc_ids,
            "text": chunk_token_ids,
        }


def build_gpt_chunk_datasets_from_gpt_datasets(
    project_dir: str, gpt_datasets: dict, sample_length: int, chunk_length: int,
) -> dict:
    '''Get train, valid, test GPT chunk datasets.'''

    # Reset iteration.
    # >>>
    # config.iteration = 0
    # config.consumed_train_samples = 0
    # <<<

    # GPT chunk datasets.
    chunk_datasets = {
        key: {
            "dataset": GPTChunkDataset(sample_ds, sample_length, chunk_length),
            "neighbor_dir": get_neighbor_dir(project_dir, key, sample_ds),
            "num_active_chunks": num_active_samples
            * get_num_chunks_per_sample(sample_length, chunk_length),
        }
        if sample_ds
        else None
        for key, (sample_ds, num_active_samples) in gpt_datasets.items()
    }

    return chunk_datasets
