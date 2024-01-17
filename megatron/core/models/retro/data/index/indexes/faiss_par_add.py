# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Multi-process & multi-node version of Faiss's index.add().

This class inherits from FaissBaseIndex, and optimizes the 'add()' method by
making it multi-node and multi-process, with bit-wise equivalence to
FaissBaseIndex. This allows 'add()' to scale out to very large datasets, since
the vast majority of the computational effort is embarrassingly parallel.
"""

import numpy as np
import os
import psutil
import shutil
import torch
from tqdm import tqdm
from typing import Tuple

from megatron.core.models.retro.data.config import Embedder, RetroPreprocessingConfig
from megatron.core.models.retro.data.external_libs import faiss, h5py
from megatron.core.models.retro.data.index.utils import get_added_code_paths, get_added_codes_dir
from megatron.core.models.retro.data.utils import get_blocks_by_rank, print_rank_0, retro_makedir, GPTToTextDataset

from .faiss_base import FaissBaseIndex


class FaissParallelAddIndex(FaissBaseIndex):
    '''
    This class parallelizes both 1) encoding vectors, and 2) adding codes to the
    index. This class is more performant than naive use of Faiss, because most
    of the computational work is in encoding the vectors, which is an
    embarassingly parallel operation.
    '''

    def encode_block(self, index: faiss.Index, embedder: Embedder, text_dataset: GPTToTextDataset, block: dict) -> Tuple[np.ndarray, np.ndarray]:
        '''Encode sub-dataset block, to be later added to index.

        Encode the data subset, generally in blocks of 1M vectors each. For
        each block, the empty/trained index is loaded, codes are computed
        via index.sa_encode(), and the resulting codes are saved to disk.
        '''

        # Embed block.
        embeddings = self.embed_text_dataset_block(embedder, text_dataset, block["range"],)

        # Encode block.
        print_rank_0("encode.")
        codes = index.sa_encode(embeddings)

        # Return embeddings for validation purposes.
        return embeddings, codes

    def save_block(self, config: RetroPreprocessingConfig, block: dict, codes: np.ndarray) -> None:
        '''Save block of codes to disk.'''
        # Save neighbors.
        print_rank_0("save codes.")
        retro_makedir(config, os.path.dirname(block["path"]))
        with h5py.File(block["path"], "w") as f:
            f.create_dataset("data", data=codes)

    def encode(self, config: RetroPreprocessingConfig, text_dataset: GPTToTextDataset) -> None:
        '''Encode text dataset, to be later added to index.'''

        codes_dir = get_added_codes_dir(config)
        retro_makedir(config, codes_dir)

        # Index.
        index = self.get_empty_index(config)

        # Bert embedder.
        embedder = config.retro_bert_embedders.mem

        # Missing code blocks.
        def validate(f: h5py.File) -> None:
            assert len(f["data"].shape) == 2

        blocks = get_blocks_by_rank(
            codes_dir, len(text_dataset), config.retro_block_size, validate=validate,
        )

        # Encode each block.
        for block_index, block in enumerate(blocks.missing):

            if block is not None:

                # Progress.
                print_rank_0(
                    "encode block %d / %d ... %s."
                    % (block_index, len(blocks.missing), block["path"],)
                )

                # Encode and save.
                _, codes = self.encode_block(index, embedder, text_dataset, block)
                self.save_block(config, block, codes)

            # Synchronize progress across all ranks. (for easier observation)
            print_rank_0(" > waiting for other ranks to finish block.")
            torch.distributed.barrier()

    def add_codes(self, config: RetroPreprocessingConfig) -> None:
        '''Read codes from disk, and add them to the index.'''

        if torch.distributed.get_rank() != 0:
            return

        added_index_path = self.get_added_index_path(config)
        if os.path.exists(added_index_path):
            return

        # Index.
        print_rank_0("read empty index.")
        index = self.get_empty_index(config)
        index_ivf = faiss.extract_index_ivf(index)

        # Add codes.
        print_rank_0("add codes.")
        code_paths = get_added_code_paths(config)
        pbar = tqdm(code_paths)
        for code_path in pbar:
            pbar.set_description(
                "add codes, mem %.3f gb, %.1f%%"
                % (psutil.virtual_memory()[3] / 1024 ** 3, psutil.virtual_memory()[2],)
            )
            with h5py.File(code_path) as f:

                nload = int(config.retro_index_add_load_fraction * f["data"].shape[0])
                offset = int(os.path.basename(code_path).split("-")[0])
                xids = np.arange(offset, offset + nload)
                codes = np.copy(f["data"][:nload])
                index_ivf.add_sa_codes(codes, xids)

        # Update index's ntotal.
        index.ntotal = index_ivf.ntotal

        # Write index.
        print_rank_0("write added index.")
        faiss.write_index(index, added_index_path)

    def remove_codes(self, config: RetroPreprocessingConfig) -> None:
        '''Remove added codes after adding to index.'''
        if torch.distributed.get_rank() != 0:
            return
        assert os.path.isfile(self.get_added_index_path(config))

        if config.retro_index_delete_added_codes:
            raise Exception("remove?")
            shutil.rmtree(get_added_codes_dir(config), ignore_errors=True)

    def add(self, config: RetroPreprocessingConfig, text_dataset: GPTToTextDataset) -> None:

        # Encode chunks.
        self.encode(config, text_dataset)

        # Add codes to index.
        self.add_codes(config)

        # Wait for (single-process) adding to complete.
        torch.distributed.barrier()

        # Remove codes.
        self.remove_codes(config)
