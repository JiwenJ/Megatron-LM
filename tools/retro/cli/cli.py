# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import json
import numpy as np
import os
# import torch
import typing as T
from types import SimpleNamespace

# >>>
from megatron.core.models.retro.data.db.dataset import DBDataset
from megatron.core.models.retro.data.db.utils import (
    get_indexed_dataset_infos as get_db_indexed_dataset_infos,
    get_merged_train_dataset as get_db_dataset,
)
# from megatron.core.models.retro.data.main import add_retro_args
from megatron.core.models.retro.data.query.retro_dataset import get_retro_datasets, RetroDataset
from megatron.core.models.retro.data.utils import get_config_path
# from megatron.global_vars import set_global_variables # , set_retro_args
# from megatron.initialize import (
#     initialize_megatron,
#     _initialize_distributed,
#     _set_random_seed,
#     _compile_dependencies,
# )
from tools.retro.preprocess_data import get_tokenizers
# from tools.retro.preprocess_data import get_gpt_tokenizer
# <<<

# >>>
from lutil import pax
# <<<


def shorten_str(s: str, n: int) -> str:
    s = "\\n".join(s.splitlines())
    return s if len(s) <= n else "%s ... %s" % (s[: n // 2], s[-n // 2 :])


class retro:

    args = None

    ##############################################
    # initialize.
    ##############################################

    # @classmethod
    # def parse_dtype_str(cls, dtype_str: str) -> type:
    #     return {
    #         "torch.float16": torch.float16,
    #         "torch.float32": torch.float32,
    #         "torch.bfloat16": torch.bfloat16,
    #     }[dtype_str]

    # @classmethod
    # def init_megatron(cls, project_dir: str) -> None:
    #     '''Custom initialization of Megatron.'''

    #     # Load config.
    #     config_path = get_config_path(project_dir)
    #     assert os.path.exists(config_path), "config.json not found in project dir."
    #     with open(config_path) as f:
    #         cls.args = SimpleNamespace(**json.load(f))
    #         cls.args.retro_project_dir = project_dir  # just in case project_dir moved
    #         cls.args.rank = 0  # override env
    #         cls.args.world_size = 1  # override env
    #         # >>>
    #         # cls.args.params_dtype = cls.parse_dtype_str(cls.args.params_dtype)
    #         # pax({"verify": cls.args.no_retro_verify_neighbor_count})
    #         # cls.args.retro_verify_neighbor_count = False
    #         # <<<
    #         # >>>
    #         # cls.args.rampup_batch_size = None
    #         # <<<

    #     set_global_variables(cls.args)
    #     set_retro_args(cls.args)
    #     _initialize_distributed()
    #     _set_random_seed(cls.args.seed, cls.args.data_parallel_random_init)
    #     _compile_dependencies()
    @classmethod
    def load_config(cls, project_dir: str) -> None:
        '''Load Retro's config.json.'''
        config_path = get_config_path(project_dir)
        assert os.path.exists(config_path), "config.json not found in project dir."
        with open(config_path) as f:
            return SimpleNamespace(**json.load(f))

    @classmethod
    def init(cls, project_dir: str) -> None:
        '''Initialize Megatron, tokenizers, and datasets.'''

        # Load args.
        # cls.init_megatron(project_dir)
        cls.config = cls.load_config(project_dir)
        cls.config.retro_project_dir = project_dir
        # >>>
        # cls.config.retro_bert_tokenizer_type = "BertWordPieceLowerCase"
        # cls.config.retro_bert_vocab_file = os.path.join(
        #     project_dir,
        #     "tokenizer/bert-large-uncased-vocab.txt",
        # )
        # <<<
        cls.tokenizers = get_tokenizers(cls.config)
        # cls.gpt_tokenizer = get_gpt_tokenizer(cls.config)
        
        # >>>
        # pax({
        #     "config" : cls.config,
        #     "tokenizers" : cls.tokenizers,
        #     # "gpt_tokenizer" : cls.gpt_tokenizer,
        # })
        # <<<

        # Load data.
        cls.db_indexed_dataset_infos = get_db_indexed_dataset_infos(project_dir)
        # >>>
        # pax({f"db_indexed_dataset_infos / {i}":d for i,d in enumerate(cls.db_indexed_dataset_infos)})
        # <<<
        cls.db_dataset = get_db_dataset()
        # >>>
        pax({"db_dataset": cls.db_dataset})
        # <<<
        pt_train_ds, pt_valid_ds, pt_test_ds = get_retro_datasets()
        cls.pt_datasets = SimpleNamespace(
            train=pt_train_ds,
            valid=pt_valid_ds,
            test=pt_test_ds,
        )

        # Retrieve max saved neighbors.
        for key in vars(cls.pt_datasets):
            getattr(cls.pt_datasets, key).num_neighbors = cls.args.retro_query_num_neighbors_save

        # Print usage.
        cls.print_usage()

    ##############################################
    # utils.
    ##############################################

    @classmethod
    def gpt_to_text(cls, token_ids: np.ndarray) -> str:
        '''GPT tokens to text.'''
        return cls.tokenizers.gpt.detokenize(
            token_ids.tolist() if isinstance(token_ids, np.ndarray) else token_ids
        )

    @classmethod
    def text_to_bert(cls, text: str) -> np.ndarray:
        '''Text to Bert tokens.'''
        return cls.tokenizers.bert.tokenize(text)

    ##############################################
    # chunk db.
    ##############################################

    @classmethod
    def get_db_num_indexed_datasets(cls) -> int:
        '''Number of indexed datasets within blended dataset.'''
        return len(cls.db_indexed_dataset_infos)

    @classmethod
    def get_db_indexed_dataset_infos(cls) -> T.List[T.Tuple[float, str]]:
        '''Dataset infos, including number of training & sampled sets.'''
        return [(info["ratio"], info["name"]) for info in cls.db_indexed_dataset_infos]

    @classmethod
    def get_db_dataset(cls) -> DBDataset:
        return cls.db_dataset

    @classmethod
    def get_db_num_chunks(cls) -> int:
        '''Number of DB chunks.'''
        return len(cls.get_db_dataset())

    @classmethod
    def get_db_chunk_gpt(cls, idx: int) -> T.List[int]:
        '''Get DB chunk as GPT token ids.'''
        return cls.get_db_dataset()[idx]["text"].tolist()

    @classmethod
    def get_db_chunk_bert(cls, idx: int) -> T.List[int]:
        '''Get DB chunk as Bert token ids.'''
        return cls.text_to_bert(cls.get_db_chunk_text(idx))

    @classmethod
    def get_db_chunk_text(cls, idx: int) -> str:
        '''Get DB chunk as text.'''
        return cls.gpt_to_text(cls.get_db_chunk_gpt(idx))

    @classmethod
    def get_db_chunk_and_continuation_text(cls, idx: int) -> T.List[str]:
        '''Get DB chunk along with continuation, as text.'''

        # Modulus used here to match original implementation (i.e., last
        # chunks continuation wraps around to first chunk).
        return [
            cls.get_db_chunk_text(idx),
            cls.get_db_chunk_text((idx + 1) % len(cls.get_db_dataset())),
        ]

    ##############################################
    # pretraining corpus.
    ##############################################

    @classmethod
    def get_pt_num_samples_and_chunks(cls, data_key: str) -> T.Tuple[int, int]:
        '''Number of samples & chunks (e.g., 32*n_samples) in corpus.'''
        assert hasattr(cls.pt_datasets, data_key), (
            "pretraining set '%s' not found (choices: %s)."
            % (data_key, ", ".join(vars(cls.pt_datasets).keys()))
        )
        chunk_dataset = getattr(cls.pt_datasets, data_key).chunk_dataset
        return (
            len(chunk_dataset.sample_dataset),
            len(chunk_dataset),
        )

    @classmethod
    def get_pt_num_samples(cls, data_key: str) -> int:
        '''Number of pretraining samples.'''
        return cls.get_pt_num_samples_and_chunks(data_key)[0]

    @classmethod
    def get_pt_num_chunks(cls, data_key: str) -> int:
        '''Number of pretraining chunks (e.g., 32*n_samples).'''
        return cls.get_pt_num_samples_and_chunks(data_key)[1]

    @classmethod
    def get_pt_dataset(cls, data_key: str) -> RetroDataset:
        return getattr(cls.pt_datasets, data_key)

    @classmethod
    def get_pt_sample(cls, data_key: str, idx: int) -> dict:
        return getattr(cls.pt_datasets, data_key)[idx]

    @classmethod
    def get_neighbor_tokens(cls, sample_id: int, chunk_id: int, data_key: str="train") -> T.Optional[dict]:
        try:
            sample = cls.get_pt_sample(data_key, sample_id)
            sample_token_ids = sample["text"]
            chunk_length = cls.args.retro_gpt_chunk_length
            chunk_start_idx = chunk_id * chunk_length
            chunk_end_idx = min(sample_token_ids.shape[0], chunk_start_idx + chunk_length)
            chunk_token_ids = sample_token_ids[chunk_start_idx:chunk_end_idx]
            neighbor_token_ids = sample["neighbor_tokens"][chunk_id]
            return {
                "chunk_tokens": chunk_token_ids,
                "neighbor_tokens": neighbor_token_ids,
            }
        except:
            return None

    @classmethod
    def print_neighbor_texts(cls, sample_id: int, chunk_id: int, data_key: str="train") -> None:
        tokens: dict = cls.get_neighbor_tokens(sample_id, chunk_id, data_key)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        try:
            print("PRETRAINING CHUNK:")
            print("  - %s" % shorten_str(cls.gpt_to_text(tokens["chunk_tokens"]), 150))
            print("NEIGHBOR_CHUNKS:")
            for token_ids in tokens["neighbor_tokens"]:
                print("  - %s" % shorten_str(cls.gpt_to_text(token_ids), 150))
        except:
            print("<no neighbors for sample %d>" % sample_id)

    ##############################################
    # usage.
    ##############################################

    @classmethod
    def print_usage(cls) -> None:
        '''Print usage.'''

        print()
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("examples ... [ *note*: 'db' = chunk db; 'pt' = pretraining corpus. ]")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")

        print()
        print("~~~~ indexed datasets ~~~~")
        print("retro.get_db_num_indexed_datasets() : %s" % cls.get_db_num_indexed_datasets())
        print("retro.get_db_indexed_dataset_infos() :")
        for i, (ratio, prefix) in enumerate(cls.get_db_indexed_dataset_infos()):
            print(
                "  %s(%f, %s)%s"
                % (
                    "[" if i == 0 else " ",
                    ratio,
                    prefix,
                    "]" if i == len(cls.db_indexed_dataset_infos) - 1 else ",",
                )
            )

        print()
        print("~~~~ counts ~~~~")
        print("retro.get_db_num_chunks : %d." % cls.get_db_num_chunks())

        print()
        for sq_key in ("sample", "chunk"):
            for data_key in ("train", "valid"):  # test?
                print(
                    "retro.get_pt_num_%ss('%s') : %d."
                    % (sq_key, data_key, getattr(cls, f"get_pt_num_{sq_key}s")(data_key))
                )

        print()
        print("~~~~ tokens, text ~~~~")
        print(
            "retro.get_db_chunk_gpt(chunk_id) : %s"
            % shorten_str(str(retro.get_db_chunk_gpt(0)), 50)
        )
        print(
            "retro.get_db_chunk_bert(chunk_id) : %s"
            % shorten_str(str(retro.get_db_chunk_bert(0)), 50)
        )
        print(
            "retro.get_db_chunk_text(chunk_id) : %s"
            % shorten_str(retro.get_db_chunk_text(0).strip(), 50)
        )
        print("retro.get_db_chunk_and_continuation_text(chunk_id) :")
        for i, t in enumerate(retro.get_db_chunk_and_continuation_text(0)):
            print(
                "  %s'%s'%s"
                % (
                    "[" if i == 0 else " ",
                    shorten_str(t.strip().replace("\n", " "), 50),
                    "]" if i == 1 else ",",
                )
            )

        sample = cls.get_pt_sample("train", 0)
        sample_chunk_id = sample["neighbor_tokens"].shape[0] // 2
        sample_neighbor_id = 0
        print()
        print("retro.get_pt_sample('train', sample_id) :")
        print("  {")
        for k, v in sample.items():
            print("    '%s' : %s" % (k, shorten_str(str(v), 50)))
        print("  }")

        print()
        print("(e.g., sample = retro.get_pt_sample(...))")
        print()
        print("  sample['text'].shape : %s" % str(sample["text"].shape))
        print("  sample['neighbor_tokens'].shape : %s" % str(sample["neighbor_tokens"].shape))
        print("  sample['text'] : %s" % shorten_str(str(sample["text"]), 50))
        print(
            "  sample['neighbor_tokens'][17][1] : %s"
            % shorten_str(str(sample["neighbor_tokens"][sample_chunk_id][sample_neighbor_id]), 50)
        )
        print(
            "  retro.gpt_to_text(sample['text']) : %s"
            % shorten_str(cls.gpt_to_text(sample["text"]), 50)
        )
        print(
            "  retro.gpt_to_text(sample['neighbor_tokens']) : %s"
            % shorten_str(
                cls.gpt_to_text(sample["neighbor_tokens"][sample_chunk_id][sample_neighbor_id]), 50
            )
        )

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
