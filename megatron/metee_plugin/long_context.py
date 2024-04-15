import gc

import faiss
import numpy as np
import torch

from megatron.core import mpu
from megatron.training import get_args

class retrieve_database(object):
    r""" database to store key, value pairs 
    """

    
    def __init__(self) -> None:
        self.faiss_database = dict()
        self.element = 0
        self.is_train = True
        self.key = list()
        self.value = list()

    def add(self, key, value) -> None:
        r"""
        """
        key = key.detach().cpu()
        value = value.detach().cpu()
        _, batch_size_k, attn_heads_k, hidden_state_projection = key.shape
        _, batch_size_v, attn_heads_v, _ = value.shape
        assert batch_size_k == batch_size_v, "the number of batch sizes should be equal in key and value"
        assert attn_heads_k == attn_heads_v, "the number of attention heads should be equal in key and value"
        self.element = self.element+batch_size_k*attn_heads_k
        for batch_index in range(0, batch_size_k):
            self.faiss_database[batch_index] = dict()
            self.key.append(list())
            self.value.append(list())
            for attn_head_index in range(0, attn_heads_k):
                index = faiss.IndexFlatL2(hidden_state_projection)
                index.add(key[:, batch_index, attn_head_index, :])
                self.faiss_database[batch_index][attn_head_index] = index
                self.key[batch_index].append(key[:, batch_index, attn_head_index, :])
                self.value[batch_index].append(value[:, batch_index, attn_head_index, :])
                
    
    def clear(self) -> None:
        r"""
        """
        self.faiss_database = dict()
        self.key = list()
        self.value = list()
        self.element = 0
        gc.collect()

    def get(self, query, top_k: int):
        r""" 
        """
        query = query.detach().cpu()
        _, batch_size, attn_heads, _ = query.shape
        ret_key = list()
        ret_value = list()
        for batch_index in range(0, batch_size):
            ret_key.append(list())
            ret_value.append(list())
            for attn_head_index in range(0, attn_heads):
                _, ret_index = self.faiss_database[batch_index][attn_head_index].search(query[:, batch_index, attn_head_index, :], top_k)
                ret_key[batch_index].append(self.key[batch_index][attn_head_index][ret_index.flatten()])
                ret_value[batch_index].append(self.value[batch_index][attn_head_index][ret_index.flatten()])

        return np.transpose(np.array(ret_key), (2, 0, 1, 3)), np.transpose(np.array(ret_value), (2, 0, 1, 3))
    
    def set_train_mode(self):
        if self.is_train == False:
           self.is_train = True
           self.clear()

    def set_eval_mode(self):
        if self.is_train == True:
           self.is_train = False
           self.clear()

    def __len__(self) -> int:
    	return self.element

class context_data(object):

    def __init__(self, context_length) -> None:
        self.context_length = context_length
    
    def get_data(self, data):
        self.tokens = data["tokens"].cuda(non_blocking = True)
        self.labels = data["labels"].cuda(non_blocking = True)
        self.loss_mask = data["loss_mask"][:, :self.context_length].cuda(non_blocking = True)
        self.attention_mask =  None if "attention_mask" not in data else data["attention_mask"][:, :, :self.context_length, :self.context_length].cuda(non_blocking = True)
        self.position_ids = data["position_ids"].cuda(non_blocking = True)
    
    def __getitem__(self, index):
        batch = {
            'tokens': self.tokens[:, index*self.context_length:(index+1)*self.context_length].contiguous(),
            'labels': self.labels[:, index*self.context_length:(index+1)*self.context_length].contiguous(),
            'loss_mask': self.loss_mask.contiguous(),
            'attention_mask': self.attention_mask.contiguous(),
            #'position_ids': self.position_ids[:, index*self.context_length:(index+1)*self.context_length].contiguous()
            'position_ids': self.position_ids[:, :self.context_length].contiguous()
        }
        return batch


class context_dataloader(object):
    __instance = None

    def __new__(cls, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
            cls.data_iterator = kwargs['data_iterator']
            cls.args = get_args()
            cls.seq_length = cls.args.seq_length
            cls.context_length = cls.args.context_length
            assert cls.data_iterator is not None or mpu.get_tensor_model_parallel_rank(), "data iterator is None"
            assert cls.seq_length%cls.context_length == 0, "sequence length should be a multiple of context length"
            cls.counter = cls.seq_length/cls.context_length
            cls.context_data = context_data(cls.context_length)
        return cls.__instance

    def __init__(self, data_iterator) -> None:
        pass
        #self.data_iterator = data_iterator
        #self.args = get_args()
        #self.seq_length = self.args.seq_length
        #self.context_length = self.args.context_length
        #assert self.data_iterator is not None or mpu.get_tensor_model_parallel_rank(), "data iterator is None"
        #assert self.seq_length%self.context_length == 0, "sequence length should be a multiple of context length"
        #self.counter = self.seq_length/self.context_length
        #self.context_data = context_data(self.context_length)
     
    def get_batch_on_this_tp_rank(self):

        def _broadcast(item):
            if item is not None:
                torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

        if mpu.get_tensor_model_parallel_rank() == 0:
            if self.counter == self.seq_length/self.context_length:
                self.counter = 0
                data = next(self.data_iterator)
                self.context_data.get_data(data)
            batch = self.context_data[self.counter]
            if self.args.pipeline_model_parallel_size == 1:
                _broadcast(batch['tokens'])
                _broadcast(batch['labels'])
                _broadcast(batch['loss_mask'])
                _broadcast(batch['attention_mask'])
                _broadcast(batch['position_ids'])

            elif mpu.is_pipeline_first_stage():
                _broadcast(batch['tokens'])
                _broadcast(batch['attention_mask'])
                _broadcast(batch['position_ids'])

            elif mpu.is_pipeline_last_stage():
                _broadcast(batch['labels'])
                _broadcast(batch['loss_mask'])
                _broadcast(batch['attention_mask'])

        else:
            tokens=torch.empty((self.args.micro_batch_size,self.context_length), dtype = torch.int64 , device = torch.cuda.current_device())
            labels=torch.empty((self.args.micro_batch_size,self.context_length), dtype = torch.int64 , device = torch.cuda.current_device())
            loss_mask=torch.empty((self.args.micro_batch_size,self.context_length), dtype = torch.float32 , device = torch.cuda.current_device())
            if self.args.create_attention_mask_in_dataloader:
                attention_mask=torch.empty(
                        (self.args.micro_batch_size,1,self.context_length,self.context_length), dtype = torch.bool , device = torch.cuda.current_device()
                    )
            else:
                attention_mask=None
            position_ids=torch.empty((self.args.micro_batch_size,self.context_length), dtype = torch.int64 , device = torch.cuda.current_device())

            if self.args.pipeline_model_parallel_size == 1:
                _broadcast(tokens)
                _broadcast(labels)
                _broadcast(loss_mask)
                _broadcast(attention_mask)
                _broadcast(position_ids)
        
            elif mpu.is_pipeline_first_stage():
                labels=None
                loss_mask=None
        
                _broadcast(tokens)
                _broadcast(attention_mask)
                _broadcast(position_ids)

            elif mpu.is_pipeline_last_stage():
                tokens=None
                position_ids=None
            
                _broadcast(labels)
                _broadcast(loss_mask)
                _broadcast(attention_mask)
        
            batch = {
                'tokens': tokens,
                'labels': labels,
                'loss_mask': loss_mask,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }
        self.counter = self.counter+1
        return batch.values(), self.counter == self.seq_length/self.context_length
