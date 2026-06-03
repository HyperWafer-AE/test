from dataclasses import replace
from typing import Dict, List

from data_notation import tensor_notation

from .types import KVStateSpec


def materialize_kv_tensors(data_dict: Dict, states: List[KVStateSpec]) -> List[KVStateSpec]:
    updated = []
    for state in states:
        data_tag = (2, state.state_id)
        tensor = tensor_notation(
            data_name=state.name,
            data_tag=data_tag,
            data_shape=[state.num_blocks, state.block_elems],
            data_type=state.dtype,
            extra_info=f"agent_kv:{state.kind}",
        )
        dimension_split = (tuple([1] * state.num_blocks), (state.block_elems,))
        tensor.dummy_generated_split(dimension_split=dimension_split)
        for split in tensor.generated_splitted_tag_dict:
            tensor.generated_split_location[split] = []
        data_dict[data_tag] = tensor
        updated.append(replace(state, data_tag=data_tag))
    return updated


def state_splits(data_dict: Dict, state: KVStateSpec):
    return list(data_dict[state.data_tag].generated_splitted_tag_dict.keys())

