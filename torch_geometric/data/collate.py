from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, List, Optional, Tuple, Union

from tqdm import tqdm
import torch
from torch import Tensor
from torch_sparse import SparseTensor, cat

from torch_geometric.data.data import BaseData
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage


def collate(
    cls,
    data_list: List[BaseData],
    increment: bool = True, # TODO: changed default from True to False
    add_batch: bool = True,
    follow_batch: Optional[Union[List[str]]] = None,
    exclude_keys: Optional[Union[List[str]]] = None,
) -> Tuple[BaseData, Mapping, Mapping]:
    # Collates a list of `data` objects into a single object of type `cls`.
    # `collate` can handle both homogeneous and heterogeneous data objects by
    # individually collating all their stores.
    # In addition, `collate` can handle nested data structures such as
    # dictionaries and lists.

    if not isinstance(data_list, (list, tuple)):
        # Materialize `data_list` to keep the `_parent` weakref alive.
        data_list = list(data_list)

    if cls != data_list[0].__class__:
        out = cls(_base_cls=data_list[0].__class__)  # Dynamic inheritance.
    else:
        out = cls()

    # Create empty stores:
    max_store_lengths = max([[len(store) for store in data.stores] for data in data_list])
    for data in data_list:
        if all([max_store_lengths[k] == len(store) for k, store in enumerate(data.stores)]):
            max_data_store = data
            break
    max_data_store_keys = [store._key for store in max_data_store.stores]
    out.stores_as(max_data_store)

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    # Group all storage objects of every data object in the `data_list` by key,
    # i.e. `key_to_store_list = { key: [store_1, store_2, ...], ... }`:
    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)
    missing_data_stores_dict = compute_missing_data_stores(data_list, max_data_store, max_data_store_keys)

    # if isinstance(data.stores[0], BaseStorage):
        #     key_to_stores[None].append(data.stores[0])
        # else:
        #     raise Exception("data.stores[0] is not BaseStorage")
        # for i, node_type in enumerate(max_data_store.node_types):
        #     key_to_stores[node_type].append(data.node_stores[i]) if (i < len(data.node_stores) and data.node_types[i]==max_data_store.node_types[i]) else key_to_stores[node_type].append(None)
        # for i, edge_type in enumerate(max_data_store.edge_types):
        #             key_to_stores[edge_type].append(data.edge_stores[i]) if (i < len(data.edge_stores) and data.edge_types[i]==max_data_store.edge_types[i]) else key_to_stores[edge_type].append(None)
        # print(key_to_stores)

    # With this, we iterate over each list of storage objects and recursively
    # collate all its attributes into a unified representation:

    # We maintain two additional dictionaries:
    # * `slice_dict` stores a compressed index representation of each attribute
    #    and is needed to re-construct individual elements from mini-batches.
    # * `inc_dict` stores how individual elements need to be incremented, e.g.,
    #   `edge_index` is incremented by the cumulated sum of previous elements.
    #   We also need to make use of `inc_dict` when re-constructuing individual
    #   elements as attributes that got incremented need to be decremented
    #   while separating to obtain original values.
    device = None
    slice_dict, inc_dict = defaultdict(dict), defaultdict(dict)


    # edge_index_batch = []
    # edge_type_batch = [] # TODO: not functioning yet!
    # for i in range(len(data.node_types)):
    #     edge_index_batch = edge_index_batch + [data_list[j].edge_stores[i].edge_index for j in range(len(data_list))]
    #     edge_type_batch = edge_type_batch+repeat_interleave([data_list[j].edge_stores[i].num_nodes for j in range(len(data_list))])
    # collected_node_stores = []

    # print("Going through stores in collate.")
    for i, out_store in tqdm(enumerate(out.stores)):
        key = out_store._key
        stores = key_to_stores[key]

        # here we need to distinguish between dict and list -> I think we should completely replace this with a recursive function..
        # if isinstance(stores, dict):
        #     for key in stores.keys():



        missing_store_indices = missing_data_stores_dict[key]
        stores_with_attributes = [store for store in stores if store is not None and len(store)>0]

        # if isinstance(missing_store_indices, list):
        #     # this is the wrong thing to assert -> should be recursive do to the nested nature of the problem
        #     assert(len([store for store in stores_with_attributes])+len(missing_store_indices)==len(data_list))

        if len(stores_with_attributes)>0: # -> again, we should go through the attributes recursively (we do not reach all of them right now)
            for attr in stores_with_attributes[0].keys():

                if attr in exclude_keys:  # Do not include top-level attribute.
                    continue
                values = [getattr(store, attr) if store is not None and hasattr(store, attr) else None for store in stores]
                values = [getattr(store, attr) if store is not None and hasattr(store, attr) else None for store in stores]

                # The `num_nodes` attribute needs special treatment, as we need to
                # sum their values up instead of merging them to a list:
                if attr == 'num_nodes':
                    out_store._num_nodes = values
                    out_store.num_nodes = sum([value for value in values if value is not None])
                    continue

                # Skip batching of `ptr` vectors for now:
                if attr == 'ptr':
                    continue

                # Collate attributes into a unified representation:
                value, slices, incs = _collate(attr, values, data_list, stores,
                                               increment)
                if attr in missing_store_indices: # not 100% sure if this is sufficient
                    slices = slice_correction(slices, missing_store_indices[attr], data_list)
                else:
                    slices = slice_correction(slices, missing_store_indices, data_list)

                if isinstance(value, Tensor) and value.is_cuda:
                    device = value.device

                out_store[attr] = value
                if key is not None:
                    slice_dict[key][attr] = slices
                    inc_dict[key][attr] = incs
                else:
                    slice_dict[attr] = slices
                    inc_dict[attr] = incs

                # Add an additional batch vector for the given attribute:
                if attr in follow_batch:
                    batch, ptr = _batch_and_ptr(slices, device)
                    out_store[f'{attr}_batch'] = batch
                    out_store[f'{attr}_ptr'] = ptr

            # In case the storage holds node, we add a top-level batch vector it:
            if (add_batch and isinstance(stores[0], NodeStorage)
                    and stores[0].can_infer_num_nodes):
                repeats = [store.num_nodes for store in stores]
                out_store.batch = repeat_interleave([repeat for repeat in repeats if repeat is not None], device=device)
                out_store.ptr = cumsum(torch.tensor([repeat for repeat in repeats if repeat is not None], device=device))

            # Add a batch vector for heterogeneous data which does satisfy the boolean statement above.
            if add_batch and not (isinstance(stores[0], NodeStorage) or isinstance(stores[0], EdgeStorage)) and isinstance(data_list[0], HeteroData):
                # try:
                #     repeats = [data_list_element.num_nodes for data_list_element in data_list]
                #     out_store.batch = repeat_interleave(repeats, device=device)
                #     out_store.ptr = cumsum(torch.tensor(repeats, device=device))
                try:
                    out_store.batch_dict = {}
                    for i in range(len(max_data_store.node_types)):
                        out_store.batch_dict[max_data_store.node_types[i]] = repeat_interleave(
                            [data_list[j].node_stores[i]._mapping['_Cochain__x'].size()[0] if len(data_list[j].node_stores)>i and len(data_list[j].node_stores[i])>0 else 0 for j in range(len(data_list))])
                        #out_store.batch_dict[max_data_store.node_types[i]] = repeat_interleave([data_list[j].node_stores[i]._mapping['_Cochain__x'].size()[0] for j in range(len(data_list)) if len(data_list[j].node_stores)>i and len(data_list[j].node_stores[i])>0])
                except:
                    raise Exception("Batching error for heterogeneous graph. Please check the data_list.")

            # out.stores[i] = out_store
    return out, slice_dict, inc_dict


def _collate(
    key: str,
    values: List[Any],
    data_list: List[BaseData],
    stores: List[BaseStorage],
    increment: bool,
) -> Tuple[Any, Any, Any]:

    # elem = values[0] # TODO: this is not good, because the first graph may have different node or edge types than others
    non_none_values = [(i,value) for i,value in enumerate(values) if value is not None]
    if any(isinstance(el, dict) for el in values):
        maximal_value_length = max([len(value[1]) for value in non_none_values])
        maximal_values = [(i,value) for (i,value) in non_none_values if len(value) == maximal_value_length]
        elem_idx, elem = maximal_values[0]
    else:
        elem_idx, elem = non_none_values[0] if len(non_none_values)>0 else (0, None)

    if isinstance(elem, Tensor):
        # Concatenate a list of `torch.Tensor` along the `cat_dim`.
        # NOTE: We need to take care of incrementing elements appropriately.
        key = str(key)
        cat_dim = data_list[elem_idx].__cat_dim__(key, elem, stores[elem_idx]) # changed index
        if cat_dim is None or elem.dim() == 0:
            values = [value.unsqueeze(0) for value in values]
        # slices = cumsum([value.size(cat_dim or 0) if value is not None else 0 for value in values])
        slices = cumsum([value.size(cat_dim or 0) for value in values if value is not None])
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if incs.dim() > 1 or int(incs[-1]) != 0:
                values = [
                    value + inc.to(value.device)
                    for value, inc in zip([value for value in values if value is not None], incs)
                ] # TODO: added if value is not None
        else:
            incs = None

        if torch.utils.data.get_worker_info() is not None:
            # Write directly into shared memory to avoid an extra copy:
            numel = sum(value.numel() for value in values)
            storage = elem.storage()._new_shared(numel)
            shape = list(elem.size())
            if cat_dim is None or elem.dim() == 0:
                shape = [len(values)] + shape
            else:
                shape[cat_dim] = int(slices[-1])
            out = elem.new(storage).resize_(*shape)
        else:
            out = None

        # value = torch.cat(values, dim=cat_dim or 0, out=out)
        # todo: This is just a workaround to an error.
        # min_size_initial = min([value.size()[-1] for value in values if value is not None])

        if not key == 'edge_index':
            try:
                value = torch.cat([value for value in values if value is not None], dim=0, out=out)
            except:
                value = torch.cat([value for value in values if value is not None], dim=1, out=out)
        else:
            value = torch.cat([value for value in values if value is not None], dim=1, out=out)

        return value, slices, incs

    elif isinstance(elem, SparseTensor) and increment:
        # Concatenate a list of `SparseTensor` along the `cat_dim`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        key = str(key)
        cat_dim = data_list[elem_idx].__cat_dim__(key, elem, stores[elem_idx])
        cat_dims = (cat_dim, ) if isinstance(cat_dim, int) else cat_dim
        repeats = [[value.size(dim) for dim in cat_dims] for value in values]
        slices = cumsum(repeats)
        value = cat(values, dim=cat_dim)
        return value, slices, None

    elif isinstance(elem, (int, float)):
        # Convert a list of numerical values to a `torch.Tensor`.
        value = torch.tensor(values)
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if int(incs[-1]) != 0:
                value.add_(incs)
        else:
            incs = None
        slices = torch.arange(len(values) + 1)
        return value, slices, incs

    elif isinstance(elem, Mapping):
        # Set missing key attributes to `None`:
        key_list = [set(v.keys()) for v in values]
        all_keys = set()
        for i in key_list:
            if i not in all_keys:
                all_keys = all_keys.union(i)
        all_keys = sorted(all_keys)
        # some graphs do not have all attributes
        for key in all_keys:
            for v in values:
                if not key in v:
                    v[key] = None
                    #v[key] = torch.tensor([torch.nan for i in range(attr_dim)]).resize(1, attr_dim)

        # Recursively collate elements of dictionaries.
        value_dict, slice_dict, inc_dict = {}, {}, {}
        for key in all_keys:
            value_dict[key], slice_dict[key], inc_dict[key] = _collate(key, [v[key] for v in values], data_list, stores,
                                                                       increment)
        return value_dict, slice_dict, inc_dict

    elif (isinstance(elem, Sequence) and not isinstance(elem, str)
          and len(elem) > 0 and isinstance(elem[0], (Tensor, SparseTensor))):
        # Recursively collate elements of lists.
        value_list, slice_list, inc_list = [], [], []
        for i in range(len(elem)):
            value, slices, incs = _collate(key, [v[i] for v in values],
                                           data_list, stores, increment)
            value_list.append(value)
            slice_list.append(slices)
            inc_list.append(incs)
        return value_list, slice_list, inc_list

    else:
        # Other-wise, just return the list of values as it is.
        slices = torch.arange(len(values) + 1)
        return values, slices, None


def _batch_and_ptr(
    slices: Any,
    device: Optional[torch.device] = None,
) -> Tuple[Any, Any]:
    if (isinstance(slices, Tensor) and slices.dim() == 1):
        # Default case, turn slices tensor into batch.
        repeats = slices[1:] - slices[:-1]
        batch = repeat_interleave(repeats.tolist(), device=device)
        ptr = cumsum(repeats.to(device))
        return batch, ptr

    elif isinstance(slices, Mapping):
        # Recursively batch elements of dictionaries.
        batch, ptr = {}, {}
        for k, v in slices.items():
            batch[k], ptr[k] = _batch_and_ptr(v, device)
        return batch, ptr

    elif (isinstance(slices, Sequence) and not isinstance(slices, str)
          and isinstance(slices[0], Tensor)):
        # Recursively batch elements of lists.
        batch, ptr = [], []
        for s in slices:
            sub_batch, sub_ptr = _batch_and_ptr(s, device)
            batch.append(sub_batch)
            ptr.append(sub_ptr)
        return batch, ptr

    else:
        # Failure of batching, usually due to slices.dim() != 1
        return None, None


###############################################################################


def repeat_interleave(
    repeats: List[int],
    device: Optional[torch.device] = None,
) -> Tensor:
    outs = [torch.full((n, ), i, device=device) for i, n in enumerate(repeats)]
    return torch.cat(outs, dim=0)


def cumsum(value: Union[Tensor, List[int]]) -> Tensor:
    if not isinstance(value, Tensor):
        value = torch.tensor(value)
    out = value.new_empty((value.size(0) + 1, ) + value.size()[1:])
    out[0] = 0
    torch.cumsum(value, 0, out=out[1:])
    return out


def get_incs(key, values: List[Any], data_list: List[BaseData],
             stores: List[BaseStorage]) -> Tensor:
    repeats = [
        data.__inc__(key, value, store)
        for value, data, store in zip(values, data_list, stores) if value is not None
    ] # TODO: added if value is not None and data is not None
    if isinstance(repeats[0], Tensor):
        repeats = torch.stack(repeats, dim=0)
    else:
        repeats = torch.tensor(repeats)
    return cumsum(repeats[:-1])

def slice_correction(slices, missing_store_indices, data_list):

    if isinstance(slices, dict):
        for key, value in slices.items():
            slices[key] = slice_correction(value, missing_store_indices[key], data_list)
        return slices

    elif isinstance(slices, Tensor):
        if slices.size()[0]+len(missing_store_indices)!=len(data_list)+1:
            UserWarning("Slices were computed incorrectly due to missing values. Setting slices to None.")
            new_slices = [torch.nan for i in range(len(data_list)+1)]
        else:
            new_slices = []
            it = 0
            for j in range(len(data_list)+1):
                new_slices.append(slices[it]) if j not in missing_store_indices else new_slices.append(torch.nan)
                it += 1 if j not in missing_store_indices else 0
            new_slices = torch.tensor(new_slices)
        return new_slices

def compute_missing_data_stores(data_list, max_data_store, max_data_store_keys):
    missing_data_stores_dict = dict()
    for j, data in enumerate(data_list):
        for store in max_data_store.stores:
            if store._key not in [data_store._key for data_store in data.stores]:
                missing_data_stores_dict[store._key] = compute_single_missing_data_store(None, j, store, missing_data_stores_dict, in_key=store._key)
            else:
                indx = [data_store._key for data_store in data.stores].index(store._key)
                missing_data_stores_dict[store._key] = compute_single_missing_data_store(data.stores[indx], j, store, missing_data_stores_dict, in_key=store._key)
    return missing_data_stores_dict

def compute_single_missing_data_store(data_store, j, max_data_store_value, missing_data_stores_dict, in_key=None):
    if isinstance(max_data_store_value, dict) or isinstance(max_data_store_value, BaseStorage):
        if not in_key in missing_data_stores_dict:
            missing_data_stores_dict[in_key] = dict()
        for key, value in max_data_store_value.items():
            if data_store is not None and key in data_store:
                missing_data_stores_dict[in_key][key] = compute_single_missing_data_store(data_store[key], j, value, missing_data_stores_dict[in_key], key)
            else:
                missing_data_stores_dict[in_key][key] = compute_single_missing_data_store(None, j, value, missing_data_stores_dict[in_key], key)
        return missing_data_stores_dict[in_key]
    else:
        if in_key not in missing_data_stores_dict:
            missing_data_stores_dict[in_key] = []
        if data_store is None:
            missing_data_stores_dict[in_key].append(j)
        return missing_data_stores_dict[in_key]