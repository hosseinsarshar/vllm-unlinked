from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch_xla.experimental.custom_kernel  # Required to register custom ops.

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.distributed.utils import get_shard_spec, get_partition_spec, get_mesh, get_device_ids, is_spmd, enable_man_sharding
import torch_xla
import torch_xla.distributed.spmd as xs
import os

class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "PALLAS"

    @staticmethod
    def get_impl_cls() -> Type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["PallasMetadata"]:
        return PallasMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_kv_heads, num_blocks, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")

    # hosseins: removed torch.compile - DONE
    # @torch.compile(backend="openxla")
    @staticmethod
    def copy_blocks(
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        src_to_dists: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        src_indices, dst_indices = src_to_dists
        for k_cache, v_cache in kv_caches:
            torch.ops.xla.dynamo_set_buffer_donor_(k_cache, True)
            k_cache[:, dst_indices] = k_cache[:, src_indices]
            torch.ops.xla.dynamo_set_buffer_donor_(v_cache, True)
            v_cache[:, dst_indices] = v_cache[:, src_indices]


@dataclass
class PallasMetadata(AttentionMetadata):

    # Currently, input sequences can only contain all prefills
    # or all decoding.
    block_tables: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    effective_query_lens: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self) -> Optional["PallasMetadata"]:
        if self.num_prefills == 0:
            return None

        assert self.num_decode_tokens == 0
        return self

    @property
    def decode_metadata(self) -> Optional["PallasMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.block_tables is not None
        assert self.context_lens is not None
        return self


class PallasAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if head_size % 128 != 0:
            raise NotImplementedError("Head size must be a multiple of 128.")
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")
        if sliding_window is not None:
            raise NotImplementedError("Sliding window is not supported.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError("FP8 KV cache dtype is not supported.")
        if blocksparse_params is not None:
            raise NotImplementedError("Blocksparse is not supported.")
        if logits_soft_cap is not None:
            raise NotImplementedError(
                "Attention logits soft-capping is not supported.")

        if torch_xla.tpu.version() < 4:
            raise NotImplementedError("TPU version must be 4 or higher.")

        self.megacore_mode = None
        tpu_env = torch_xla.tpu.get_tpu_env()
        tpu_type = (tpu_env.get("ACCELERATOR_TYPE", None)
                    or tpu_env.get("TYPE", None)
                    or tpu_env.get("TPU_ACCELERATOR_TYPE", None))
        assert tpu_type is not None
        tpu_type = tpu_type.lower()

        if (("lite" not in tpu_type) and ("v6" not in tpu_type)):
            if self.num_kv_heads % 2 == 0:
                self.megacore_mode = "kv_head"
            else:
                # NOTE(woosuk): If the batch size is not a multiple of 2, the
                # megacore mode will be None.
                self.megacore_mode = "batch"

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: PallasMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Pallas attention.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache[0] = [num_kv_heads, num_blocks, block_size, head_size]
            kv_cache[1] = [num_kv_heads, num_blocks, block_size, head_size]
                NOTE: kv_cache[0] and kv_cache[1] will be an empty tensor 
                with shape [0] for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        key_cache, value_cache = kv_cache
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{query.shape=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{get_shard_spec(query)=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{key.shape=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{get_shard_spec(key)=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{value.shape=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{get_shard_spec(value)=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{key_cache.shape=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{get_shard_spec(key_cache)=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{value_cache.shape=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{get_shard_spec(value_cache)=}]")

        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads

        assert k_scale == 1.0 and v_scale == 1.0
        batch_size, seq_len, hidden_size = query.shape
        query = query.view(batch_size, seq_len, num_heads, self.head_size)
        key = key.view(batch_size, seq_len, num_kv_heads, self.head_size)
        value = value.view(batch_size, seq_len, num_kv_heads, self.head_size)

        print(f"hosseins: PallasAttentionBackendImpl -> forward() 2 [{query.shape=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 2 [{key.shape=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 2 [{value.shape=}]")
        
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 2 [{get_shard_spec(query)=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 2 [{get_shard_spec(key)=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 2 [{get_shard_spec(value)=}]")

        print(f"hosseins: PallasAttentionBackendImpl -> forward() 3 [{query.shape=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 3 [{key.shape=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 3 [{value.shape=}]")

        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{is_spmd()=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{get_device_ids()=}]")
        
        # if is_spmd():
        #     num_heads = self.num_heads // len(get_device_ids())
        #     num_kv_heads = self.num_kv_heads // len(get_device_ids())
        # else:
        #     num_heads = self.num_heads
        #     num_kv_heads = self.num_kv_heads
        
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{num_heads=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 1 [{num_kv_heads=}]")

        if kv_cache[0].numel() > 0:
            print(f"hosseins: PallasAttentionBackendImpl -> forward() 4 [{attn_metadata.slot_mapping.shape=}]")
            print(f"hosseins: PallasAttentionBackendImpl -> forward() 4 [{attn_metadata.slot_mapping.device=}]")
            # hosseins: todo: this is the culprit!!!
            # if is_spmd(): slot_mapping = attn_metadata.slot_mapping[:attn_metadata.slot_mapping.shape[0] // len(get_device_ids())]
            slot_mapping = attn_metadata.slot_mapping

            write_to_kv_cache(key, value, key_cache, value_cache, slot_mapping)

        query = query * self.scale
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 4 [{attn_metadata.num_prefills > 0=}]")
        
        if attn_metadata.num_prefills > 0:
            print(f"hosseins: PallasAttentionBackendImpl -> forward() 4 [attn_metadata.num_prefills > 0]")
            print(f"hosseins: PallasAttentionBackendImpl -> forward() 4 [{attn_metadata.block_tables is None=}]")

            if attn_metadata.block_tables is None:
                {}
                # Prefill without paged KV cache.
                assert seq_len % 16 == 0, (
                    "Pallas FlashAttention kernel requires seq_len to be a "
                    f"multiple of 16 but got {seq_len}")

                # Handle GQA/MQA.
                if num_kv_heads != num_heads:
                    key = key.repeat_interleave(self.num_queries_per_kv,
                                                dim=-2)
                    key = key.view(batch_size, seq_len, num_heads,
                                   self.head_size)
                    value = value.repeat_interleave(self.num_queries_per_kv,
                                                    dim=-2)
                    value = value.view(batch_size, seq_len, num_heads,
                                       self.head_size)
                # FlashAttention kernel requires the input shape to be
                # [batch_size, num_heads, seq_len, d_model]
                # while the input is [batch_size, seq_len, num_heads, d_model].
                # Permute the input to match the required format.
                output = torch.ops.xla.flash_attention(
                    query.permute(0, 2, 1, 3),
                    key.permute(0, 2, 1, 3),
                    value.permute(0, 2, 1, 3),
                    True,
                )
                output = output.permute(0, 2, 1, 3)
            else:
                # Prefill with paged KV cache.
                # TODO(woosuk): Tune the below knobs.
                num_kv_pages_per_compute_block = 16
                num_queries_per_compute_block = 16
                assert seq_len % num_queries_per_compute_block == 0
                output = torch.ops.xla.multi_queries_paged_attention(
                    query,
                    key_cache,
                    value_cache,
                    attn_metadata.context_lens,
                    attn_metadata.block_tables,
                    attn_metadata.effective_query_lens,
                    num_kv_pages_per_compute_block,
                    num_queries_per_compute_block,
                    use_kernel=True,
                )
        else:
            # Decoding run.
            assert kv_cache[0].numel() > 0
            query = query.squeeze(dim=1)
            pages_per_compute_block = 16  # TODO(woosuk): Tune this value.

            assert attn_metadata.block_tables is not None
            assert attn_metadata.context_lens is not None
            # NOTE(woosuk): The PagedAttention Pallas kernel stores the entire
            # block table in SMEM. Therefore, if the block table is too large,
            # the kernel compilation will fail. To avoid this, we split the
            # batch dimension into smaller chunks and run the kernel multiple
            # times.
            MAX_SMEM_USAGE = 512 * 1024
            size_per_seq = 4 * attn_metadata.block_tables.shape[1]
            max_num_seq = MAX_SMEM_USAGE // size_per_seq

            if batch_size <= max_num_seq:
                output = paged_attention(
                    query,
                    key_cache,
                    value_cache,
                    attn_metadata.context_lens,
                    attn_metadata.block_tables,
                    pages_per_compute_block,
                    self.megacore_mode,
                )
            else:
                chunk_size = max_num_seq
                # Make sure the chunk size is a multiple of 2.
                chunk_size = chunk_size // 2 * 2
                num_chunks = (batch_size + chunk_size - 1) // chunk_size

                output = torch.empty_like(query)
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = chunk_start + chunk_size
                    # NOTE(woosuk): We skip this line because it causes Dynamo
                    # compilation error. Instead, we rely on the slice operation
                    # to handle the out-of-bound case.
                    # chunk_end = min(chunk_end, batch_size)
                    chunk_output = paged_attention(
                        query[chunk_start:chunk_end],
                        key_cache,
                        value_cache,
                        attn_metadata.context_lens[chunk_start:chunk_end],
                        attn_metadata.block_tables[chunk_start:chunk_end],
                        pages_per_compute_block,
                        self.megacore_mode,
                    )
                    output[chunk_start:chunk_end] = chunk_output

        # Reshape the output tensor.
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 5 [{output.shape=}]")

        ret_o = output.reshape(batch_size, seq_len, hidden_size)

        print(f"hosseins: PallasAttentionBackendImpl -> forward() 6 [{key.shape=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 6 [{get_shard_spec(key)=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 6 [{value.shape=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 6 [{get_shard_spec(value)=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 6 [{ret_o.shape=}]")
        print(f"hosseins: PallasAttentionBackendImpl -> forward() 6 [{get_shard_spec(ret_o)=}]")

        return ret_o


def write_to_kv_cache(
    key_org: torch.Tensor,
    value_org: torch.Tensor,
    key_cache_org: torch.Tensor,
    value_cache_org: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    # todo: bring this block to the previous function call - 
    if is_spmd():
        # key_part_spec = get_partition_spec(key)
        # value_part_spec = get_partition_spec(value)
        # key_cache_spec = get_partition_spec(key_cache)
        # slot_mapping_spec = get_partition_spec(slot_mapping)
        # print("hosseins: -1")
        # value_cache_spec = get_partition_spec(value_cache)
        # print("hosseins: 0")
        # k_full_shape = key.shape
        # print("hosseins: 1")
        # v_full_shape = value.shape
        # print("hosseins: 2")
        # key_cache_full_shape = key_cache.shape
        # print("hosseins: 3")
        # value_cache_full_shape = value_cache.shape
        # slot_mapping_full_shape = slot_mapping.shape
        # print("hosseins: 4")
        # print("hosseins: 5")

        print(f"hosseins: write_to_kv_cache() 3 [{get_shard_spec(key_org)=}]")
        print(f"hosseins: write_to_kv_cache() 3 [{get_partition_spec(key_org)=}]")
        print(f"hosseins: write_to_kv_cache() 3 [{get_shard_spec(value_org)=}]")
        print(f"hosseins: write_to_kv_cache() 3 [{get_partition_spec(value_org)=}]")
        print(f"hosseins: write_to_kv_cache() 3 [{get_shard_spec(key_cache_org)=}]")
        print(f"hosseins: write_to_kv_cache() 3 [{get_partition_spec(key_cache_org)=}]")
        print(f"hosseins: write_to_kv_cache() 3 [{get_shard_spec(value_cache_org)=}]")
        print(f"hosseins: write_to_kv_cache() 3 [{get_partition_spec(value_cache_org)=}]")
        print(f"hosseins: write_to_kv_cache() 3 [{get_shard_spec(slot_mapping)=}]")
        print(f"hosseins: write_to_kv_cache() 3 [{get_partition_spec(slot_mapping)=}]")
        # print(f"hosseins: write_to_kv_cache() 3 [{key_part_spec=}]")
        # print(f"hosseins: write_to_kv_cache() 3 [{value_part_spec=}]")
        # print(f"hosseins: write_to_kv_cache() 3 [{slot_mapping_spec=}]")
        # print(f"hosseins: write_to_kv_cache() 3 [{key_cache_spec=}]")
        # print(f"hosseins: write_to_kv_cache() 3 [{value_cache_spec=}]")

        print("hosseins: write_to_kv_cache() 3 - calling xs.enable_manual_sharding")
        # key = xs.enable_manual_sharding(key_org, get_partition_spec(key_org), mesh=get_mesh()).global_tensor
        # value = xs.enable_manual_sharding(value_org, get_partition_spec(value_org), mesh=get_mesh()).global_tensor
        # key_cache = xs.enable_manual_sharding(key_cache_org, get_partition_spec(key_cache_org), mesh=get_mesh()).global_tensor
        # value_cache = xs.enable_manual_sharding(value_cache_org, get_partition_spec(value_cache_org), mesh=get_mesh()).global_tensor

        key = enable_man_sharding(key_org).global_tensor
        value = enable_man_sharding(value_org).global_tensor
        key_cache = enable_man_sharding(key_cache_org).global_tensor
        value_cache = enable_man_sharding(value_cache_org).global_tensor
        
    else:
        key = key_org
        value = value_org
        key_cache = key_cache_org
        value_cache = value_cache_org


    torch.ops.xla.dynamo_set_buffer_donor_(key_cache, True)
    torch.ops.xla.dynamo_set_buffer_donor_(value_cache, True)
    
    # print out the sharding the key, value, key_cache, value_cache in eager mode
    print(f"hosseins: write_to_kv_cache() 1 [{get_shard_spec(key)=}]")
    print(f"hosseins: write_to_kv_cache() 1 [{key.shape=}]")
    key = key.flatten(0, 2) # why we need flatten in first place - whether the flattening axis is sharded
    print(f"hosseins: write_to_kv_cache() 2 [{get_shard_spec(key)=}]")
    print(f"hosseins: write_to_kv_cache() 2 [{key.shape=}]")
    print(f"hosseins: write_to_kv_cache() 1 [{get_shard_spec(value)=}]")
    print(f"hosseins: write_to_kv_cache() 1 [{value.shape=}]")
    value = value.flatten(0, 2)
    print(f"hosseins: write_to_kv_cache() 2 [{get_shard_spec(value)=}]")
    print(f"hosseins: write_to_kv_cache() 2 [{value.shape=}]")
    print(f"hosseins: write_to_kv_cache() 1 [{get_shard_spec(key_cache)=}]")
    print(f"hosseins: write_to_kv_cache() 1 [{key_cache.shape=}]")
    key_cache = key_cache.flatten(0, 2) # hosseins: sharding should align with key_cache
    print(f"hosseins: write_to_kv_cache() 2 [{get_shard_spec(key_cache)=}]")
    print(f"hosseins: write_to_kv_cache() 2 [{key_cache.shape=}]")
    print(f"hosseins: write_to_kv_cache() 1 [{get_shard_spec(value_cache)=}]")
    print(f"hosseins: write_to_kv_cache() 1 [{value_cache.shape=}]")
    value_cache = value_cache.flatten(0, 2)
    print(f"hosseins: write_to_kv_cache() 2 [{get_shard_spec(value_cache)=}]")
    print(f"hosseins: write_to_kv_cache() 2 [{value_cache.shape=}]")
    print(f"hosseins: write_to_kv_cache() [{slot_mapping.shape=}]")
    key_cache.index_copy_(0, slot_mapping, key)
    value_cache.index_copy_(0, slot_mapping, value)

    print(f"hosseins: write_to_kv_cache() 3 [{key.device=}]")
    print(f"hosseins: write_to_kv_cache() 3 [{value.device=}]")
    print(f"hosseins: write_to_kv_cache() 3 [{key_cache.device=}]")
    print(f"hosseins: write_to_kv_cache() 3 [{value_cache.device=}]")
    print(f"hosseins: write_to_kv_cache() 3 [{slot_mapping.device=}]")

    # read this: https://github.com/pytorch/xla/issues/8742#issuecomment-2691473071
    # it means that this function failed: Check failed: IsNonDeviceDataIR(input) - this means that you have data and I can't .. - the moment you enable or disable manual sharidng,
    # we first do marksharding.

    # if is_spmd():
    #     print("hosseins: PallasAttentionBackendImpl -> forward() 4 - calling xs.disable_manual_sharding")
    #     key = xs.disable_manual_sharding(key, key_part_spec, k_full_shape, mesh=get_mesh()).global_tensor
    #     value = xs.disable_manual_sharding(value, value_part_spec, v_full_shape, mesh=get_mesh()).global_tensor
    #     key_cache = xs.disable_manual_sharding(key_cache, key_cache_spec, key_cache_full_shape, mesh=get_mesh()).global_tensor
    #     value_cache = xs.disable_manual_sharding(value_cache, value_cache_spec, value_cache_full_shape, mesh=get_mesh()).global_tensor
    #     # slot_mapping = xs.disable_manual_sharding(slot_mapping, slot_mapping_spec, slot_mapping_full_shape, mesh=get_mesh()).global_tensor
    #     # xs.mark_sharding()

# [[0 1]] [[2 3]]
# [[4 5]] [[6 7]]

def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    pages_per_compute_block: int,
    megacore_mode: Optional[str],
) -> torch.Tensor:
    batch_size = query.shape[0]
    if megacore_mode == "batch" and batch_size % 2 != 0:
        megacore_mode = None
    else:
        megacore_mode = megacore_mode

    # NOTE(woosuk): A temporary workaround to avoid the error:
    # "xla::paged_attention() Expected a value of type 'str' for
    # argument 'megacore_mode' but instead found type 'NoneType'."
    if megacore_mode is not None:
        output = torch.ops.xla.paged_attention(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            pages_per_compute_block,
            megacore_mode=megacore_mode,
        )
    else:
        output = torch.ops.xla.paged_attention(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            pages_per_compute_block,
        )
    return output
