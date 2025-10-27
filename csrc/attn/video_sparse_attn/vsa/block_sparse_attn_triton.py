"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao
(https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team
"""

import pytest
import torch
import triton
import triton.language as tl

# ──────────────────────────── SPARSE ADDITION BEGIN ───────────────────────────
import math  # small utility needed by the sparse wrapper
# ──────────────────────────── SPARSE ADDITION END ─────────────────────────────


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.

def get_device_triton_configs():
    """Get device-specific Triton configurations based on detected GPU."""
    try:
        # Detect device type using torch
        import torch
        if torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip is not None:
            device_name = torch.cuda.get_device_name().lower()
            
            if "mi300x" in device_name or "mi300" in device_name:
                # MI300X: High performance configs (same shared memory as MI210)
                return [
                    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
                    for BM in [32, 64]\
                    for BN in [32, 64]\
                    for s in [1, 2, 3]\
                    for w in [2, 4, 8]\
                ]
            elif "mi250" in device_name or "m250" in device_name:
                # MI250: Aggressive configs for 128KB shared memory
                return [
                    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
                    for BM in [32, 64, 128]\
                    for BN in [32, 64, 128]\
                    for s in [2, 3, 4]\
                    for w in [4, 8, 16]\
                ]
            elif "w7800" in device_name or "radeon pro" in device_name:
                # W7800: Most conservative configs for 32KB shared memory
                return [
                    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
                    for BM in [16, 32]\
                    for BN in [16, 32]\
                    for s in [1, 2]\
                    for w in [2, 4]\
                ]
            else:
                # MI210 and generic ROCm: Current optimized configs
                return [
                    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
                    for BM in [32, 64]\
                    for BN in [32, 64]\
                    for s in [1, 2, 3]\
                    for w in [2, 4]\
                ]
    except:
        pass
    
    # Fallback: ROCm-optimized configs for MI210 (64KB shared memory limit)
    return [
        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
        for BM in [32, 64]\
        for BN in [32, 64]\
        for s in [1, 2, 3]\
        for w in [2, 4]\
    ]

# Get device-specific configurations
configs = get_device_triton_configs()

# ──────────────────────────── SPARSE ADDITION BEGIN ───────────────────────────
@triton.autotune(configs, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd_sparse(Q, K, V, sm_scale,                         #
                     q2k_index, q2k_num, max_kv_blks,           #
                     variable_block_sizes,
                     M, Out,                                    #
                     stride_qz, stride_qh, stride_qm, stride_qk,
                     stride_kz, stride_kh, stride_kn, stride_kk,
                     stride_vz, stride_vh, stride_vk, stride_vn,
                     stride_oz, stride_oh, stride_om, stride_on,
                     Z, H, N_CTX,                               #
                     HEAD_DIM: tl.constexpr,                    #
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     STAGE: tl.constexpr):
    """
    64×64 **block-sparse** forward kernel. Back-prop kernels remain dense
    (32×64 and 64×32) – memory footprint unchanged.
    """

    # ----- program-id mapping -----
    q_blk   = tl.program_id(0)          # Q-tile index
    off_hz  = tl.program_id(1)          # fused (batch, head)
    b       = off_hz // H
    h       = off_hz %  H
    q_tiles = N_CTX // BLOCK_M
    meta_base = ((b * H + h) * q_tiles + q_blk)

    kv_blocks = tl.load(q2k_num  + meta_base)                 # int32
    kv_ptr    = q2k_index + meta_base * max_kv_blks           # ptr to list

    # ----- base pointers -----
    qvk_off = (b.to(tl.int64) * stride_qz +
               h.to(tl.int64) * stride_qh)

    Q_ptr = tl.make_block_ptr(
        base=Q + qvk_off, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(q_blk * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))

    K_base = tl.make_block_ptr(
        base=K + qvk_off, shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))

    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_base = tl.make_block_ptr(
        base=V + qvk_off, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=v_order)

    O_ptr = tl.make_block_ptr(
        base=Out + qvk_off, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(q_blk * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))

    # ----- accumulators -----
    offs_m = q_blk * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # 1/ln2
    q = tl.load(Q_ptr)

    # ----- sparse loop over valid K/V tiles -----
    for i in range(0, kv_blocks):
        kv_idx = tl.load(kv_ptr + i).to(tl.int32)
        block_size = tl.load(variable_block_sizes + kv_idx)
        # Use direct pointer arithmetic instead of tl.advance for ROCm compatibility
        K_ptr = K_base + kv_idx * BLOCK_N * stride_kn
        V_ptr = V_base + kv_idx * BLOCK_N * stride_vk

        # Load K with proper indexing - K is [HEAD_DIM, N_CTX] so we need to transpose
        k_offsets = kv_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        k = tl.load(K + qvk_off + tl.arange(0, HEAD_DIM)[:, None] * stride_kk + k_offsets[None, :] * stride_kn)
        qk = tl.dot(q, k)
        # mask out invalid columns
        mask = tl.arange(0, BLOCK_N) < block_size
        qk = tl.where(mask[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        p = tl.math.exp2(qk * qk_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Load V with proper indexing - V is [N_CTX, HEAD_DIM]
        v_offsets = kv_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        v = tl.load(V + qvk_off + v_offsets[:, None] * stride_vk + tl.arange(0, HEAD_DIM)[None, :] * stride_vn)
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # ----- epilogue -----
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(M + off_hz * N_CTX + offs_m, m_i)
    tl.store(O_ptr, acc.to(Out.type.element_ty))
# ──────────────────────────── SPARSE ADDITION END ─────────────────────────────




@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, sm_scale,  #
                   DO,  #
                   M, D,  #
                   k2q_index, k2q_num, max_q_blks,
                   variable_block_sizes,
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d,  #
                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   # Filled in by the wrapper.
                   start_n, start_m, num_steps):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    step_m = BLOCK_M1
    kv_blk   = tl.program_id(0)          # Q-tile index
    off_hz  = tl.program_id(2)          # fused (batch, head)
    b       = off_hz // H
    h       = off_hz %  H
    q_tiles = N_CTX // BLOCK_N1
    meta_base = ((b * H + h) * q_tiles + kv_blk)

    q_blocks = tl.load(k2q_num  + meta_base)                 # int32
    q_ptr    = k2q_index + meta_base * max_q_blks           # ptr to list
    block_size = tl.load(variable_block_sizes + kv_blk)
    
    
        
    for blk_idx in range(q_blocks*2):
        block_sparse_offset = (tl.load(q_ptr + blk_idx//2).to(tl.int32)*2 + blk_idx%2) *step_m
        # Use direct indexing instead of pointer arithmetic
        qT_offsets = block_sparse_offset + tl.arange(0, BLOCK_M1)
        qT = tl.load(Q + qT_offsets[None, :] * stride_tok + offs_k[:, None] * stride_d)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = start_m + block_sparse_offset + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        mask = tl.arange(0, BLOCK_N1) < block_size
        pT = tl.where(mask[:, None], pT, 0.0)

        do_offsets = block_sparse_offset + tl.arange(0, BLOCK_M1)
        do = tl.load(DO + do_offsets[:, None] * stride_tok + offs_k[None, :] * stride_d)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.bfloat16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.bfloat16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
    return dk, dv



# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V,  #
                 do, m, D,
                 # shared by Q/K/V/DO.
                 q2k_index, q2k_num, max_kv_blks,
                 variable_block_sizes,
                 stride_tok, stride_d,  #
                 H, N_CTX,  #
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m, start_n, num_steps):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    step_n = BLOCK_N2
    
    q_blk   = tl.program_id(0)          # Q-tile index
    off_hz  = tl.program_id(2)          # fused (batch, head)
    b       = off_hz // H
    h       = off_hz %  H
    q_tiles = N_CTX // BLOCK_M2
    meta_base = ((b * H + h) * q_tiles + q_blk)

    kv_blocks = tl.load(q2k_num  + meta_base)                 # int32
    kv_ptr    = q2k_index + meta_base * max_kv_blks           # ptr to list
    
    
    for blk_idx in range(kv_blocks*2):
        block_sparse_offset = (tl.load(kv_ptr + blk_idx//2).to(tl.int32)*2 + blk_idx%2) *step_n
        block_size = tl.load(variable_block_sizes + blk_idx//2) - (blk_idx%2) * step_n
        # Use direct indexing instead of pointer arithmetic
        kT_offsets = block_sparse_offset + tl.arange(0, BLOCK_N2)
        vT_offsets = block_sparse_offset + tl.arange(0, BLOCK_N2)
        kT = tl.load(K + kT_offsets[None, :] * stride_tok + offs_k[:, None] * stride_d)
        vT = tl.load(V + vT_offsets[None, :] * stride_tok + offs_k[:, None] * stride_d)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        mask = tl.arange(0, BLOCK_N2) < block_size.to(tl.int32)
        p = tl.where(mask[None, :], p , 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.bfloat16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
    return dq



@triton.jit
def _attn_bwd(Q, K, V, sm_scale,  #
              DO,  #
              DQ, DK, DV,  #
              M, D,
              q2k_index, q2k_num, max_kv_blks,
              k2q_index, k2q_num, max_q_blks,
              variable_block_sizes,
              # shared by Q/K/V/DO.
              stride_z, stride_h, stride_tok, stride_d,  #
              H, N_CTX,  #
              BLOCK_M1: tl.constexpr,  #
              BLOCK_N1: tl.constexpr,  #
              BLOCK_M2: tl.constexpr,  #
              BLOCK_N2: tl.constexpr,  #
              HEAD_DIM: tl.constexpr):
    LN2 = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)


    num_steps = N_CTX // BLOCK_M1

    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, k, v, sm_scale,  #
        DO,  #
        M, D,  #
        k2q_index, k2q_num, max_q_blks,
        variable_block_sizes,
        stride_tok, stride_d,  #
        H, N_CTX,  #
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        start_n, start_m, num_steps  #
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = 0

    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    num_steps = N_CTX // BLOCK_N2
    dq = _attn_bwd_dq(dq, q, K, V,  #
                      do, m, D,  #
                      q2k_index, q2k_num, max_kv_blks,
                      variable_block_sizes,
                      stride_tok, stride_d,  #
                      H, N_CTX,  #
                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                      start_m, end_n, num_steps  #
                      )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)
    
    



# ──────────────────────────── SPARSE ADDITION BEGIN ───────────────────────────
def triton_block_sparse_attn_forward(q, k, v, q2k_index, q2k_num, variable_block_sizes):
    B, H, T, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)
    max_kv_blks = q2k_index.shape[-1]
    # Calculate the actual block size based on the block map dimensions
    num_blocks = q2k_num.shape[-1]
    block_size = T // num_blocks
    
    # Device-specific block size handling
    try:
        import torch
        if torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip is not None:
            device_name = torch.cuda.get_device_name().lower()
            
            if "w7800" in device_name or "radeon pro" in device_name:
                # W7800: Most conservative, force 16 or 32 element blocks
                target_block_size = 16 if block_size <= 16 else 32
            elif "mi250" in device_name or "m250" in device_name:
                # MI250: Can handle larger blocks due to 128KB shared memory
                if block_size <= 16:
                    target_block_size = 16
                elif block_size <= 32:
                    target_block_size = 32
                elif block_size <= 64:
                    target_block_size = 64
                else:
                    target_block_size = 128
            else:
                # MI210, MI300X, generic: Support 32 and 64 element blocks
                target_block_size = 32 if block_size <= 32 else 64
        else:
            # Non-ROCm: Use 32 as default
            target_block_size = 32
    except:
        # Fallback: Use 32 as default
        target_block_size = 32
    
    # Pad sequence to match target block size if needed
    if block_size != target_block_size:
        pad_length = target_block_size - (T % target_block_size) if T % target_block_size != 0 else 0
        if pad_length > 0:
            q = torch.nn.functional.pad(q, (0, 0, 0, pad_length), value=0)
            k = torch.nn.functional.pad(k, (0, 0, 0, pad_length), value=0)
            v = torch.nn.functional.pad(v, (0, 0, 0, pad_length), value=0)
            T = q.shape[2]
            num_blocks = T // target_block_size
            # Update q2k_num to match the new number of blocks
            q2k_num = torch.full((B, H, num_blocks), 1, device=q.device, dtype=torch.int32)
            q2k_index = torch.zeros((B, H, num_blocks, 1), device=q.device, dtype=torch.int32)
            for i in range(num_blocks):
                q2k_index[:, :, i, 0] = i
        block_size = target_block_size
    
    assert T % block_size == 0, f"T must be a multiple of {block_size}, but got {T}"
    assert T // block_size == num_blocks, f"shape mismatch, T // {block_size} = {T // block_size}, q2k_num.shape[-1] = {num_blocks}"
    
    o = torch.empty_like(q)
    M = torch.empty((B, H, T), dtype=torch.float32, device=q.device)

    grid = lambda _: (triton.cdiv(T, block_size), B * H, 1)
    _attn_fwd_sparse[grid](
        q, k, v, sm_scale,
        q2k_index, q2k_num, max_kv_blks,
        variable_block_sizes,
        M, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, T,
        HEAD_DIM=D, STAGE=3
    )

    return o, M

def triton_block_sparse_attn_backward(do, q, k, v, o, M, q2k_index, q2k_num, k2q_index, k2q_num, variable_block_sizes):
    assert do.is_contiguous()
    assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
    
    B, H, T, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    BATCH, N_HEAD, N_CTX = q.shape[:3]
    
    # Device-specific block sizes for backward pass
    try:
        import torch
        if torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip is not None:
            device_name = torch.cuda.get_device_name().lower()
            
            if "w7800" in device_name or "radeon pro" in device_name:
                # W7800: Most conservative block sizes for 32KB shared memory
                BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 16, 16, 16, 16
                PRE_BLOCK = 16
            elif "mi250" in device_name or "m250" in device_name:
                # MI250: Larger block sizes for 128KB shared memory
                BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 64, 64, 64, 64
                PRE_BLOCK = 64
            else:
                # MI210, MI300X, generic: Current optimized block sizes
                BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32
                PRE_BLOCK = 32
        else:
            # Non-ROCm: Use 32 as default
            BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32
            PRE_BLOCK = 32
    except:
        # Fallback: ROCm-optimized block sizes for MI210 shared memory constraints
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 32, 32, 32
        PRE_BLOCK = 32
    
    RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
    arg_k = k
    arg_k = arg_k * (sm_scale * RCP_LN2)
    assert N_CTX % PRE_BLOCK == 0
    pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
    delta = torch.empty_like(M)
    _attn_bwd_preprocess[pre_grid](
        o, do,  #
        delta,  #
        BATCH, N_HEAD, N_CTX,  #
        BLOCK_M=PRE_BLOCK, HEAD_DIM=D  #
    )
    
    
    max_q_blks = k2q_index.shape[-1]
    max_kv_blks = q2k_index.shape[-1]
    
    grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
    _attn_bwd[grid](
        q, arg_k, v, sm_scale, do, dq, dk, dv,  #
        M, delta,  #
        q2k_index, q2k_num, max_kv_blks,
        k2q_index, k2q_num, max_q_blks,
        variable_block_sizes,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        N_HEAD, N_CTX,  #
        BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
        BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
        HEAD_DIM=D  #
    )

    return dq, dk, dv


