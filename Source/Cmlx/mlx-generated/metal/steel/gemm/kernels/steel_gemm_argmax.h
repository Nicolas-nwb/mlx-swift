// Copyright © 2026 Apple Inc.

using namespace mlx::steel;

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

struct ArgMaxAddMMPair {
  float value;
  uint index;
};

METAL_FUNC ArgMaxAddMMPair argmax_addmm_choose(
    ArgMaxAddMMPair best,
    float value,
    uint index) {
  if (best.value < value || (best.value == value && best.index > index)) {
    return ArgMaxAddMMPair{value, index};
  }
  return best;
}

template <
    typename T,
    typename MMA,
    int BM,
    int BN,
    int WM,
    int WN>
METAL_FUNC void argmax_addmm_store_tile(
    thread MMA& mma_op,
    device float* max_values,
    device uint* max_indices,
    threadgroup float* row_values,
    threadgroup uint* row_indices,
    const constant GEMMParams* params,
    int c_row,
    int c_col,
    short tgp_bm,
    short tgp_bn,
    uint simd_lane_id,
    uint simd_group_id) {
  constexpr int groups_per_tile = WM * WN;
  uint lane_base = simd_lane_id < 16
      ? ((simd_lane_id / 2) & 3) * 2
      : 16 + (((simd_lane_id - 16) / 2) & 3) * 2;

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < MMA::TM; i++) {
    ArgMaxAddMMPair local{Limits<float>::min, uint(-1)};
    const int row_in_tile = mma_op.sm + i * MMA::TM_stride;

    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < MMA::TN; j++) {
      thread const auto& accum = mma_op.Ctile.frag_at(i, j);
      const int col_in_tile_base = mma_op.sn + j * MMA::TN_stride;

      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < decltype(mma_op.Ctile)::kElemsPerFrag; k++) {
        const int col_in_tile = col_in_tile_base + k;
        if (row_in_tile < tgp_bm && col_in_tile < tgp_bn) {
          const uint index = uint(c_col + col_in_tile);
          const float value = float(static_cast<T>(accum[k]));
          local = argmax_addmm_choose(local, value, index);
        }
      }
    }

    ArgMaxAddMMPair best{
        simd_shuffle(local.value, lane_base),
        simd_shuffle(local.index, lane_base)};
    best = argmax_addmm_choose(
        best,
        simd_shuffle(local.value, lane_base + 1),
        simd_shuffle(local.index, lane_base + 1));
    best = argmax_addmm_choose(
        best,
        simd_shuffle(local.value, lane_base + 8),
        simd_shuffle(local.index, lane_base + 8));
    best = argmax_addmm_choose(
        best,
        simd_shuffle(local.value, lane_base + 9),
        simd_shuffle(local.index, lane_base + 9));

    if (simd_lane_id == lane_base && row_in_tile < tgp_bm) {
      const int slot = row_in_tile * groups_per_tile + simd_group_id;
      row_values[slot] = best.value;
      row_indices[slot] = best.index;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group_id != 0) {
    return;
  }

  for (uint row_in_tile = simd_lane_id; row_in_tile < uint(tgp_bm);
       row_in_tile += 32) {
    ArgMaxAddMMPair best{Limits<float>::min, uint(-1)};
    STEEL_PRAGMA_UNROLL
    for (int group = 0; group < groups_per_tile; group++) {
      const int slot = int(row_in_tile) * groups_per_tile + group;
      best = argmax_addmm_choose(best, row_values[slot], row_indices[slot]);
    }

    const int out_offset = (c_row + int(row_in_tile)) * params->tiles_n +
        (c_col / BN);
    max_values[out_offset] = best.value;
    max_indices[out_offset] = best.index;
  }
}

// clang-format off
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM* WN * 32)]] void gemm_argmax(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    const device T* C [[buffer(2)]],
    device float* max_values [[buffer(3)]],
    device uint* max_indices [[buffer(4)]],
    const constant GEMMParams* params [[buffer(5)]],
    const constant GEMMAddMMParams* addmm_params [[buffer(6)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on
  (void)lid;

  using gemm_kernel = GEMMKernel<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      transpose_a,
      transpose_b,
      true,
      true,
      AccumType>;

  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;

  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }

  A += params->batch_stride_a * tid.z;
  B += params->batch_stride_b * tid.z;
  C += addmm_params->batch_stride_c * tid.z;

  max_values += params->batch_stride_d * tid.z;
  max_indices += params->batch_stride_d * tid.z;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  threadgroup_barrier(mem_flags::mem_none);

  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  C += c_row_long * addmm_params->ldc + c_col_long * addmm_params->fdc;

  thread mma_t mma_op(simd_group_id, simd_lane_id);

  thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

  const short tgp_bm = align_M ? BM : short(min(BM, params->M - c_row));
  const short tgp_bn = align_N ? BN : short(min(BN, params->N - c_col));

  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  if (!align_K) {
    const int k_last = params->gemm_k_iterations_aligned * BK;
    const int k_remain = params->K - k_last;
    const size_t k_jump_a =
        transpose_a ? params->lda * size_t(k_last) : size_t(k_last);
    const size_t k_jump_b =
        transpose_b ? size_t(k_last) : params->ldb * size_t(k_last);

    loader_a.src += k_jump_a;
    loader_b.src += k_jump_b;

    const short2 tile_dims_A =
        transpose_a ? short2(tgp_bm, k_remain) : short2(k_remain, tgp_bm);
    const short2 tile_dims_B =
        transpose_b ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

    loader_a.load_safe(tile_dims_A);
    loader_b.load_safe(tile_dims_B);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);

    loader_a.src -= k_jump_a;
    loader_b.src -= k_jump_b;
  }

  if (align_M && align_N) {
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
    threadgroup_barrier(mem_flags::mem_none);
  } else {
    const int leftover_bk = 0;
    if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<true, true, true>{});
    } else if (align_N || tgp_bn == BN) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<false, true, true>{});
    } else if (align_M || tgp_bm == BM) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<true, false, true>{});
    } else {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<false, false, true>{});
    }
  }

  const TransformAdd<AccumType, AccumType> epilogue_op_add(
      addmm_params->alpha, addmm_params->beta);
  mma_op.apply_epilogue_safe(
      C,
      addmm_params->ldc,
      addmm_params->fdc,
      short2(tgp_bn, tgp_bm),
      epilogue_op_add);

  threadgroup float row_values[BM * WM * WN];
  threadgroup uint row_indices[BM * WM * WN];
  const uint linear_lid = simd_group_id * 32 + simd_lane_id;
  for (uint slot = linear_lid; slot < uint(BM * WM * WN);
       slot += uint(32 * WM * WN)) {
    row_values[slot] = Limits<float>::min;
    row_indices[slot] = uint(-1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  argmax_addmm_store_tile<T, mma_t, BM, BN, WM, WN>(
      mma_op,
      max_values,
      max_indices,
      row_values,
      row_indices,
      params,
      c_row,
      c_col,
      tgp_bm,
      tgp_bn,
      simd_lane_id,
      simd_group_id);
}

[[kernel]] void argmax_addmm_finalize(
    const device float* max_values [[buffer(0)]],
    const device uint* max_indices [[buffer(1)]],
    device uint* out [[buffer(2)]],
    const constant int& tiles_n [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 gsize [[threads_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  (void)gsize;

  const uint row = gid.y;
  const device float* row_values = max_values + row * tiles_n;
  const device uint* row_indices = max_indices + row * tiles_n;

  ArgMaxAddMMPair best{Limits<float>::min, uint(-1)};
  for (uint tile = lid.x; tile < uint(tiles_n); tile += lsize.x) {
    best = argmax_addmm_choose(best, row_values[tile], row_indices[tile]);
  }

  for (uint offset = simd_size / 2; offset > 0; offset /= 2) {
    best = argmax_addmm_choose(
        best,
        simd_shuffle_down(best.value, offset),
        simd_shuffle_down(best.index, offset));
  }

  threadgroup ArgMaxAddMMPair local_data[32];
  if (simd_lane_id == 0) {
    local_data[simd_group_id] = best;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id != 0) {
    return;
  }

  uint simd_groups = ceildiv(lsize.x, simd_size);
  if (simd_lane_id < simd_groups) {
    best = local_data[simd_lane_id];
  } else {
    best = ArgMaxAddMMPair{Limits<float>::min, uint(-1)};
  }
  for (uint offset = simd_size / 2; offset > 0; offset /= 2) {
    best = argmax_addmm_choose(
        best,
        simd_shuffle_down(best.value, offset),
        simd_shuffle_down(best.index, offset));
  }

  if (lid.x == 0) {
    out[row] = best.index;
  }
}
