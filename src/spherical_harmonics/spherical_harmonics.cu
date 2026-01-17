/**
 * Spherical Harmonics CUDA Kernel
 *
 * Computes real spherical harmonics Y_lm matching e3nn's component normalization convention.
 * Uses hardcoded polynomial expressions for maximum performance (no trig calls).
 *
 * e3nn convention:
 *   - Component normalization: |Y^l|^2 = 2l+1 on the unit sphere
 *   - L=1 ordering: (x, y, z) scaled by sqrt(3)
 *   - Higher L: polynomial recurrences in x, y, z
 *
 * Output indexing: [l=0], [l=1: 3 components], [l=2: 5 components], ...
 * Total components for L_max: (L_max+1)^2
 */

#include "spherical_harmonics.cuh"
#include <cmath>

namespace batteries {

// Compile-time square root for normalization constants
template <int N, int D = 1>
constexpr float sqrt_ratio() {
    // Using 32-bit float constants
    return sqrtf(static_cast<float>(N) / static_cast<float>(D));
}

/**
 * Spherical harmonics kernel using e3nn polynomial convention.
 * Each thread processes one edge vector.
 * Templated on L_MAX for compile-time unrolling.
 */
template <int L_MAX>
__global__ void spherical_harmonics_kernel(
    const float* __restrict__ edge_vectors,  // [N, 3]
    float* __restrict__ out,                  // [N, (L_MAX+1)^2]
    int64_t N
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    constexpr int num_components = (L_MAX + 1) * (L_MAX + 1);

    // Load unit vector
    const float x = edge_vectors[idx * 3 + 0];
    const float y = edge_vectors[idx * 3 + 1];
    const float z = edge_vectors[idx * 3 + 2];

    float* o = out + idx * num_components;

    // Precompute powers
    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;

    // L=0: Y_0 = 1
    o[0] = 1.0f;
    if constexpr (L_MAX == 0) return;

    // L=1: sqrt(3) * (x, y, z)
    constexpr float c1 = 1.7320508075688772f;  // sqrt(3)
    o[1] = c1 * x;
    o[2] = c1 * y;
    o[3] = c1 * z;
    if constexpr (L_MAX == 1) return;

    // L=2: 5 components
    // Following e3nn's polynomial formulas
    constexpr float sqrt15 = 3.872983346207417f;
    constexpr float sqrt5 = 2.23606797749979f;
    const float x2z2 = x2 + z2;

    o[4] = sqrt15 * x * z;                    // sh_2_0
    o[5] = sqrt15 * x * y;                    // sh_2_1
    o[6] = sqrt5 * (y2 - 0.5f * x2z2);       // sh_2_2
    o[7] = sqrt15 * y * z;                    // sh_2_3
    o[8] = 0.5f * sqrt15 * (z2 - x2);        // sh_2_4
    if constexpr (L_MAX == 2) return;

    // L=3: 7 components
    // Using recurrence from L=2
    constexpr float c3_0 = 1.0801234497346432f;   // (1/6) * sqrt(42)
    constexpr float c3_1 = 2.6457513110645907f;   // sqrt(7)
    constexpr float c3_2 = 1.6201851746019651f;   // (1/8) * sqrt(168)
    constexpr float c3_3 = 1.3228756555322954f;   // (1/2) * sqrt(7)

    const float sh_2_0 = o[4];
    const float sh_2_4 = o[8];

    o[9]  = c3_0 * (sh_2_0 * z + sh_2_4 * x);           // sh_3_0
    o[10] = c3_1 * sh_2_0 * y;                          // sh_3_1
    o[11] = c3_2 * (4.0f * y2 - x2z2) * x;             // sh_3_2
    o[12] = c3_3 * y * (2.0f * y2 - 3.0f * x2z2);      // sh_3_3
    o[13] = c3_2 * z * (4.0f * y2 - x2z2);             // sh_3_4
    o[14] = c3_1 * sh_2_4 * y;                          // sh_3_5
    o[15] = c3_0 * (sh_2_4 * z - sh_2_0 * x);          // sh_3_6
    if constexpr (L_MAX == 3) return;

    // L=4: 9 components - transcribed directly from e3nn
    const float sh_3_0 = o[9];
    const float sh_3_1 = o[10];
    const float sh_3_2 = o[11];
    const float sh_3_3 = o[12];
    const float sh_3_4 = o[13];
    const float sh_3_5 = o[14];
    const float sh_3_6 = o[15];

    // sh_4_0 = (3/4)*sqrt(2) * (sh_3_0*z + sh_3_6*x)
    o[16] = 1.0606601717798214f * (sh_3_0 * z + sh_3_6 * x);
    // sh_4_1 = (3/4)*sh_3_0*y + (3/8)*sqrt(6)*sh_3_1*z + (3/8)*sqrt(6)*sh_3_5*x
    o[17] = 0.75f * sh_3_0 * y + 0.9185586535436917f * sh_3_1 * z + 0.9185586535436917f * sh_3_5 * x;
    // sh_4_2 = -3/56*sqrt(14)*sh_3_0*z + (3/14)*sqrt(21)*sh_3_1*y + (3/56)*sqrt(210)*sh_3_2*z
    //        + (3/56)*sqrt(210)*sh_3_4*x + (3/56)*sqrt(14)*sh_3_6*x
    o[18] = -0.2004459314343183f * sh_3_0 * z + 0.9819805060619656f * sh_3_1 * y
          + 0.7763237542601484f * sh_3_2 * z + 0.7763237542601484f * sh_3_4 * x
          + 0.2004459314343183f * sh_3_6 * x;
    // sh_4_3 = -3/56*sqrt(42)*sh_3_1*z + (3/28)*sqrt(105)*sh_3_2*y + (3/28)*sqrt(70)*sh_3_3*x
    //        + (3/56)*sqrt(42)*sh_3_5*x
    o[19] = -0.3471825374147068f * sh_3_1 * z + 1.0978875820670997f * sh_3_2 * y
          + 0.8964214570007952f * sh_3_3 * x + 0.3471825374147068f * sh_3_5 * x;
    // sh_4_4 = -3/28*sqrt(42)*sh_3_2*x + (3/7)*sqrt(7)*sh_3_3*y - 3/28*sqrt(42)*sh_3_4*z
    o[20] = -0.6943650748294136f * sh_3_2 * x + 1.1338934190276817f * sh_3_3 * y
          - 0.6943650748294136f * sh_3_4 * z;
    // sh_4_5 = -3/56*sqrt(42)*sh_3_1*x + (3/28)*sqrt(70)*sh_3_3*z + (3/28)*sqrt(105)*sh_3_4*y
    //        - 3/56*sqrt(42)*sh_3_5*z
    o[21] = -0.3471825374147068f * sh_3_1 * x + 0.8964214570007952f * sh_3_3 * z
          + 1.0978875820670997f * sh_3_4 * y - 0.3471825374147068f * sh_3_5 * z;
    // sh_4_6 = -3/56*sqrt(14)*sh_3_0*x - 3/56*sqrt(210)*sh_3_2*x + (3/56)*sqrt(210)*sh_3_4*z
    //        + (3/14)*sqrt(21)*sh_3_5*y - 3/56*sqrt(14)*sh_3_6*z
    o[22] = -0.2004459314343183f * sh_3_0 * x - 0.7763237542601484f * sh_3_2 * x
          + 0.7763237542601484f * sh_3_4 * z + 0.9819805060619656f * sh_3_5 * y
          - 0.2004459314343183f * sh_3_6 * z;
    // sh_4_7 = -3/8*sqrt(6)*sh_3_1*x + (3/8)*sqrt(6)*sh_3_5*z + (3/4)*sh_3_6*y
    o[23] = -0.9185586535436917f * sh_3_1 * x + 0.9185586535436917f * sh_3_5 * z + 0.75f * sh_3_6 * y;
    // sh_4_8 = (3/4)*sqrt(2)*(-sh_3_0*x + sh_3_6*z)
    o[24] = 1.0606601717798214f * (-sh_3_0 * x + sh_3_6 * z);
    if constexpr (L_MAX == 4) return;

    // L=5: 11 components - transcribed directly from e3nn
    const float sh_4_0 = o[16];
    const float sh_4_1 = o[17];
    const float sh_4_2 = o[18];
    const float sh_4_3 = o[19];
    const float sh_4_4 = o[20];
    const float sh_4_5 = o[21];
    const float sh_4_6 = o[22];
    const float sh_4_7 = o[23];
    const float sh_4_8 = o[24];

    // sh_5_0 = (1/10)*sqrt(110)*(sh_4_0*z + sh_4_8*x)
    o[25] = 1.0488088481701516f * (sh_4_0 * z + sh_4_8 * x);
    // sh_5_1 = (1/5)*sqrt(11)*sh_4_0*y + (1/5)*sqrt(22)*sh_4_1*z + (1/5)*sqrt(22)*sh_4_7*x
    o[26] = 0.6633249580710801f * sh_4_0 * y + 0.9380831519646859f * sh_4_1 * z
          + 0.9380831519646859f * sh_4_7 * x;
    // sh_5_2 = -1/30*sqrt(22)*sh_4_0*z + (4/15)*sqrt(11)*sh_4_1*y + (1/15)*sqrt(154)*sh_4_2*z
    //        + (1/15)*sqrt(154)*sh_4_6*x + (1/30)*sqrt(22)*sh_4_8*x
    o[27] = -0.1563471919941143f * sh_4_0 * z + 0.8844332774281066f * sh_4_1 * y
          + 0.8273115763993905f * sh_4_2 * z + 0.8273115763993905f * sh_4_6 * x
          + 0.1563471919941143f * sh_4_8 * x;
    // sh_5_3 = -1/30*sqrt(66)*sh_4_1*z + (1/15)*sqrt(231)*sh_4_2*y + (1/30)*sqrt(462)*sh_4_3*z
    //        + (1/30)*sqrt(462)*sh_4_5*x + (1/30)*sqrt(66)*sh_4_7*x
    o[28] = -0.2708012801545320f * sh_4_1 * z + 1.0132456102380443f * sh_4_2 * y
          + 0.7164728420068226f * sh_4_3 * z + 0.7164728420068226f * sh_4_5 * x
          + 0.2708012801545320f * sh_4_7 * x;
    // sh_5_4 = -1/15*sqrt(33)*sh_4_2*z + (2/15)*sqrt(66)*sh_4_3*y + (1/15)*sqrt(165)*sh_4_4*x
    //        + (1/15)*sqrt(33)*sh_4_6*x
    o[29] = -0.3829708431025353f * sh_4_2 * z + 1.0832051206181281f * sh_4_3 * y
          + 0.8563488385776753f * sh_4_4 * x + 0.3829708431025353f * sh_4_6 * x;
    // sh_5_5 = -1/15*sqrt(110)*sh_4_3*x + (1/3)*sqrt(11)*sh_4_4*y - 1/15*sqrt(110)*sh_4_5*z
    o[30] = -0.6992058987801010f * sh_4_3 * x + 1.1055415967851332f * sh_4_4 * y
          - 0.6992058987801010f * sh_4_5 * z;
    // sh_5_6 = -1/15*sqrt(33)*sh_4_2*x + (1/15)*sqrt(165)*sh_4_4*z + (2/15)*sqrt(66)*sh_4_5*y
    //        - 1/15*sqrt(33)*sh_4_6*z
    o[31] = -0.3829708431025353f * sh_4_2 * x + 0.8563488385776753f * sh_4_4 * z
          + 1.0832051206181281f * sh_4_5 * y - 0.3829708431025353f * sh_4_6 * z;
    // sh_5_7 = -1/30*sqrt(66)*sh_4_1*x - 1/30*sqrt(462)*sh_4_3*x + (1/30)*sqrt(462)*sh_4_5*z
    //        + (1/15)*sqrt(231)*sh_4_6*y - 1/30*sqrt(66)*sh_4_7*z
    o[32] = -0.2708012801545320f * sh_4_1 * x - 0.7164728420068226f * sh_4_3 * x
          + 0.7164728420068226f * sh_4_5 * z + 1.0132456102380443f * sh_4_6 * y
          - 0.2708012801545320f * sh_4_7 * z;
    // sh_5_8 = -1/30*sqrt(22)*sh_4_0*x - 1/15*sqrt(154)*sh_4_2*x + (1/15)*sqrt(154)*sh_4_6*z
    //        + (4/15)*sqrt(11)*sh_4_7*y - 1/30*sqrt(22)*sh_4_8*z
    o[33] = -0.1563471919941143f * sh_4_0 * x - 0.8273115763993905f * sh_4_2 * x
          + 0.8273115763993905f * sh_4_6 * z + 0.8844332774281066f * sh_4_7 * y
          - 0.1563471919941143f * sh_4_8 * z;
    // sh_5_9 = -1/5*sqrt(22)*sh_4_1*x + (1/5)*sqrt(22)*sh_4_7*z + (1/5)*sqrt(11)*sh_4_8*y
    o[34] = -0.9380831519646859f * sh_4_1 * x + 0.9380831519646859f * sh_4_7 * z
          + 0.6633249580710801f * sh_4_8 * y;
    // sh_5_10 = (1/10)*sqrt(110)*(-sh_4_0*x + sh_4_8*z)
    o[35] = 1.0488088481701516f * (-sh_4_0 * x + sh_4_8 * z);
    if constexpr (L_MAX == 5) return;

    // L=6: 13 components - transcribed directly from e3nn
    const float sh_5_0 = o[25];
    const float sh_5_1 = o[26];
    const float sh_5_2 = o[27];
    const float sh_5_3 = o[28];
    const float sh_5_4 = o[29];
    const float sh_5_5 = o[30];
    const float sh_5_6 = o[31];
    const float sh_5_7 = o[32];
    const float sh_5_8 = o[33];
    const float sh_5_9 = o[34];
    const float sh_5_10 = o[35];

    // sh_6_0 = (1/6)*sqrt(39)*(sh_5_0*z + sh_5_10*x)
    o[36] = 1.0408329997330663f * (sh_5_0 * z + sh_5_10 * x);
    // sh_6_1 = (1/6)*sqrt(13)*sh_5_0*y + (1/12)*sqrt(130)*sh_5_1*z + (1/12)*sqrt(130)*sh_5_9*x
    o[37] = 0.6009252125773314f * sh_5_0 * y + 0.9501461875826149f * sh_5_1 * z
          + 0.9501461875826149f * sh_5_9 * x;
    // sh_6_2 = -1/132*sqrt(286)*sh_5_0*z + (1/33)*sqrt(715)*sh_5_1*y + (1/132)*sqrt(286)*sh_5_10*x
    //        + (1/44)*sqrt(1430)*sh_5_2*z + (1/44)*sqrt(1430)*sh_5_8*x
    o[38] = -0.1281176857976346f * sh_5_0 * z + 0.8102873913406630f * sh_5_1 * y
          + 0.1281176857976346f * sh_5_10 * x + 0.8594395636904107f * sh_5_2 * z
          + 0.8594395636904107f * sh_5_8 * x;
    // sh_6_3 = -1/132*sqrt(858)*sh_5_1*z + (1/22)*sqrt(429)*sh_5_2*y + (1/22)*sqrt(286)*sh_5_3*z
    //        + (1/22)*sqrt(286)*sh_5_7*x + (1/132)*sqrt(858)*sh_5_9*x
    o[39] = -0.2219063411496486f * sh_5_1 * z + 0.9414688716912718f * sh_5_2 * y
          + 0.7687061147858074f * sh_5_3 * z + 0.7687061147858074f * sh_5_7 * x
          + 0.2219063411496486f * sh_5_9 * x;
    // sh_6_4 = -1/66*sqrt(429)*sh_5_2*z + (2/33)*sqrt(286)*sh_5_3*y + (1/66)*sqrt(2002)*sh_5_4*z
    //        + (1/66)*sqrt(2002)*sh_5_6*x + (1/66)*sqrt(429)*sh_5_8*x
    o[40] = -0.3138229572304240f * sh_5_2 * z + 1.0249414863810766f * sh_5_3 * y
          + 0.6779350703393059f * sh_5_4 * z + 0.6779350703393059f * sh_5_6 * x
          + 0.3138229572304240f * sh_5_8 * x;
    // sh_6_5 = -1/66*sqrt(715)*sh_5_3*z + (1/66)*sqrt(5005)*sh_5_4*y + (1/66)*sqrt(3003)*sh_5_5*x
    //        + (1/66)*sqrt(715)*sh_5_7*x
    o[41] = -0.4051436956703315f * sh_5_3 * z + 1.0719094639893330f * sh_5_4 * y
          + 0.8302975005345611f * sh_5_5 * x + 0.4051436956703315f * sh_5_7 * x;
    // sh_6_6 = -1/66*sqrt(2145)*sh_5_4*x + (1/11)*sqrt(143)*sh_5_5*y - 1/66*sqrt(2145)*sh_5_6*z
    o[42] = -0.7017294652672371f * sh_5_4 * x + 1.0871146130092180f * sh_5_5 * y
          - 0.7017294652672371f * sh_5_6 * z;
    // sh_6_7 = -1/66*sqrt(715)*sh_5_3*x + (1/66)*sqrt(3003)*sh_5_5*z + (1/66)*sqrt(5005)*sh_5_6*y
    //        - 1/66*sqrt(715)*sh_5_7*z
    o[43] = -0.4051436956703315f * sh_5_3 * x + 0.8302975005345611f * sh_5_5 * z
          + 1.0719094639893330f * sh_5_6 * y - 0.4051436956703315f * sh_5_7 * z;
    // sh_6_8 = -1/66*sqrt(429)*sh_5_2*x - 1/66*sqrt(2002)*sh_5_4*x + (1/66)*sqrt(2002)*sh_5_6*z
    //        + (2/33)*sqrt(286)*sh_5_7*y - 1/66*sqrt(429)*sh_5_8*z
    o[44] = -0.3138229572304240f * sh_5_2 * x - 0.6779350703393059f * sh_5_4 * x
          + 0.6779350703393059f * sh_5_6 * z + 1.0249414863810766f * sh_5_7 * y
          - 0.3138229572304240f * sh_5_8 * z;
    // sh_6_9 = -1/132*sqrt(858)*sh_5_1*x - 1/22*sqrt(286)*sh_5_3*x + (1/22)*sqrt(286)*sh_5_7*z
    //        + (1/22)*sqrt(429)*sh_5_8*y - 1/132*sqrt(858)*sh_5_9*z
    o[45] = -0.2219063411496486f * sh_5_1 * x - 0.7687061147858074f * sh_5_3 * x
          + 0.7687061147858074f * sh_5_7 * z + 0.9414688716912718f * sh_5_8 * y
          - 0.2219063411496486f * sh_5_9 * z;
    // sh_6_10 = -1/132*sqrt(286)*sh_5_0*x - 1/132*sqrt(286)*sh_5_10*z - 1/44*sqrt(1430)*sh_5_2*x
    //         + (1/44)*sqrt(1430)*sh_5_8*z + (1/33)*sqrt(715)*sh_5_9*y
    o[46] = -0.1281176857976346f * sh_5_0 * x - 0.1281176857976346f * sh_5_10 * z
          - 0.8594395636904107f * sh_5_2 * x + 0.8594395636904107f * sh_5_8 * z
          + 0.8102873913406630f * sh_5_9 * y;
    // sh_6_11 = -1/12*sqrt(130)*sh_5_1*x + (1/6)*sqrt(13)*sh_5_10*y + (1/12)*sqrt(130)*sh_5_9*z
    o[47] = -0.9501461875826149f * sh_5_1 * x + 0.6009252125773314f * sh_5_10 * y
          + 0.9501461875826149f * sh_5_9 * z;
    // sh_6_12 = (1/6)*sqrt(39)*(-sh_5_0*x + sh_5_10*z)
    o[48] = 1.0408329997330663f * (-sh_5_0 * x + sh_5_10 * z);
}

// Kernel launch wrapper
template <int L_MAX>
cudaError_t spherical_harmonics(
    const float* edge_vectors,
    float* Y_lm,
    int64_t N,
    cudaStream_t stream
) {
    if (N == 0) return cudaSuccess;

    constexpr int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    spherical_harmonics_kernel<L_MAX><<<num_blocks, block_size, 0, stream>>>(
        edge_vectors, Y_lm, N
    );

    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t spherical_harmonics<0>(
    const float*, float*, int64_t, cudaStream_t);
template cudaError_t spherical_harmonics<1>(
    const float*, float*, int64_t, cudaStream_t);
template cudaError_t spherical_harmonics<2>(
    const float*, float*, int64_t, cudaStream_t);
template cudaError_t spherical_harmonics<3>(
    const float*, float*, int64_t, cudaStream_t);
template cudaError_t spherical_harmonics<4>(
    const float*, float*, int64_t, cudaStream_t);
template cudaError_t spherical_harmonics<5>(
    const float*, float*, int64_t, cudaStream_t);
template cudaError_t spherical_harmonics<6>(
    const float*, float*, int64_t, cudaStream_t);

// Runtime dispatch
cudaError_t spherical_harmonics_dispatch(
    const float* edge_vectors,
    float* Y_lm,
    int64_t N,
    int L_max,
    cudaStream_t stream
) {
    switch (L_max) {
        case 0: return spherical_harmonics<0>(edge_vectors, Y_lm, N, stream);
        case 1: return spherical_harmonics<1>(edge_vectors, Y_lm, N, stream);
        case 2: return spherical_harmonics<2>(edge_vectors, Y_lm, N, stream);
        case 3: return spherical_harmonics<3>(edge_vectors, Y_lm, N, stream);
        case 4: return spherical_harmonics<4>(edge_vectors, Y_lm, N, stream);
        case 5: return spherical_harmonics<5>(edge_vectors, Y_lm, N, stream);
        case 6: return spherical_harmonics<6>(edge_vectors, Y_lm, N, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

} // namespace batteries
