#include <immintrin.h>

namespace avx2_infernal_math
{
  constexpr float k_pi = 3.14159274101;             // IEEE-754 FLOAT
  constexpr float k_pi_neg = -0.318309873343;       // IEEE-754 FLOAT
  constexpr float k_pi_recip = 0.318309873343;      // IEEE-754 FLOAT
  constexpr float k_pi_half = 1.57079637051;        // IEEE-754 FLOAT
  constexpr float k_pi_half_neg = -1.57079637051;   // IEEE-754 FLOAT
  constexpr float k_pi_half_recip = 0.636619746685; // IEEE-754 FLOAT

  namespace fast
  {

    // only valid from -pi/2 to pi/2
    void k_cos(float *in, float *out, size_t size)
    {
      const __m256 k1 = _mm256_set1_ps(-0.497228264809);
      const __m256 k2 = _mm256_set1_ps(0.0375291258097);
      const __m256 one = _mm256_set1_ps(1.0);
      for (auto *i = arr; i <= in + size; i = i + 8)
      {
        __m256 x = _mm256_loadu_ps(i);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        __m256 result = _mm256_fmadd_ps(x4, k2, _mm256_fmadd_ps(x2, k1, one));
        _mm256_store_ps(out, result);
      }
    }

    // only valid from -pi/2 to pi/2
    void k_sin(float *in, float *out, size_t size)
    {
      const __m256 k1 = _mm256_set1_ps(-0.497228264809);
      const __m256 k2 = _mm256_set1_ps(0.0375291258097);
      const __m256 one = _mm256_set1_ps(1.0);
      const __m256 pi_half = _mm256_set1_ps(k_pi_half);
      for (auto *i = in; i <= in + size; i = i + 8)
      {
        __m256 x1 = _mm256_loadu_ps(i);
        __m256 x1 = _mm256_add_ps(x1, pi_half);
        __m256 x2 = _mm256_mul_ps(x1, x1);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        __m256 result = _mm256_fmadd_ps(x4, k2, _mm256_fmadd_ps(x2, k1, one));
        _mm256_store_ps(out, result);
      }
    }
  } // namespace fast

  namespace slow
  {
    void k_cos(float *in, float *out, size_t size)
    {
      const __m256 k1 = _mm256_set1_ps(-0.497228264809);
      const __m256 k2 = _mm256_set1_ps(0.0375291258097);
      const __m256 pi_recip = _mm256_set1_ps(k_pi_recip);
      const __m256 pi = _mm256_set1_ps(k_pi);
      const __m256 pi_half_recip = _mm256_set1_ps(k_pi_recip);
      const __m256 one = _mm256_set1_ps(1.0);
      const __m256 half = _mm256_set1_ps(0.5);
      const __m256 two = _mm256_set1_ps(2.0);
      const __m256 pi_half = _mm256_set1_ps(k_pi_half);
      for (auto *i = in; i <= in + size; i = i + 8)
      {
        __m256 x1 = _mm256_loadu_ps(i);
        __m256 aa = _mm256_mul_ps(x1, pi_half_recip);
        __m256 bb = _mm256_round_ps(aa, 0);
        __m256 cc = _mm256_fmsub_ps(bb, pi, x1);
        __m256 x2 = _mm256_mul_ps(cc, cc);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        __m256 dd = _mm256_fmadd_ps(x2, k1, one);
        __m256 ee = _mm256_fmadd_ps(x4, k2, dd);
        __m256 ff = _mm256_fmadd_ps(x1, pi_recip, half);
        __m256 gg = _mm256_ceil_ps(ff);
        __m256 hh = _mm256_fmsub_ps(gg, half, half);
        __m256 ii = _mm256_ceil_ps(hh);
        __m256 jj = _mm256_fmadd_ps(ii, two, one);
        __m256 kk = _mm256_sub_ps(gg, jj);
        __m256 ll = _mm256_fmadd_ps(kk, two, one);
        __m256 mm = _mm256_mul_ps(ee, ll);
        _mm256_store_ps(i, mm);
      }
    }

    void k_sin(float *in, float *out, size_t size)
    {
      const __m256 k1 = _mm256_set1_ps(-0.497228264809);
      const __m256 k2 = _mm256_set1_ps(0.0375291258097);
      const __m256 pi_recip = _mm256_set1_ps(k_pi_recip);
      const __m256 pi = _mm256_set1_ps(k_pi);
      const __m256 pi_half_recip = _mm256_set1_ps(k_pi_recip);
      const __m256 one = _mm256_set1_ps(1.0);
      const __m256 half = _mm256_set1_ps(0.5);
      const __m256 two = _mm256_set1_ps(2.0);
      for (auto *i = in; i <= in + size; i = i + 8)
      {
        __m256 x0 = _mm256_loadu_ps(i);
        __m256 x1 = _mm256_add_ps(x0, pi_half);
        __m256 aa = _mm256_mul_ps(x1, pi_half_recip);
        __m256 bb = _mm256_round_ps(aa, 0);
        __m256 cc = _mm256_fmsub_ps(bb, pi, x1);
        __m256 x2 = _mm256_mul_ps(cc, cc);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        __m256 dd = _mm256_fmadd_ps(x2, k1, one);
        __m256 ee = _mm256_fmadd_ps(x4, k2, dd);
        __m256 ff = _mm256_fmadd_ps(x1, pi_recip, half);
        __m256 gg = _mm256_ceil_ps(ff);
        __m256 hh = _mm256_fmsub_ps(gg, half, half);
        __m256 ii = _mm256_ceil_ps(hh);
        __m256 jj = _mm256_fmadd_ps(ii, two, one);
        __m256 kk = _mm256_sub_ps(gg, jj);
        __m256 ll = _mm256_fmadd_ps(kk, two, one);
        __m256 mm = _mm256_mul_ps(ee, ll);
        _mm256_store_ps(i, mm);
      }
    }
  } // namespace slow

} // namespace avx2_infernal_math
