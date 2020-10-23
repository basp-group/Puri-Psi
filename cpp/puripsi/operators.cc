#include "puripsi/operators.h"
#include "puripsi/wproj_utilities.h"

namespace puripsi {

namespace details {
//! Construct gridding matrix
Sparse<t_complex> init_gridding_matrix_2d(const Vector<t_real> &u, const Vector<t_real> &v,
                                          const Vector<t_complex> &weights, const t_uint &imsizey_,
                                          const t_uint &imsizex_, const t_real &oversample_ratio,
                                          const std::function<t_real(t_real)> kernelu,
                                          const std::function<t_real(t_real)> kernelv, const t_real nshifty_, const t_real nshiftx_,
                                          const t_uint Ju /*= 4*/, const t_uint Jv /*= 4*/) {
  const t_uint ftsizev_ = std::floor(imsizey_ * oversample_ratio);
  const t_uint ftsizeu_ = std::floor(imsizex_ * oversample_ratio);
  const t_uint rows = u.size();
  const t_uint cols = ftsizeu_ * ftsizev_;
  auto omega_to_k = [](const Vector<t_real> &omega) {
    return omega.unaryExpr(std::ptr_fun<double, double>(std::floor));
  };
  const Vector<t_real> k_u = omega_to_k(u - Vector<t_real>::Constant(rows, Ju * 0.5));
  const Vector<t_real> k_v = omega_to_k(v - Vector<t_real>::Constant(rows, Jv * 0.5));
      if (u.size() != v.size())
        throw std::runtime_error("Size of u and v vectors are not the same for creating gridding matrix.");

  Sparse<t_complex> interpolation_matrix(rows, cols);
  interpolation_matrix.reserve(Vector<t_int>::Constant(rows, Ju * Jv));

  const t_complex I(0, 1);
  // const t_int ju_max = std::min(Ju + 1, ftsizeu_ + 1);
  // const t_int jv_max = std::min(Jv + 1, ftsizev_ + 1);
// #ifdef PURIPSI_OPENMP
// #pragma omp parallel for collapse(3) default(shared)
// #endif
//   for(t_int m = 0; m < rows; ++m) {
//     for(t_int ju = 1; ju < ju_max; ++ju) {
//       for(t_int jv = 1; jv < jv_max; ++jv) {
//         const t_uint q = utilities::mod(k_u(m) + ju, ftsizeu_);
//         const t_uint p = utilities::mod(k_v(m) + jv, ftsizev_);
//         const t_uint index = utilities::sub2ind(p, q, ftsizev_, ftsizeu_);
//         interpolation_matrix.coeffRef(m, index)
//             = std::exp(-2 * constant::pi * I * ((k_u(m) + ju) * 0.5 + (k_v(m) + jv) * 0.5))
//               * kernelu(u(m) - (k_u(m) + ju)) * kernelv(v(m) - (k_v(m) + jv)) * weights(m);
//       }
//     }
//   }

#ifdef PURIPSI_OPENMP
#pragma omp parallel for default(shared)
#endif
	for(t_int m = 0; m < rows; ++m) {
		//! adding phase shift from Fessler's code (need to undo scaling applied to the uv-coverage) 
		psi::t_real phase_shiftx = 0;
		if(std::abs(nshiftx_) > 0) 
				phase_shiftx = u(m)*(2 * constant::pi)*nshiftx_/ static_cast<t_real>(ftsizeu_); 
		psi::t_real phase_shifty = 0;
		if(std::abs(nshifty_) > 0)
			phase_shifty = v(m)*(2 * constant::pi)*nshifty_/ static_cast<t_real>(ftsizev_);

		for(t_int i = 1; i <= Ju; ++i) {
			const t_int q = utilities::mod(k_u(m) + i, ftsizeu_);

			psi::t_real cu = kernelu(u(m) - (k_u(m) + i));
			psi::t_complex phaseu = constant::pi * I * static_cast<t_real>(imsizex_-1)/static_cast<t_real>(ftsizeu_) * (u(m) - (k_u(m) + i));
			psi::t_complex coeffu = cu*std::exp(phaseu);

			for(t_int j = 1; j <= Jv; ++j) {
				const t_int p = utilities::mod(k_v(m) + j, ftsizev_);
				const t_int index = utilities::sub2ind(p, q, ftsizev_, ftsizeu_);
				
				psi::t_real cv = kernelv(v(m) - (k_v(m) + j));
				psi::t_complex phasev = constant::pi * I * static_cast<t_real>(imsizey_-1)/static_cast<t_real>(ftsizev_) * (v(m) - (k_v(m) + j));
				psi::t_complex coeffv = cv*std::exp(phasev);
				interpolation_matrix.coeffRef(m, index) = std::conj(coeffu * coeffv)*std::exp(I*(phase_shifty + phase_shiftx));

				// const t_int index_local = utilities::sub2ind(j-1, i-1, Jv, Ju);
				// full_fourier_indices(m, index_local) = index;
			}
		}
	}
  return interpolation_matrix;
}
 
//! Construct gridding matrix with w projection
// TODO: to be revised with Arwa (check this is properly fixed...)
Sparse<t_complex>
init_gridding_matrix_2d(const Vector<t_real> &u, const Vector<t_real> &v, const Vector<t_real> &w,
                        const Vector<t_complex> &weights, const t_uint &imsizey_,
                        const t_uint &imsizex_, const t_real oversample_ratio,
                        const std::function<t_real(t_real)> kernelu,
                        const std::function<t_real(t_real)> kernelv, 
			const std::function<t_complex(t_real, t_real, t_real)> kernelw, const t_real nshifty_, const t_real nshiftx_, const t_uint Ju /*= 4*/,
                        const t_uint Jv /*= 4*/, const t_uint Jw /*= 6*/, const bool w_term /*= false*/){
  if (!w_term)
    return init_gridding_matrix_2d(u, v, weights, imsizey_, imsizex_, oversample_ratio, kernelu,
                                   kernelv, nshifty_, nshiftx_, Ju, Jv);
  const t_uint ftsizev_ = std::floor(imsizey_ * oversample_ratio);
  const t_uint ftsizeu_ = std::floor(imsizex_ * oversample_ratio);
  const t_uint rows = u.size();
  const t_uint cols = ftsizeu_ * ftsizev_;
  if (u.size() != v.size())
    throw std::runtime_error(
        "Size of u and v vectors are not the same for creating gridding matrix.");
  if (u.size() != w.size())
    throw std::runtime_error(
        "Size of u and w vectors are not the same for creating gridding matrix.");
  if (u.size() != weights.size())
    throw std::runtime_error(
        "Size of u and w vectors are not the same for creating gridding matrix.");

  Sparse<t_complex> interpolation_matrix(rows, cols);
  const t_int Jwu = Jw + Ju - 1;
  const t_int Jwv = Jw + Jv - 1;
  interpolation_matrix.reserve(Vector<t_int>::Constant(rows, Jwv * Jwu));

  const t_complex I(0, 1);
#ifdef PURIPSI_OPENMP
#pragma omp parallel for default(shared)
#endif
  for (t_int m = 0; m < rows; ++m) {
    // w_projection convolution setup
    const Matrix<t_complex> projection_kernel =
        projection_kernels::projection(kernelu, kernelv, kernelw, u(m), v(m), w(m), Ju, Jv, Jw);

    const t_int kwu = std::floor(u(m) - Jwu * 0.5);
    const t_int kwv = std::floor(v(m) - Jwv * 0.5);
    for (t_int ju = 1; ju < Jwu + 1; ++ju) {
      for (t_int jv = 1; jv < Jwv + 1; ++jv) {
        const t_uint q = utilities::mod(kwu + ju, ftsizeu_);
        const t_uint p = utilities::mod(kwv + jv, ftsizev_);
        const t_uint index = utilities::sub2ind(p, q, ftsizev_, ftsizeu_);
        interpolation_matrix.insert(m, index) =
            std::exp(-2 * constant::pi * I * ((kwu + ju) * 0.5 + (kwv + jv) * 0.5)) *
            projection_kernel(jv - 1, ju - 1) * weights(m);
      }
    }
  }
  return interpolation_matrix;
}

Image<t_real> init_correction2d(const t_real &oversample_ratio, const t_uint &imsizey_,
                                const t_uint imsizex_,
                                const std::function<t_real(t_real)> ftkernelu,
                                const std::function<t_real(t_real)> ftkernelv) {

  psi::Array<psi::t_real> range_v;
	range_v.setLinSpaced(imsizey_, static_cast<psi::t_real>(-(imsizey_-1))/2., static_cast<psi::t_real>(imsizey_-1)/2.);  
	psi::Matrix<psi::t_real> ftv = (1. / range_v.unaryExpr(ftkernelv)).matrix();

	psi::Array<psi::t_real> range_u;
	range_u.setLinSpaced(imsizex_, static_cast<psi::t_real>(-(imsizex_-1))/2., static_cast<psi::t_real>(imsizex_-1)/2.);

	return ( (1. / range_v.unaryExpr(ftkernelv)).matrix() * (1. / range_u.unaryExpr(ftkernelu)).matrix().transpose() ).array();
}

Image<t_real> init_correction2d_fft(const t_real &oversample_ratio, const t_uint &imsizey_,
                                    const t_uint imsizex_, 
                                    const std::function<t_real(t_real)> kernelu,
                                    const std::function<t_real(t_real)> kernelv,
                                    const t_int Ju, const t_int Jv) {
  /*
    Given the gridding kernel, creates the scaling image for gridding correction using an fft.
  */
  // TODO: needs to be fixed given the changes above?
  const t_uint ftsizeu_ = std::floor(imsizex_ * oversample_ratio);
  const t_uint ftsizev_ = std::floor(imsizey_ * oversample_ratio);
  Matrix<t_complex> K = Matrix<t_complex>::Zero(ftsizeu_, ftsizev_);
  for(int i = 0; i < Ju; ++i) {
    t_int n = utilities::mod(i - Ju / 2, ftsizeu_);
    for(int j = 0; j < Jv; ++j) {
      t_int m = utilities::mod(j - Jv / 2, ftsizev_);
      const t_complex I(0, 1);
      K(n, m) = kernelu(i - Ju / 2) * kernelv(j - Jv / 2)
                * std::exp(-2 * constant::pi * I * ((i - Ju / 2) * 0.5 + (j - Jv / 2) * 0.5));
    }
  }
  t_int x_start = std::floor(ftsizeu_ * 0.5 - imsizex_ * 0.5);
  t_int y_start = std::floor(ftsizev_ * 0.5 - imsizey_ * 0.5);

  // iFFT section
  t_int plan_flag = (FFTW_MEASURE | FFTW_PRESERVE_INPUT);
#ifdef PURIPSI_OPENMP_FFTW
  PURIPSI_LOW_LOG("Using OpenMP threading with FFTW.");
  fftw_init_threads();
  fftw_plan_with_nthreads(omp_get_max_threads());
#endif
  Vector<t_complex> src = Vector<t_complex>::Zero(ftsizev_ * ftsizeu_);
  Vector<t_complex> dst = Vector<t_complex>::Zero(ftsizev_ * ftsizeu_);
  // creating plan
  const auto del = [](fftw_plan_s *plan) { fftw_destroy_plan(plan); };
  const std::shared_ptr<fftw_plan_s> m_plan_inverse(
      fftw_plan_dft_2d(ftsizev_, ftsizeu_, reinterpret_cast<fftw_complex *>(src.data()),
                       reinterpret_cast<fftw_complex *>(dst.data()), FFTW_BACKWARD, plan_flag),
      del);

  Matrix<t_complex> output = Matrix<t_complex>::Zero(K.rows(), K.cols());
  fftw_execute_dft(
      m_plan_inverse.get(),
      const_cast<fftw_complex *>(reinterpret_cast<const fftw_complex *>(K.data())),
      reinterpret_cast<fftw_complex *>(output.data()));

  Image<t_real> S = output.array().real().block(y_start, x_start, imsizey_, imsizex_);
  return 1 / S;
}
} // namespace details
} // namespace puripsi
