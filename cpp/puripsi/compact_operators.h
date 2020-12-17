#ifndef PURIPSI_COMPACT_OPERATORS_H
#define PURIPSI_COMPACT_OPERATORS_H

#include "puripsi/config.h"
#include <iostream>
#include <tuple>
#include "psi/chained_operators.h"
#include "psi/linear_transform.h"
#include "puripsi/kernels.h"
#include "puripsi/logging.h"
#include "puripsi/operators.h"
#include "puripsi/types.h"

namespace puripsi {

namespace operators {

//! Constructs a combined gridding and degridding operator
template <class T>
psi::OperatorFunction<T> init_gridding_degridding_matrix_2d(puripsi::operators::MeasurementOperator<Vector<t_complex>, t_complex> &measure,
		const Vector<t_real> &u, const Vector<t_real> &v, const Vector<t_real> &w,
		const Vector<t_complex> &weights, const t_uint &imsizey_, const t_uint &imsizex_,
		const t_uint &oversample_ratio, const std::function<t_real(t_real)> kernelu,
		const std::function<t_real(t_real)> kernelv,
		const std::function<t_complex(t_real, t_real, t_real)> kernelw,
		const t_uint Ju = 4, const t_uint Jv = 4, const t_uint Jw = 6,
		const bool w_term = false, const t_real &cellx = 1, const t_real &celly = 1){

	const Sparse<t_complex> interpolation_matrix =
			measure.init_inner_gridding_matrix_2d(u, v, w, weights, kernelu, kernelv, kernelw);
	PURIPSI_LOW_LOG("G non zeros: {}", interpolation_matrix.nonZeros());
	Sparse<t_complex> GTG = interpolation_matrix.adjoint() * interpolation_matrix;
	GTG.prune([&](const t_uint &i, const t_uint &j, const t_complex &value) {
		return std::abs(value) > 1e-7;
	});
	PURIPSI_LOW_LOG("GTG non zeros: {}", GTG.nonZeros());
	return [=](T &output, const T &input) { output = utilities::sparse_multiply_matrix(GTG, input); };
}

template <class T>
psi::OperatorFunction<T>
base_grid_degrid_operator_2d(const Vector<t_real> &u, const Vector<t_real> &v,
		const Vector<t_real> &w, const Vector<t_complex> &weights,
		const t_uint &imsizey, const t_uint &imsizex,
		const t_real &oversample_ratio = 2,
		const kernels::kernel kernel = kernels::kernel::kb,
		const t_uint Ju = 4, const t_uint Jv = 4,
		const operators::fftw_plan ft_plan = operators::fftw_plan::measure,
		const bool w_term = false,
		const t_real &cellx = 1, const t_real &celly = 1,
		const t_real &energy_chirp_fraction = 1,
		const t_real &energy_kernel_fraction = 1) {

	auto measure = puripsi::operators::MeasurementOperator<Vector<t_complex>, t_complex>(
			u, v, w, weights, imsizey, imsizex, oversample_ratio, kernel, Ju, Jv, ft_plan, w_term, cellx,
			celly, energy_chirp_fraction, energy_kernel_fraction);
	std::function<t_real(t_real)> kernelu, kernelv, ftkernelu, ftkernelv;
	std::tie(kernelu, kernelv, ftkernelu, ftkernelv)
	= puripsi::create_kernels(kernel, Ju, Jv, imsizey, imsizex, oversample_ratio);
	psi::OperatorFunction<T> directFZ, indirectFZ;
	std::tie(directFZ, indirectFZ) = measure.base_padding_and_FFT_2d(ftkernelu, ftkernelv, ft_plan);
	psi::OperatorFunction<T> GTG;
	PURIPSI_MEDIUM_LOG("FoV (width, height): {} deg x {} deg", imsizex * cellx / (60. * 60.),
			imsizey * celly / (60. * 60.));
	PURIPSI_LOW_LOG("Constructing Weighting and Gridding Operators: WG");
	PURIPSI_MEDIUM_LOG("Number of visibilities: {}", u.size());
	PURIPSI_MEDIUM_LOG("Kernel Support: {} x {}", Ju, Jv);
	GTG = puripsi::operators::init_gridding_degridding_matrix_2d<T>(
			measure, u, v, w, weights, imsizey, imsizex, oversample_ratio,
			kernelv, kernelu, Ju, Jv, w_term, cellx,
			celly, energy_chirp_fraction, energy_kernel_fraction);
	return psi::chained_operators<T>(indirectFZ, GTG, directFZ);
}

template <class T>
psi::OperatorFunction<T>
init_grid_degrid_operator_2d(puripsi::operators::MeasurementOperator<Vector<t_complex>, t_complex> &measure,
		const Vector<t_real> &u, const Vector<t_real> &v,
		const Vector<t_real> &w, const Vector<t_complex> &weights,
		const t_uint &imsizey, const t_uint &imsizex,
		const t_real &oversample_ratio = 2, const t_uint &power_iters = 100,
		const t_real &power_tol = 1e-4, const std::string &kernel = "kb",
		const t_uint Ju = 4, const t_uint Jv = 4,
		const operators::fftw_plan ft_plan = operators::fftw_plan::measure,
		const bool w_term = false,
		const t_real &cellx = 1, const t_real &celly = 1,
		const t_real &energy_chirp_fraction = 1,
		const t_real &energy_kernel_fraction = 1) {

	/*
	 *  Returns operator that degrids and grids
	 */
	std::array<t_int, 3> N = {0, 1, static_cast<t_int>(imsizey * imsizex)};
	const psi::OperatorFunction<T> phiTphi = base_grid_degrid_operator_2d<T>(measure,
			u, v, w, weights, imsizey, imsizex, oversample_ratio, kernel, Ju, Jv, ft_plan, w_term, cellx,
			celly, energy_chirp_fraction, energy_kernel_fraction);
	auto direct = phiTphi;
	auto id = [](T &out, const T &in) { out = in; };
	const t_real op_norm =  puripsi::operators::power_method<T>({direct, N, id, N}, power_iters, power_tol,
			T::Random(imsizex * imsizey));
	const auto operator_norm = puripsi::operators::init_normalise<T>(op_norm * op_norm);
	direct = psi::chained_operators<T>(direct, operator_norm);
	return direct;
}
template <class T>
psi::OperatorFunction<T>
init_grid_degrid_operator_2d(puripsi::operators::MeasurementOperator<Vector<t_complex>, t_complex> &measure,
		const utilities::vis_params &uv_vis_input, const t_uint &imsizey,
		const t_uint &imsizex, const t_real &cell_x, const t_real &cell_y,
		const t_real &oversample_ratio = 2, const t_uint &power_iters = 100,
		const t_real &power_tol = 1e-4, const std::string &kernel = "kb",
		const t_uint Ju = 4, const t_uint Jv = 4,
		const operators::fftw_plan ft_plan = operators::fftw_plan::measure,
		const bool w_term = false,
		const t_real &energy_chirp_fraction = 1,
		const t_real &energy_kernel_fraction = 1) {

	auto uv_vis = uv_vis_input;
	if(uv_vis.units == utilities::vis_units::lambda)
		uv_vis = utilities::set_cell_size(uv_vis, cell_x, cell_y);
	if(uv_vis.units == utilities::vis_units::radians)
		uv_vis = utilities::uv_scale(uv_vis, std::floor(oversample_ratio * imsizex),
				std::floor(oversample_ratio * imsizey));
	return init_grid_degrid_operator_2d<T>(measure, uv_vis.u, uv_vis.v, uv_vis.w, uv_vis.weights, imsizey,
			imsizex, oversample_ratio, power_iters, power_tol, kernel,
			Ju, Jv, ft_plan, w_term, cell_x, cell_y,
			energy_chirp_fraction, energy_kernel_fraction);
}
template <class T>
psi::OperatorFunction<T>
init_psf_convolve_2d(const std::shared_ptr<psi::LinearTransform<T> const> &degrid_grid,
		const t_uint &imsizey, const t_uint &imsizex, const t_uint &power_iters = 100,
		const t_real &power_tol = 1e-4) {
	std::array<t_int, 3> N = {0, 1, static_cast<t_int>(imsizey * imsizex)};
	t_uint const index = utilities::sub2ind(std::floor(imsizey * 0.5) - 1,
			std::floor(imsizex * 0.5) - 1, imsizey, imsizex);
	std::cout << index << std::endl;
	T delta = T::Zero(imsizey * imsizex);
	delta(index) = 1.;
	const T psf = degrid_grid->adjoint() * (*degrid_grid * delta);
	psi::OperatorFunction<T> fftop, ifftop;
	std::tie(fftop, ifftop) = operators::init_FFT_2d<T>(imsizey, imsizex, 1, operators::fftw_plan::estimate);
	T ft_psf = T::Zero(imsizey * imsizex);
	fftop(ft_psf, psf);
	assert(ft_psf.size() == psf.size());
	const auto ft_psf_multiply = [=](T &out, const T &input) {
		out = input;
#ifdef PURIPSI_OPENMP
#pragma omp parallel for default(shared)
#endif
for(t_uint i = 0; i < input.size(); i++)
	out(i) = ft_psf(i) * input(i);
	};
	const auto psf_convolve = psi::chained_operators<T>(ifftop, ft_psf_multiply, fftop);
	const auto id = [](T &out, const T &in) { out = in; };
	const t_real op_norm = puripsi::operators::power_method<T>({psf_convolve, N, id, N}, power_iters, power_tol,
			T::Random(imsizex * imsizey));
	const auto operator_norm = puripsi::operators::init_normalise<T>(op_norm * op_norm);
	return psi::chained_operators<T>(psf_convolve, operator_norm);
}
template <class T>
psi::OperatorFunction<T>
init_psf_convolve_2d(const utilities::vis_params &uv_vis_input, const t_uint &imsizey,
		const t_uint &imsizex, const t_real &cell_x, const t_real &cell_y,
		const t_real &oversample_ratio = 2, const t_uint &power_iters = 100,
		const t_real &power_tol = 1e-4, const std::string &kernel = "kb",
		const t_uint Ju = 4, const t_uint Jv = 4,
		const operators::fftw_plan ft_plan = operators::fftw_plan::measure) {
	/*
	 *  Returns operator convolve image with direction independent point spread function
	 */
	const auto meas_op = puripsi::operators::MeasurementOperator<T, t_complex>(
			uv_vis_input, imsizey, imsizex, cell_x, cell_y, oversample_ratio, power_iters, power_tol,
			kernel, Ju, Jv, ft_plan);
	return init_psf_convolve_2d<T>(meas_op, imsizey, imsizex, power_iters, power_tol);
}
} // namespace operators

} // namespace puripsi
#endif
