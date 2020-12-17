#ifndef PURIPSI_OPERATORS_H
#define PURIPSI_OPERATORS_H

#include "puripsi/config.h"
#include <iostream>
#include <tuple>
#include <type_traits>
#include <mpi.h>
#include "psi/chained_operators.h"
#include "psi/linear_transform.h"
#include "puripsi/kernels.h"
#include "puripsi/logging.h"
#include "puripsi/types.h"
#include "puripsi/utilities.h"
#include "puripsi/wproj_utilities.h"
#include "puripsi/projection_kernels.h"
#include "puripsi/index_mapping.h"
#ifdef PURIPSI_OPENMP_FFTW
#include <omp.h>
#endif
#include <fftw3.h>

namespace puripsi {

namespace operators {

//! enum for fftw plans
enum class fftw_plan { estimate, measure };

//! Constructs FFT operator
template <class T> std::tuple<psi::OperatorFunction<T>, psi::OperatorFunction<T>> init_FFT_2d(const t_uint imsizey_, const t_uint imsizex_, const t_real oversample_factor_,
		const fftw_plan fftw_plan_flag_ = fftw_plan::measure) {
	t_int const ftsizeu_ = std::floor(imsizex_ * oversample_factor_);
	t_int const ftsizev_ = std::floor(imsizey_ * oversample_factor_);
	t_int plan_flag = (FFTW_MEASURE | FFTW_PRESERVE_INPUT);
	switch(fftw_plan_flag_) {
	case(fftw_plan::measure):
    																			plan_flag = (FFTW_MEASURE | FFTW_PRESERVE_INPUT);
	break;
	case(fftw_plan::estimate):
    																			plan_flag = (FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
	break;
	}

#ifdef PURIPSI_OPENMP_FFTW
	PURIPSI_LOW_LOG("Using OpenMP threading with FFTW.");
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
#endif
	Vector<typename T::Scalar> src = Vector<t_complex>::Zero(ftsizev_ * ftsizeu_);
	Vector<typename T::Scalar> dst = Vector<t_complex>::Zero(ftsizev_ * ftsizeu_);
	// creating plans
	const auto del = [](fftw_plan_s *plan) { fftw_destroy_plan(plan); };
	const std::shared_ptr<fftw_plan_s> m_plan_forward(
			fftw_plan_dft_2d(ftsizev_, ftsizeu_, reinterpret_cast<fftw_complex *>(src.data()),
					reinterpret_cast<fftw_complex *>(dst.data()), FFTW_FORWARD, plan_flag),
					del);
	const std::shared_ptr<fftw_plan_s> m_plan_inverse(
			fftw_plan_dft_2d(ftsizev_, ftsizeu_, reinterpret_cast<fftw_complex *>(src.data()),
					reinterpret_cast<fftw_complex *>(dst.data()), FFTW_BACKWARD, plan_flag),
					del);
	auto const direct = [m_plan_forward, ftsizeu_, ftsizev_](T &output, const T &input) {
		assert(input.size() == ftsizev_ * ftsizeu_);
		output = Matrix<typename T::Scalar>::Zero(input.rows(), input.cols());
		fftw_execute_dft(
				m_plan_forward.get(),
				const_cast<fftw_complex *>(reinterpret_cast<const fftw_complex *>(input.data())),
				reinterpret_cast<fftw_complex *>(output.data()));
	};
	auto const indirect = [m_plan_inverse, ftsizeu_, ftsizev_](T &output, const T &input) {
		assert(input.size() == ftsizev_ * ftsizeu_);
		output = Matrix<typename T::Scalar>::Zero(input.rows(), input.cols());
		fftw_execute_dft(
				m_plan_inverse.get(),
				const_cast<fftw_complex *>(reinterpret_cast<const fftw_complex *>(input.data())),
				reinterpret_cast<fftw_complex *>(output.data()));
	};
	return std::make_tuple(direct, indirect);
}


template <class T> t_real power_method(const psi::LinearTransform<T> &op, const t_uint &niters,
		const t_real &relative_difference, const T &initial_vector) {


	if(niters <= 0)
		return 1;
	t_real estimate_eigen_value = 1;
	t_real old_value = 0;
	T estimate_eigen_vector = initial_vector;
	estimate_eigen_vector = estimate_eigen_vector / estimate_eigen_vector.matrix().norm();
	PURIPSI_DEBUG("Starting power method");
	PURIPSI_DEBUG("Iteration: 0, norm = {}", estimate_eigen_value);
	for(t_int i = 0; i < niters; ++i) {
		estimate_eigen_vector = op.adjoint() * (op * estimate_eigen_vector);
		estimate_eigen_value = estimate_eigen_vector.matrix().norm();
		PURIPSI_DEBUG("Iteration: {}, norm = {}", i + 1, estimate_eigen_value);
		if(estimate_eigen_value <= 0)
			throw std::runtime_error("Error in operator.");
		if(estimate_eigen_value != estimate_eigen_value)
			throw std::runtime_error("Error in operator or data corrupted.");
		estimate_eigen_vector = estimate_eigen_vector / estimate_eigen_value;
		if(relative_difference * relative_difference
				> std::abs(old_value - estimate_eigen_value) / old_value) {
			old_value = estimate_eigen_value;
			PURIPSI_DEBUG("Converged to norm = {}, relative difference < {}", std::sqrt(old_value),
					relative_difference);
			break;
		}
		old_value = estimate_eigen_value;
	}
	return std::sqrt(old_value);
}

template <class T> psi::OperatorFunction<T> init_normalise(const t_real &op_norm) {
	if(not(op_norm > 0))
		throw std::runtime_error("Operator norm is not greater than zero.");
	return [=](T &output, const T &x) { output = x / op_norm; };
}

//! Constructs zero-padding operator
template <class T> std::tuple<psi::OperatorFunction<T>, psi::OperatorFunction<T>> init_weights_(const Vector<t_complex> &weights) {
	PURIPSI_DEBUG("Calculating weights: W");
	auto direct = [=](T &output, const T &x) {
		assert(weights.size() == x.size());
		output = weights.array() * x.array();
	};
	auto indirect = [=](T &output, const T &x) {
		assert(weights.size() == x.size());
		output = weights.conjugate().array() * x.array();
	};
	return std::make_tuple(direct, indirect);
}


//! Construct gridding matrix
Sparse<t_complex> init_inner_gridding_matrix_2d(const Vector<t_real> &u, const Vector<t_real> &v,
		const Vector<t_complex> &weights,
		const std::function<t_real(t_real)> kernelu,
		const std::function<t_real(t_real)> kernelv);

//! Construct gridding matrix with w projection
Sparse<t_complex> init_inner_gridding_matrix_2d(const Vector<t_real> &u, const Vector<t_real> &v, const Vector<t_real> &w,
		const Vector<t_complex> &weights,
		const std::function<t_real(t_real)> kernelu,
		const std::function<t_real(t_real)> kernelv,
		const std::function<t_complex(t_real, t_real, t_real)> kernelw);

template <class T1, class T2> class MeasurementOperator : public psi::LinearTransform<T1> {

public:

	MeasurementOperator(MeasurementOperator const &m) : psi::LinearTransform<psi::Vector<t_complex>>(m),
	S_(m.S_), N_(m.N_), M_(m.M_), imsizex_(m.imsizex_), imsizey_(m.imsizey_),
	Ju_(m.Ju_), Jv_(m.Jv_), Jw_(m.Jw_), weights_(m.weights_),
	w_term_(m.w_term_), kernel_(m.kernel_), oversample_ratio_(m.oversample_ratio_),
	power_iters_(m.power_iters_), power_tol_(m.power_tol_),
	ftsizeu_(m.ftsizeu_), ftsizev_(m.ftsizev_),
	cellx_(m.cellx_), celly_(m.celly_), preconditioning_(m.preconditioning_), preconditioner_(m.preconditioner_),
	kernelu_(m.kernelu_), kernelv_(m.kernelv_), ftkernelu_(m.ftkernelu_), ftkernelv_(m.ftkernelv_),
	nshiftx_(m.nshiftx_), nshifty_(m.nshifty_), fourier_indices_(m.fourier_indices_),
	directZ_(m.directZ_), indirectZ_(m.indirectZ_),
	directFFT_(m.directFFT_), indirectFFT_(m.indirectFFT_),
	directG_(m.directG_), indirectG_(m.indirectG_){

		PURIPSI_HIGH_LOG("Copy operator called");

		directFZ_ = psi::chained_operators<T1>(directFFT(), directZ());
		indirectFZ_ = psi::chained_operators<T1>(indirectZ(), indirectFFT());

	};

	MeasurementOperator(MeasurementOperator const &&m) : psi::LinearTransform<psi::Vector<t_complex>>(m),
			S_(std::move(m.S_)), N_(std::move(m.N_)), M_(std::move(m.M_)), imsizex_(std::move(m.imsizex_)), imsizey_(std::move(m.imsizey_)),
			Ju_(std::move(m.Ju_)), Jv_(std::move(m.Jv_)), Jw_(std::move(m.Jw_)), weights_(std::move(m.weights_)),
			w_term_(std::move(m.w_term_)), kernel_(std::move(m.kernel_)), oversample_ratio_(std::move(m.oversample_ratio_)),
			power_iters_(std::move(m.power_iters_)), power_tol_(std::move(m.power_tol_)),
			ftsizeu_(std::move(m.ftsizeu_)), ftsizev_(std::move(m.ftsizev_)),
			cellx_(std::move(m.cellx_)), celly_(std::move(m.celly_)), preconditioning_(std::move(m.preconditioning_)), preconditioner_(std::move(m.preconditioner_)),
			kernelu_(std::move(m.kernelu_)), kernelv_(std::move(m.kernelv_)), ftkernelu_(std::move(m.ftkernelu_)), ftkernelv_(std::move(m.ftkernelv_)),
			nshiftx_(std::move(m.nshiftx_)), nshifty_(std::move(m.nshifty_)), fourier_indices_(std::move(m.fourier_indices_)),
			directZ_(std::move(m.directZ_)), indirectZ_(std::move(m.indirectZ_)),
			directFFT_(std::move(m.directFFT_)), indirectFFT_(std::move(m.indirectFFT_)),
			directG_(std::move(m.directG_)), indirectG_(std::move(m.indirectG_)){

		PURIPSI_HIGH_LOG("Move operator called");

		directFZ_ = psi::chained_operators<T1>(directFFT(), directZ());
		indirectFZ_ = psi::chained_operators<T1>(indirectZ(), indirectFFT());

	};

	MeasurementOperator()  : psi::LinearTransform<T1>(psi::linear_transform_identity<T2>()){}

	MeasurementOperator(const Vector<t_real> &u, const Vector<t_real> &v, const Vector<t_real> &w,
			const Vector<t_complex> &weights, const Vector<t_real>  &preconditioner,
			const t_uint &imsizey, const t_uint &imsizex, const t_real &oversample_ratio = 2,
			const t_uint &power_iters = 100, const t_real &power_tol = 1e-4,
			const kernels::kernel kernel = kernels::kernel::kb, const t_real nshifty = 0., const t_real nshiftx = 0.,
			const t_uint Ju = 4, const t_uint Jv = 4,
			const bool w_term = false, const t_real &cell_x = 1, const t_real &cell_y = 1)
	: psi::LinearTransform<psi::Vector<t_complex>>(linear_transform(u.size(), imsizey, imsizex, oversample_ratio, preconditioner)),
	  preconditioning_(true), Jw_(Jw), preconditioner_(preconditioner), imsizex_(imsizex), imsizey_(imsizey), N_({0, 1, static_cast<t_int>(imsizey * imsizex)}),
	  M_({0, 1, static_cast<t_int>(u.size())}), Ju_(Ju), Jv_(Jv), weights_(weights), w_term_(w_term), nshifty_(nshifty),
	  nshiftx_(nshiftx), kernel_(kernel), oversample_ratio_(oversample_ratio), power_iters_(power_iters), power_tol_(power_tol),
	  cellx_(cell_x), celly_(cell_y), ftsizev_(imsizey*oversample_ratio), ftsizeu_ (imsizex*oversample_ratio) {


		init_degrid_operator_2d(u, v, w, weights, imsizey, imsizex,
				oversample_ratio, power_iters, power_tol, kernel, nshifty, nshiftx, Ju, Jv,
				w_term, cell_x, cell_y);
	}

	MeasurementOperator(const utilities::vis_params &uv_vis_input,  const Vector<t_real> &preconditioner,
			const t_uint &imsizey, const t_uint &imsizex, const t_real &cell_x, const t_real &cell_y,
			const t_real &oversample_ratio = 2, const t_uint &power_iters = 100,
			const t_real &power_tol = 1e-4,
			const kernels::kernel kernel = kernels::kernel::kb, const t_real nshiftx = 0., const t_real nshifty = 0.,
			const t_uint Ju = 4, const t_uint Jv = 4,
			const bool w_term = false,  const t_uint Jw = 6)
	: psi::LinearTransform<psi::Vector<t_complex>>(linear_transform(uv_vis_input.u.size(), imsizey, imsizex, oversample_ratio, preconditioner)),
	  preconditioning_(true), Jw_(Jw), preconditioner_(preconditioner), imsizex_(imsizex), imsizey_(imsizey), N_({0, 1, static_cast<t_int>(imsizey * imsizex)}),
	  M_({0, 1, static_cast<t_int>(uv_vis_input.u.size())}), Ju_(Ju), Jv_(Jv), weights_(uv_vis_input.weights), w_term_(w_term), nshifty_(nshifty),
	  nshiftx_(nshiftx), kernel_(kernel), oversample_ratio_(oversample_ratio), power_iters_(power_iters), power_tol_(power_tol),
	  cellx_(cell_x), celly_(cell_y), ftsizev_(imsizey*oversample_ratio), ftsizeu_ (imsizex*oversample_ratio){

		auto uv_vis = uv_vis_input;
		if (uv_vis.units == utilities::vis_units::lambda)
			uv_vis = utilities::set_cell_size(uv_vis, cell_x, cell_y);
		if (uv_vis.units == utilities::vis_units::radians)
			uv_vis = utilities::uv_scale(uv_vis, std::floor(oversample_ratio * imsizex),
					std::floor(oversample_ratio * imsizey));
		init_degrid_operator_2d(uv_vis.u, uv_vis.v, uv_vis.w, uv_vis.weights, imsizey, imsizex,
				oversample_ratio, power_iters, power_tol, kernel, nshifty, nshiftx, Ju, Jv,
				w_term, cell_x, cell_y);
	}

	MeasurementOperator(const Vector<t_real> &u, const Vector<t_real> &v, const Vector<t_real> &w,
			const Vector<t_complex> &weights, const t_uint &imsizey,
			const t_uint &imsizex, const t_real &oversample_ratio = 2,
			const t_uint &power_iters = 100, const t_real &power_tol = 1e-4,
			const kernels::kernel kernel = kernels::kernel::kb, const t_real nshifty = 0., const t_real nshiftx = 0.,
			const t_uint Ju = 4, const t_uint Jv = 4,
			const bool w_term = false, const t_real &cell_x = 1, const t_real &cell_y = 1)
	: psi::LinearTransform<psi::Vector<t_complex>>(linear_transform(u.size(), imsizey, imsizex, oversample_ratio)),
	  Jw_(6), imsizex_(imsizex), imsizey_(imsizey), N_({0, 1, static_cast<t_int>(imsizey * imsizex)}),
	  M_({0, 1, static_cast<t_int>(u.size())}), Ju_(Ju), Jv_(Jv), weights_(weights), w_term_(w_term), nshifty_(nshifty),
	  nshiftx_(nshiftx), kernel_(kernel), oversample_ratio_(oversample_ratio), power_iters_(power_iters), power_tol_(power_tol),
	  cellx_(cell_x), celly_(cell_y), preconditioning_(false), ftsizev_(imsizey*oversample_ratio), ftsizeu_ (imsizex*oversample_ratio){

		init_degrid_operator_2d(u, v, w, weights, imsizey, imsizex,
				oversample_ratio, power_iters, power_tol, kernel, nshifty, nshiftx, Ju, Jv,
				w_term, cell_x, cell_y);
	}


	MeasurementOperator(const utilities::vis_params &uv_vis_input, const t_uint &imsizey,
			const t_uint &imsizex, const t_real &cell_x, const t_real &cell_y,
			const t_real &oversample_ratio = 2, const t_uint &power_iters = 100,
			const t_real &power_tol = 1e-4,
			const kernels::kernel kernel = kernels::kernel::kb, const t_real nshiftx = 0., const t_real nshifty = 0.,
			const t_uint Ju = 4, const t_uint Jv = 4,
			const bool w_term = false,  const t_uint Jw = 6)
	: psi::LinearTransform<psi::Vector<t_complex>>(linear_transform(uv_vis_input.u.size(), imsizey, imsizex, oversample_ratio)),
	  Jw_(Jw), imsizex_(imsizex), imsizey_(imsizey), N_({0, 1, static_cast<t_int>(imsizey * imsizex)}),
	  M_({0, 1, static_cast<t_int>(uv_vis_input.u.size())}), Ju_(Ju), Jv_(Jv), weights_(uv_vis_input.weights), w_term_(w_term), nshifty_(nshifty),
	  nshiftx_(nshiftx), kernel_(kernel), oversample_ratio_(oversample_ratio), power_iters_(power_iters), power_tol_(power_tol),
	  cellx_(cell_x), celly_(cell_y), preconditioning_(true), ftsizev_(imsizey*oversample_ratio), ftsizeu_ (imsizex*oversample_ratio){

		auto uv_vis = uv_vis_input;
		if (uv_vis.units == utilities::vis_units::lambda)
			uv_vis = utilities::set_cell_size(uv_vis, cell_x, cell_y);
		if (uv_vis.units == utilities::vis_units::radians)
			uv_vis = utilities::uv_scale(uv_vis, std::floor(oversample_ratio * imsizex),
					std::floor(oversample_ratio * imsizey));
		init_degrid_operator_2d(uv_vis.u, uv_vis.v, uv_vis.w, uv_vis.weights, imsizey, imsizex,
				oversample_ratio, power_iters, power_tol, kernel, nshifty, nshiftx, Ju, Jv,
				w_term, cell_x, cell_y);
	}

	virtual ~MeasurementOperator() {};

	psi::OperatorFunction<T1> directG() const { return directG_; }
	void directG(psi::OperatorFunction<T1> value) { directG_ = value; }
	psi::OperatorFunction<T1> indirectG() const { return indirectG_; }
	void indirectG(psi::OperatorFunction<T1> value) { indirectG_ = value; }
	psi::OperatorFunction<T1> directFZ() const { return directFZ_; }
	void directFZ(psi::OperatorFunction<T1> value) { directFZ_ = value; }
	psi::OperatorFunction<T1> indirectFZ() const { return indirectFZ_; }
	void indirectFZ(psi::OperatorFunction<T1> value) { indirectFZ_ = value; }
	psi::OperatorFunction<T1> directZ() const { return directZ_; }
	void directZ(psi::OperatorFunction<T1> value) { directZ_ = value; }
	psi::OperatorFunction<T1> indirectZ() const { return indirectZ_; }
	void indirectZ(psi::OperatorFunction<T1> value) { indirectZ_ = value; }
	psi::OperatorFunction<T1> directFFT() const { return directFFT_; }
	void directFFT(psi::OperatorFunction<T1> value) { directFFT_ = value; }
	psi::OperatorFunction<T1> indirectFFT() const { return indirectFFT_; }
	void indirectFFT(psi::OperatorFunction<T1> value) { indirectFFT_ = value; }
	t_uint imsizex() const { return imsizex_; }
	t_uint imsizey() const { return imsizey_; }
	void imsizex(t_uint value) { imsizex_ = value; }
	void imsizey(t_uint value) { imsizey_ = value; }
	t_uint ftsizev() const { return ftsizev_; }
	t_uint ftsizeu() const { return ftsizeu_; }
	void ftsizev(t_uint value) { ftsizev_ = value; }
	void ftsizeu(t_uint value) { ftsizeu_ = value; }
	t_uint Ju() const { return Ju_; }
	t_uint Jv() const { return Jv_; }
	t_uint Jw() const { return Jw_; }
	void Ju(t_uint value) { Ju_ = value; }
	void Jv(t_uint value) { Jv_ = value; }
	void Jw(t_uint value) { Jw_ = value; }
	bool w_term() const { return w_term_; }
	void w_term(bool value) { w_term_ = value; }
	t_real nshifty() const { return nshifty_; }
	t_real nshiftx() const { return nshiftx_; }
	void nshifty(t_real value) { nshifty_ = value; }
	void nshiftx(t_real value) { nshiftx_ = value; }
	kernels::kernel kernel() const { return kernel_; }
	void kernel(kernels::kernel value) { kernel_ = value; }
	t_real oversample_ratio() const { return oversample_ratio_; }
	void oversample_ratio(t_real value) { oversample_ratio_ = value; }
	t_uint power_iters() const { return power_iters_; }
	void power_iters(t_uint value) { power_iters_ = value; }
	t_real power_tol() const { return power_tol_; }
	void power_tol(t_real value) { power_tol_ = value; }
	t_real cellx() const { return cellx_; }
	void cellx(t_real value) { cellx_ = value; }
	t_real celly() const { return celly_; }
	void celly(t_real value) { celly_ = value; }
	bool preconditioning() const { return preconditioning_; }
	void preconditioning(bool value) { preconditioning_ = value; }
	std::array<t_int, 3> M() { return M_; }
	std::array<t_int, 3> N() { return N_; }
	Image<t_real> S(){ return S_; }

protected:

	Vector<t_int> fourier_indices_;
	psi::OperatorFunction<T1> directZ_;
	psi::OperatorFunction<T1> indirectZ_;
	psi::OperatorFunction<T1> directFFT_;
	psi::OperatorFunction<T1> indirectFFT_;
	psi::OperatorFunction<T1> directG_;
	psi::OperatorFunction<T1> indirectG_;
	psi::OperatorFunction<T1> directFZ_;
	psi::OperatorFunction<T1> indirectFZ_;

	std::function<t_real(t_real)> kernelu_;
	std::function<t_real(t_real)> kernelv_;
	std::function<t_real(t_real)> ftkernelu_;
	std::function<t_real(t_real)> ftkernelv_;

	Image<t_real> S_;
	std::array<t_int, 3> N_;
	std::array<t_int, 3> M_;
	t_uint imsizex_;
	t_uint imsizey_;
	t_uint ftsizev_;
	t_uint ftsizeu_;
	t_uint Ju_;
	t_uint Jv_;
	t_uint Jw_;
	Vector<t_complex> weights_;
	bool w_term_;
	t_real nshifty_;
	t_real nshiftx_;
	kernels::kernel kernel_;
	t_real oversample_ratio_;
	t_uint power_iters_;
	t_real power_tol_;
	t_real cellx_;
	t_real celly_;
	bool preconditioning_ = false;
	Vector<t_real> preconditioner_;

public:

	//! Construct gridding matrix with mixing
	template <class... ARGS> Sparse<t_complex> init_gridding_matrix_2d(const Sparse<T1> &mixing_matrix, ARGS &&... args) {

		if(mixing_matrix.rows() * mixing_matrix.cols() < 2)
			return init_gridding_matrix_2d(std::forward<ARGS>(args)...);
		const Sparse<t_complex> G = init_gridding_matrix_2d(std::forward<ARGS>(args)...);
		if(mixing_matrix.cols() != G.rows())
			throw std::runtime_error(
					"The columns of the mixing matrix do not match the number of visibilities");
		return mixing_matrix * init_gridding_matrix_2d(std::forward<ARGS>(args)...);
	}

	std::tuple<psi::OperatorFunction<T1>, psi::OperatorFunction<T1>> init_gridding_matrix_2d(const Vector<t_real> &u, const Vector<t_real> &v, const Vector<t_real> &w,
			const Vector<t_complex> &weights,
			const std::function<t_real(t_real)> kernelu,
			const std::function<t_real(t_real)> kernelv,
			const std::function<t_complex(t_real, t_real, t_real)> kernelw){

		const std::shared_ptr<const Sparse<t_complex>> interpolation_matrix = std::make_shared<const Sparse<t_complex>>(init_inner_gridding_matrix_2d(u, v, w, weights, kernelu, kernelv, kernelw));

		const std::shared_ptr<Sparse<t_complex>> adjoint =
				std::make_shared<Sparse<t_complex>>(interpolation_matrix->adjoint());

		(*adjoint).makeCompressed();

		return std::make_tuple(
				[=](T1 &output, const T1 &input) {
			output = utilities::sparse_multiply_matrix(*interpolation_matrix, input);
			return output;
		},
		[=](T1 &output, const T1 &input) {
			output = utilities::sparse_multiply_matrix(*adjoint, input);
			return output;
		});
	}

	//! Constructs zero-padding operator
	std::tuple<psi::OperatorFunction<T1>, psi::OperatorFunction<T1>> init_zero_padding_2d() {
		const t_int sizex_ = S_.cols();
		const t_int sizey_ = S_.rows();
		const t_int sizeu_ = std::floor(sizex_ * oversample_ratio_);
		const t_int sizev_ = std::floor(sizey_ * oversample_ratio_);
		const t_int x_start = 0;
		const t_int y_start = 0;
		const t_int imsizey = imsizey_;
		const t_int imsizex = imsizex_;
		const Image<t_real> localS = S_;

		auto direct = [=](T1 &output, const T1 &x) {
			assert(sizey_ >= 0);
			assert(sizex_ >= 0);
			assert(x.size() == sizex_ * sizey_);
			output = T1::Zero(sizeu_ * sizev_);
			//! check whether this is correct
#ifdef PURIPSI_OPENMP
#pragma omp parallel for collapse(2) default(shared)
#endif
			for(t_int j = 0; j < sizey_; j++) {
				for(t_int i = 0; i < sizex_; i++) {
					const t_int input_index = utilities::sub2ind(j, i, sizey_, sizex_);
					const t_int output_index = utilities::sub2ind(y_start + j, x_start + i, sizev_, sizeu_);
					output(output_index) = localS(j, i) * x(input_index);
				}
			}
		};
		auto indirect = [=](T1 &output, const T1 &x) {
			assert(sizey_ >= 0);
			assert(sizex_ >= 0);
			assert(x.size() == sizeu_ * sizev_);
			output = T1::Zero(sizey_ * sizex_);
#ifdef PURIPSI_OPENMP
#pragma omp parallel for collapse(2) default(shared)
#endif
			for(t_int j = 0; j < sizey_; j++) {
				for(t_int i = 0; i < sizex_; i++) {
					const t_int output_index = utilities::sub2ind(j, i, sizey_, sizex_);
					const t_int input_index = utilities::sub2ind(y_start + j, x_start + i, sizev_, sizeu_);
					output(output_index) = std::conj(localS(j, i)) * x(input_index);
				}
			}
		};
		return std::make_tuple(direct, indirect);
	}


	void base_padding_and_FFT_2d(const std::function<t_real(t_real)> &kernelu,
			const std::function<t_real(t_real)> &kernelv,
			const fftw_plan &ft_plan = fftw_plan::measure) {

		S_ = init_correction2d(oversample_ratio(), imsizey(), imsizex(), kernelu, kernelv);

		PURIPSI_LOW_LOG("Norm of S: {}", S_.matrix().norm());
		PURIPSI_LOW_LOG("Building Measurement Operator: WGFZDB");
		PURIPSI_LOW_LOG("Constructing Zero Padding and Correction Operator: ZDB");
		PURIPSI_MEDIUM_LOG("Image size (width, height): {} x {}", imsizex_, imsizey_);
		PURIPSI_MEDIUM_LOG("Oversampling Factor: {}", oversample_ratio_);
		std::tie(directZ_, indirectZ_) = init_zero_padding_2d();
		PURIPSI_LOW_LOG("Constructing FFT operator: F");
		switch(ft_plan) {
		case fftw_plan::measure:
			PURIPSI_MEDIUM_LOG("Measuring Plans...");
			break;
		case fftw_plan::estimate:
			PURIPSI_MEDIUM_LOG("Estimating Plans...");
			break;
		}
		std::tie(directFFT_, indirectFFT_) = puripsi::operators::init_FFT_2d<T1>(imsizey_, imsizex_, oversample_ratio_, ft_plan);
		directFZ_ = psi::chained_operators<T1>(directFFT(), directZ());
		indirectFZ_ = psi::chained_operators<T1>(indirectZ(), indirectFFT());

	}

	void base_degrid_operator_2d(const Vector<t_real> &u, const Vector<t_real> &v, const Vector<t_real> &w,
			const Vector<t_complex> &weights, const fftw_plan &ft_plan = fftw_plan::measure){
		std::tie(kernelu_, kernelv_, ftkernelu_, ftkernelv_)
		= puripsi::create_kernels(kernel_, Ju_, Jv_, imsizey_, imsizex_, oversample_ratio_);
		base_padding_and_FFT_2d(ftkernelu_, ftkernelv_, ft_plan);
		PURIPSI_LOW_LOG("directFZ_ and indirectFZ_ created");
		if(w_term_ == true)
			PURIPSI_MEDIUM_LOG("FoV (width, height): {} deg x {} deg", imsizex_ * cellx_ / (60. * 60.), imsizey_ * celly_ / (60. * 60.));
		PURIPSI_LOW_LOG("Constructing Weighting and Gridding Operators: WG");
		PURIPSI_MEDIUM_LOG("Number of visibilities: {}", u.size());
		std::function<t_complex(t_real, t_real, t_real)> kernelw =
				projection_kernels::w_projection_kernel_approx(cellx_, celly_, imsizex_, imsizey_, oversample_ratio_);
		std::tie(directG_, indirectG_) = init_gridding_matrix_2d(
				u, v, w.array() - w.array().mean(), weights, kernelv_,
				kernelu_, kernelw);

		PURIPSI_LOW_LOG("directG_ and indirectG_ created");
		PURIPSI_MEDIUM_LOG("Finished construction of Φ.");
		return;
	}

	//! Returns linear transform that is the standard degridding operator
	void init_degrid_operator_2d(const Vector<t_real> &u, const Vector<t_real> &v, const Vector<t_real> &w,
			const Vector<t_complex> &weights, const t_uint &imsizey,
			const t_uint &imsizex, const t_real &oversample_ratio = 2,
			const t_uint &power_iters = 100, const t_real &power_tol = 1e-4,
			const kernels::kernel kernel = kernels::kernel::kb, const t_real nshifty = 0., const t_real nshiftx = 0.,
			const t_uint Ju = 4, const t_uint Jv = 4,
			const bool w_term = false, const t_real &cellx = 1, const t_real &celly = 1) {
		const operators::fftw_plan ft_plan = fftw_plan::measure;
		base_degrid_operator_2d(u, v, w, weights_, ft_plan);
		return;
	}

	template <class... ARGS> std::shared_ptr<psi::LinearTransform<T1> const> init_combine_operators(const std::vector<std::shared_ptr<psi::LinearTransform<T1>>> &measure_op,
			const std::vector<std::tuple<t_uint, t_uint>> seg) {
		const auto direct = [=](T1 &output, const T1 &input) {
			for(t_uint i = 0; i < measure_op.size(); i++)
				output.segment(std::get<0>(seg.at(i)), std::get<1>(seg.at(i))) = *(measure_op.at(i)) * input;
		};
		const auto indirect = [=](T1 &output, const T1 &input) {
			for(t_uint i = 0; i < measure_op.size(); i++)
				output += measure_op.at(i)->adjoint()
				* input.segment(std::get<0>(seg.at(i)), std::get<1>(seg.at(i)));
		};
		return std::make_shared<psi::LinearTransform<T1> const>(direct, indirect);
	}

	psi::LinearTransform<T1> linear_transform(t_uint nvis) {
		return linear_transform(nvis, imsizey(), imsizex(), oversample_ratio());
	}

	psi::LinearTransform<T1> linear_transform(t_uint nvis, t_int imsizey, t_int imsizex, t_int oversample_ratio) {
		auto const height = imsizey;
		auto const width = imsizex;
		auto direct = [this, height, width](Vector<t_complex> &out, Vector<t_complex> const &x) {
			assert(x.size() == width * height);
			auto const image = Image<t_complex>::Map(x.data(), height, width);
			out = this->degrid(image);
			return;
		};
		auto adjoint = [this, height, width](Vector<t_complex> &out, Vector<t_complex> const &x) {
			assert(out.size() == width * height);
			auto image = Image<t_complex>::Map(out.data(), height, width);
			image = this->grid(x);
			return;
		};
		return psi::linear_transform<Vector<t_complex>>(direct, {{0, 1, static_cast<t_int>(nvis)}},
				adjoint,
				{{0, 1, static_cast<t_int>(width * height)}},
				imsizey, imsizex, oversample_ratio);
	}


	psi::LinearTransform<T1> linear_transform(t_uint nvis, Vector<t_real> const &preconditioner) {
		return linear_transform(nvis, imsizey(), imsizex(), oversample_ratio(), preconditioner);
	}

	psi::LinearTransform<T1> linear_transform(t_uint nvis, t_int imsizey, t_int imsizex, t_int oversample_ratio, Vector<t_real> const &preconditioner) {
		auto const height = imsizey;
		auto const width = imsizex;
		auto direct = [this, width, height, &preconditioner](Vector<t_complex> &out, Vector<t_complex> const &x) {
			assert(x.size() == width * height);
			auto const image = Image<t_complex>::Map(x.data(), height, width);
			out = preconditioned_degrid(image, preconditioner);
			return;
		};
		auto adjoint
		= [this, width, height, &preconditioner](Vector<t_complex> &out, Vector<t_complex> const &x) {
			auto image = Image<t_complex>::Map(out.data(), height, width);
			image = preconditioned_grid(x, preconditioner);
			return;
		};
		std::shared_ptr<psi::LinearTransform<T1> const> solver = std::make_shared<psi::LinearTransform<T1> const>(psi::linear_transform<Vector<t_complex>>(direct, {{0, 1, static_cast<t_int>(nvis)}},
				adjoint,{{0, 1, static_cast<t_int>(width * height)}},
				imsizey, imsizex, oversample_ratio));
		return *solver;

	}

	psi::LinearTransform<T1> linear_transform(std::shared_ptr<MeasurementOperator const> const &measurements, t_uint nvis) {
		auto const height = measurements->imsizey();
		auto const width = measurements->imsizex();
		auto direct = [measurements, width, height](Vector<t_complex> &out, Vector<t_complex> const &x) {
			assert(x.size() == width * height);
			auto const image = Image<t_complex>::Map(x.data(), height, width);
			out = measurements->degrid(image);
			return;
		};
		auto adjoint
		= [measurements, width, height](Vector<t_complex> &out, Vector<t_complex> const &x) {
			auto image = Image<t_complex>::Map(out.data(), height, width);
			image = measurements->grid(x);
			return;
		};
		return psi::linear_transform<Vector<t_complex>>(direct, {{0, 1, static_cast<t_int>(nvis)}},
				adjoint,
				{{0, 1, static_cast<t_int>(width * height)}},
				measurements->imsizey(),
				measurements->imsizex(),
				measurements->oversample_factor());
	}

	psi::LinearTransform<T1> linear_transform(std::shared_ptr<MeasurementOperator const> const &measurements, t_uint nvis, Vector<t_real> const &preconditioner) {
		auto const height = measurements->imsizey();
		auto const width = measurements->imsizex();
		auto direct = [measurements, width, height, &preconditioner](Vector<t_complex> &out, Vector<t_complex> const &x) {
			assert(x.size() == width * height);
			auto const image = Image<t_complex>::Map(x.data(), height, width);
			out = measurements->preconditioned_degrid(image, preconditioner);
			return;
		};
		auto adjoint
		= [measurements, width, height, &preconditioner](Vector<t_complex> &out, Vector<t_complex> const &x) {
			auto image = Image<t_complex>::Map(out.data(), height, width);
			image = measurements->preconditioned_grid(x, preconditioner);
			return;
		};
		return psi::linear_transform<Vector<t_complex>>(direct, {{0, 1, static_cast<t_int>(nvis)}},
				adjoint,
				{{0, 1, static_cast<t_int>(width * height)}});
	}


	//! Construct gridding matrix
	Sparse<t_complex> init_inner_gridding_matrix_2d(const Vector<t_real> &u, const Vector<t_real> &v,
			const Vector<t_complex> &weights,
			const std::function<t_real(t_real)> kernelu,
			const std::function<t_real(t_real)> kernelv) {
		const t_uint ftsizev_ = std::floor(imsizey_ * oversample_ratio_);
		const t_uint ftsizeu_ = std::floor(imsizex_ * oversample_ratio_);
		const t_uint rows = u.size();
		const t_uint cols = ftsizeu_ * ftsizev_;
		auto omega_to_k = [](const Vector<t_real> &omega) {
			return omega.unaryExpr(std::ptr_fun<double, double>(std::floor));
		};
		const Vector<t_real> k_u = omega_to_k(u - Vector<t_real>::Constant(rows, Ju_ * 0.5));
		const Vector<t_real> k_v = omega_to_k(v - Vector<t_real>::Constant(rows, Jv_ * 0.5));
		if (u.size() != v.size())
			throw std::runtime_error("Size of u and v vectors are not the same for creating gridding matrix.");

		Sparse<t_complex> interpolation_matrix(rows, cols);
		interpolation_matrix.reserve(Vector<t_int>::Constant(rows, Ju_ * Jv_));
		Matrix<t_int> full_fourier_indices = Matrix<t_int>::Zero(rows, Ju_ * Jv_); // matrices of indices to be kept (quite huge here, trimmed-down later on)

		const t_complex I(0, 1);

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

			for(t_int i = 1; i <= Ju_; ++i) {
				const t_int q = utilities::mod(k_u(m) + i, ftsizeu_);

				psi::t_real cu = kernelu(u(m) - (k_u(m) + i));
				psi::t_complex phaseu = constant::pi * I * static_cast<t_real>(imsizex_-1)/static_cast<t_real>(ftsizeu_) * (u(m) - (k_u(m) + i));
				psi::t_complex coeffu = cu*std::exp(phaseu);

				for(t_int j = 1; j <= Jv_; ++j) {
					const t_int p = utilities::mod(k_v(m) + j, ftsizev_);
					const t_int index = utilities::sub2ind(p, q, ftsizev_, ftsizeu_);

					psi::t_real cv = kernelv(v(m) - (k_v(m) + j));
					psi::t_complex phasev = constant::pi * I * static_cast<t_real>(imsizey_-1)/static_cast<t_real>(ftsizev_) * (v(m) - (k_v(m) + j));
					psi::t_complex coeffv = cv*std::exp(phasev);
					interpolation_matrix.coeffRef(m, index) = std::conj(coeffu * coeffv)*std::exp(I*(phase_shifty + phase_shiftx))*weights(m);

					const t_int index_local = utilities::sub2ind(j-1, i-1, Jv_, Ju_);
					full_fourier_indices(m, index_local) = index;
				}
			}
		}

		// unique-sort full_fourier_indices (remove duplicate entries, keep only the index of the elements to be broadcasted to the workers)
		std::vector<t_int> index_vec(full_fourier_indices.data(), full_fourier_indices.data()+full_fourier_indices.size());
		std::sort(index_vec.begin(), index_vec.end());
		index_vec.erase(std::unique(index_vec.begin(), index_vec.end()), index_vec.end());
		fourier_indices_ = Vector<t_int>::Map(&index_vec[0], index_vec.size());
		interpolation_matrix.makeCompressed();
		return interpolation_matrix;
	}

	//! Construct gridding matrix with w projection
	// TODO: to be revised with Arwa (check this is properly fixed...)
	Sparse<t_complex> init_inner_gridding_matrix_2d(const Vector<t_real> &u, const Vector<t_real> &v, const Vector<t_real> &w,
			const Vector<t_complex> &weights,
			const std::function<t_real(t_real)> kernelu,
			const std::function<t_real(t_real)> kernelv,
			const std::function<t_complex(t_real, t_real, t_real)> kernelw){
		if (!w_term_)
			return init_inner_gridding_matrix_2d(u, v, weights, kernelu, kernelv);
		const t_uint ftsizev_ = std::floor(imsizey_ * oversample_ratio_);
		const t_uint ftsizeu_ = std::floor(imsizex_ * oversample_ratio_);
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
		const t_int Jwu = Jw_ + Ju_ - 1;
		const t_int Jwv = Jw_ + Jv_ - 1;
		interpolation_matrix.reserve(Vector<t_int>::Constant(rows, Jwv * Jwu));
		Matrix<t_int> full_fourier_indices = Matrix<t_int>::Zero(rows, Jwv * Jwu); // matrices of indices to be kept (quite huge here, trimmed-down later on)

		const t_complex I(0, 1);
#ifdef PURIPSI_OPENMP
#pragma omp parallel for default(shared)
#endif
		for (t_int m = 0; m < rows; ++m) {
			// w_projection convolution setup
			const Matrix<t_complex> projection_kernel =
					projection_kernels::projection(kernelu, kernelv, kernelw, u(m), v(m), w(m), Ju_, Jv_, Jw_);

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

					const t_int index_local = utilities::sub2ind(jv-1, ju-1, Jwv, Jwu);
					full_fourier_indices(m, index_local) = index;
				}
			}
		}

		// unique-sort full_fourier_indices (remove duplicate entries, keep only the index of the elements to be broadcasted to the workers)
		std::vector<t_int> index_vec(full_fourier_indices.data(), full_fourier_indices.data()+full_fourier_indices.size());
		std::sort(index_vec.begin(), index_vec.end());
		index_vec.erase(std::unique(index_vec.begin(), index_vec.end()), index_vec.end());
		fourier_indices_ = Vector<t_int>::Map(&index_vec[0], index_vec.size());

		return interpolation_matrix;
	}


	Vector<t_int> get_fourier_indices() const {

		return fourier_indices_;

	}

	T1 G_function(const Matrix<T2> &ft_vector) const{

		T1 result(ft_vector.size());
		directG()(result, T1::Map(ft_vector.data(), ft_vector.size()));
		return result;

	}

	T1 G_function(const Eigen::SparseMatrix<T2> &ft_vector) const{

		T1 result(ft_vector.size());
		directG()(result, ft_vector);
		return result;

	}


	T1 G_function_adjoint(const T1 &visibilities) const{

		T1 result(visibilities.size());
		indirectG()(result, visibilities);
		return result;

	}


	Matrix<T2> FFT(const Image<T2> &eigen_image) const{

		Vector<T2> result(ftsizev_*ftsizeu_);
		Vector<T2> input = Vector<T2>::Map(eigen_image.data(), eigen_image.size());
		directFZ()(result, input);
		return Matrix<t_complex>::Map(result.data(), ftsizev_ * ftsizeu_, 1);
	}

	Image<T2> inverse_FFT(Matrix<T2> &ft_vector) const{

		Vector<T2> result(ft_vector.size());
		indirectFZ()(result, T1::Map(ft_vector.data(), ft_vector.size()));
		return Image<t_complex>::Map(result.data(), imsizey_, imsizex_);

	}

	T1 degrid(const Image<T2> &eigen_image) const {

		Matrix<T2> ft_vector(imsizey_, imsizex_);
		ft_vector = FFT(eigen_image);
		Vector<T2> result(imsizey_*imsizex_);
		result = G_function(ft_vector);
		return result;

	}

	T1 preconditioned_degrid(const Image<T2> &eigen_image, const Vector<t_real> &preconditioner) const {

		Matrix<T2> ft_vector(imsizey_, imsizex_);
		ft_vector = FFT(eigen_image);
		Vector<T2> result(imsizey_*imsizex_);
		result = G_function(ft_vector);
		// Enable the preconditioning to be configured by setting the boolean preconditioning variable
		// to true or false in the object. This lets users create one operator that can do preconditioning
		// or not as required.
		if(preconditioning_){
			result = ((preconditioner.array().sqrt().cast<T2>()) * result.array()).eval();
		}
		return result;

	}

	Image<T2> grid(const T1 &visibilities) const {

		Matrix<T2> data(imsizey_, imsizex_);
		data = G_function_adjoint(visibilities);
		Image<T2> result(imsizey_, imsizex_);
		result = inverse_FFT(data);
		return result;

	}

	Image<T2> preconditioned_grid(const T1 &visibilities, const Vector<t_real> &preconditioner) const {

		T1 data(imsizey_*imsizex_);
		// Enable the preconditioning to be configured by setting the boolean preconditioning variable
		// to true or false in the object. This lets users create one operator that can do preconditioning
		// or not as required.
		if(preconditioning_){
			data = ((preconditioner.array().sqrt().cast<T2>()) * visibilities.array()).eval();
		}else{
			data = visibilities.array();
		}
		Matrix<T2> data2(imsizey_, imsizex_);
		data2 = G_function_adjoint(data);
		Image<T2> result(imsizey_, imsizex_);
		result = inverse_FFT(data2);
		return result;

	}

	void disable_preconditioning()  {

		preconditioning(false);

	}

	void enable_preconditioning()  {

		if(preconditioner_.size() != imsizey_*imsizex_){
			PURIPSI_HIGH_LOG("Attempt to enable preconditioning but the preconditioning matrix is not the correct size. Preconditioning not enabled.");
			return;
		}
		preconditioning(true);
		return;
	}

	Image<t_real> init_correction2d(const t_real oversample_ratio, const t_uint imsizey_,
			const t_uint imsizex_, const std::function<t_real(t_real)> ftkernelu,
			const std::function<t_real(t_real)> ftkernelv) {

		psi::Array<psi::t_real> range_v;
		range_v.setLinSpaced(imsizey_, (-(static_cast<psi::t_real>(imsizey_)-1)/2.), (static_cast<psi::t_real>(imsizey_)-1)/2.);

		psi::Array<psi::t_real> range_u;
		range_u.setLinSpaced(imsizex_, (-(static_cast<psi::t_real>(imsizex_)-1)/2.), (static_cast<psi::t_real>(imsizex_)-1)/2.);

		return ( (1. / range_v.unaryExpr(ftkernelv)).matrix() * (1. / range_u.unaryExpr(ftkernelu)).matrix().transpose() ).array();

	}


}; // MeasurementOperator class

} // namespace operators

}; // namespace puripsi
#endif
