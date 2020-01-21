#ifndef PURIPSI_MEASUREMENT_OPERATOR_H
#define PURIPSI_MEASUREMENT_OPERATOR_H

#include "puripsi/config.h"
#include <memory>
#include <string>
#include <psi/linear_transform.h>
#include "puripsi/FFTOperator.h"
#include "puripsi/kernels.h"
#include "puripsi/types.h"
#include "puripsi/utilities.h"

namespace puripsi {

//! This does something
class MeasurementOperator : public psi::LinearTransform<psi::Vector<t_complex>> {
public:
  Sparse<t_complex> G;
  Image<t_real> S;
  Array<t_complex> W;
  Image<t_complex> C;
  t_real norm = 1;
  t_real resample_factor = 1;

  MeasurementOperator() : psi::LinearTransform<psi::Vector<t_complex>>(psi::linear_transform_identity<t_complex>()) {};

  MeasurementOperator(MeasurementOperator const &m) : psi::LinearTransform<psi::Vector<t_complex>>(m),
		  Ju_(m.Ju_), Jv_(m.Jv_), kernel_name_(m.kernel_name_), norm_iterations_(m.norm_iterations_), cell_x_(m.cell_x_),
	      cell_y_(m.cell_y_), weighting_type_(m.weighting_type_), R_(m.R_), use_w_term_(m.use_w_term_),
	      energy_fraction_(m.energy_fraction_), fft_grid_correction_(m.fft_grid_correction_),
	      primary_beam_(m.primary_beam_), fftoperator_(m.fftoperator_), ftsizeu_(m.ftsizeu_), ftsizev_(m.ftsizev_),
		  G(m.G), S(m.S), W(m.W), C(m.C), norm(m.norm), resample_factor(m.resample_factor),
		  fftw_plan_flag_(m.fftw_plan_flag_), gradient_(m.gradient_) {};

  MeasurementOperator(const utilities::vis_params &uv_vis_input, const t_int &Ju = 4,
		      const t_int &Jv = 4, const std::string &kernel_name = "kb",
		      const t_int &imsizex = 256, const t_int &imsizey = 256,
                      const t_int &norm_iterations = 20, const t_real &oversample_factor = 2,
                      const t_real &cell_x = 1, const t_real &cell_y = 1,
                      const std::string &weighting_type = "none", const t_real &R = 0,
                      bool use_w_term = false, const t_real &energy_fraction = 1,
                      const std::string &primary_beam = "none", bool fft_grid_correction = false);

  MeasurementOperator(const utilities::vis_params &uv_vis_input, Vector<t_real> const &preconditioner, const t_int &Ju = 4,
		      const t_int &Jv = 4, const std::string &kernel_name = "kb",
		      const t_int &imsizex = 256, const t_int &imsizey = 256,
                      const t_int &norm_iterations = 20, const t_real &oversample_factor = 2,
                      const t_real &cell_x = 1, const t_real &cell_y = 1,
                      const std::string &weighting_type = "none", const t_real &R = 0,
                      bool use_w_term = false, const t_real &energy_fraction = 1,
                      const std::string &primary_beam = "none", bool fft_grid_correction = false);



 virtual ~MeasurementOperator() {}

#define PURIPSI_MACRO(NAME, TYPE, VALUE)                                                            \
protected:                                                                                         \
  TYPE NAME##_ = VALUE;                                                                            \
                                                                                                   \
public:                                                                                            \
  TYPE const &NAME() const { return NAME##_; };                                                    \
  MeasurementOperator &NAME(TYPE const &NAME) {                                                    \
    NAME##_ = NAME;                                                                                \
    return *this;                                                                                  \
  };

  PURIPSI_MACRO(Ju, t_int, 4);
  PURIPSI_MACRO(Jv, t_int, 4);
  PURIPSI_MACRO(kernel_name, std::string, "kb");
  PURIPSI_MACRO(norm_iterations, t_int, 20);
  PURIPSI_MACRO(cell_x, t_real, 1);
  PURIPSI_MACRO(cell_y, t_real, 1);
  PURIPSI_MACRO(weighting_type, std::string, "none");
  PURIPSI_MACRO(R, t_real, 0);
  PURIPSI_MACRO(use_w_term, bool, false);
  PURIPSI_MACRO(energy_fraction, t_real, 1.);
  PURIPSI_MACRO(fft_grid_correction, bool, false);
  PURIPSI_MACRO(primary_beam, std::string, "none");
  PURIPSI_MACRO(fftw_plan_flag, std::string, "estimate");
  PURIPSI_MACRO(gradient, std::string, "none");
  //! Reads in visiblities and uses them to construct the operator for use
  MeasurementOperator &construct_operator(const utilities::vis_params &uv_vis_input) {
    MeasurementOperator::init_operator(uv_vis_input);
    return *this;
  };

  // writing definiton of fftoperator so that it is mutable.
protected:
  mutable FFTOperator fftoperator_
      = puripsi::FFTOperator();

public:
  FFTOperator &fftoperator() { return fftoperator_; };
  MeasurementOperator &fftoperator(FFTOperator const &fftoperator) {
    fftoperator_ = fftoperator;
    return *this;
  };
#undef PURIPSI_MACRO
  // Default values
protected:
  t_int ftsizeu_;
  t_int ftsizev_;
  Vector<t_int> fourier_indices_;

public:
  //! Degridding operator that degrids image to visibilities
  virtual Vector<t_complex> degrid(const Image<t_complex> &eigen_image) const;
  virtual Vector<t_complex> preconditioned_degrid(const Image<t_complex> &eigen_image, const Vector<t_real> &preconditioner) const;
  
  //! Gridding operator that grids image from visibilities
  virtual Image<t_complex> grid(const Vector<t_complex> &visibilities) const;
  virtual Image<t_complex> preconditioned_grid(const Vector<t_complex> &visibilities, const Vector<t_real> &preconditioner) const;
  Matrix<t_complex> FFT(const Image<t_complex> &eigen_image) const;
  Image<t_complex> inverse_FFT(Matrix<t_complex> &ft_vector) const;
  Vector<t_complex> G_function(const Matrix<t_complex> &ft_vector) const;
  Vector<t_complex> G_function(const Eigen::SparseMatrix<t_complex> &ft_vector) const;
  Vector<t_complex> G_function_adjoint(const Vector<t_complex> &visibilities) const;
  Vector<t_int> get_fourier_indices() const;
  

protected:
  //! Match uv coordinates to grid
  Vector<t_real> omega_to_k(const Vector<t_real> &omega);
  //! Generates interpolation matrix
  Sparse<t_complex> init_interpolation_matrix2d(const Vector<t_real> &u, const Vector<t_real> &v,
                                                const t_int Ju, const t_int Jv,
                                                const std::function<t_real(t_real)> kernelu,
                                                const std::function<t_real(t_real)> kernelv,
                                                Vector<t_int> &fourier_indices);
  //! Generates scaling factors for gridding correction using an fft
  Image<t_real> init_correction2d_fft(const std::function<t_real(t_real)> kernelu,
                                      const std::function<t_real(t_real)> kernelv, const t_int Ju,
                                      const t_int Jv);
  //! Generates scaling factors for gridding correction
  Image<t_real> init_correction2d(const std::function<t_real(t_real)> ftkernelu,
                                  const std::function<t_real(t_real)> ftkernelv);
  //! Generates and calculates weights
  Array<t_complex> init_weights(const Vector<t_real> &u, const Vector<t_real> &v,
                                const Vector<t_complex> &weights, const t_real &oversample_factor,
                                const std::string &weighting_type, const t_real &R);
  //! Calculate Primary Beam
  Image<t_real>
  init_primary_beam(const std::string &primary_beam, const t_real &cell_x, const t_real &cell_y);

public:
  //! Construct operator
  void init_operator(const utilities::vis_params &uv_vis_input);

public:
  //! Estimates norm of operator
  t_real power_method(const t_int &niters, const t_real &relative_difference = 1e-9);

  psi::LinearTransform<psi::Vector<psi::t_complex>> linear_transform(psi::t_uint nvis, t_int imsizey, t_int imsizex, t_int oversample_factor);

  psi::LinearTransform<psi::Vector<psi::t_complex>> linear_transform(psi::t_uint nvis); 

  psi::LinearTransform<psi::Vector<psi::t_complex>> linear_transform(psi::t_uint nvis, t_int imsizey, t_int imsizex, t_int oversample_factor, Vector<t_real> const &preconditioner);

  psi::LinearTransform<psi::Vector<psi::t_complex>> linear_transform(psi::t_uint nvis, Vector<t_real> const &preconditioner);

};

//! Helper function to create a linear transform from a measurement operator
psi::LinearTransform<psi::Vector<psi::t_complex>>
  linear_transform(std::shared_ptr<MeasurementOperator const> const &measurements, t_uint nvis);
psi::LinearTransform<psi::Vector<psi::t_complex>>
  linear_transform(std::shared_ptr<MeasurementOperator const> const &measurements, t_uint nvis, psi::Vector<psi::t_real> const &preconditioner);


}
#endif
