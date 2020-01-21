#include <array>
#include <memory>
#include <random>
#include <boost/math/special_functions/erf.hpp>
#include <psi/preconditioned_primal_dual.h>
#include <psi/relative_variation.h>
#include <psi/utilities.h>
#include <psi/wavelets.h>
#include <psi/wavelets/sara.h>
#include <psi/power_method.h>
#include "puripsi/operators.h"
#include "puripsi/preconditioner.h"
#include "puripsi/directories.h"
#include "puripsi/pfitsio.h"
#include "puripsi/types.h"
#include "puripsi/utilities.h"
#include "puripsi/logging.h"

using namespace puripsi;
using namespace puripsi::notinstalled;

void pd(const std::string & name, const Image<t_complex> & M31, const std::string & kernel, const t_int J,  utilities::vis_params & uv_data, const t_real sigma, const t_real cellsize, const bool wterm){
  std::string const outfile = output_filename(name + "_" + kernel + ".tiff");
  std::string const outfile_fits = output_filename(name +  "_" + kernel + "_solution.fits");
  std::string const residual_fits = output_filename(name +  "_" + kernel + "_residual.fits");
  std::string const dirty_image = output_filename(name +  "_" + kernel + "_dirty.tiff");
  std::string const dirty_image_fits = output_filename(name +  "_" + kernel + "_dirty.fits");

  // Initialise the random number generator used by Eigen to a fixed seed to allow for reproducible tests
  std::srand(10000);

  t_real const over_sample = 2;
  t_uint const imsizey = M31.rows();
  t_uint const imsizex = M31.cols();
  auto const measurements_transform =
      measurementoperator::init_degrid_operator_2d<Vector<t_complex>>(
	  uv_data, imsizey, imsizex, cellsize, cellsize, over_sample, 1000,
          0.0001, kernels::kernel_from_string.at(kernel), J, J, wterm);


  psi::wavelets::SARA const sara{
      std::make_tuple("Dirac", 3u), std::make_tuple("DB1", 3u), std::make_tuple("DB2", 3u),
      std::make_tuple("DB3", 3u),   std::make_tuple("DB4", 3u), std::make_tuple("DB5", 3u),
      std::make_tuple("DB6", 3u),   std::make_tuple("DB7", 3u), std::make_tuple("DB8", 3u)};

  auto const Psi
      = psi::linear_transform<t_complex>(sara, imsizey, imsizex);

  auto const nlevels = sara.size();


  auto const epsilon = utilities::calculate_l2_radius(uv_data.vis, sigma, 2,"chi^2");

  PURIPSI_MEDIUM_LOG("epsilon is {} ", epsilon);
  
  auto const tau = 0.49;
  PURIPSI_MEDIUM_LOG("tau is {} ", tau);

  psi::Vector<t_complex> rand = psi::Vector<t_complex>::Random(imsizey * imsizex * nlevels);
  PURIPSI_HIGH_LOG("Setting up power method to calculate sigma values for primal dual");
  auto const pm = psi::algorithm::PowerMethod<psi::t_complex>().tolerance(1e-6).itermax(10000);

  PURIPSI_HIGH_LOG("Calculating sigma1");
  auto const nu1data = pm.AtA(Psi, rand);
  auto const nu1 = nu1data.magnitude.real();
  auto sigma1 = 1e0 / nu1;
  PURIPSI_MEDIUM_LOG("sigma1 is {} ", sigma1);
  
  rand = psi::Vector<t_complex>::Random(imsizey * imsizex * (over_sample/2));

  PURIPSI_HIGH_LOG("Calculating sigma2");  
  auto const nu2data = pm.AtA(*measurements_transform, rand);
  auto const nu2 = nu2data.magnitude.real();
  auto sigma2 = 1e0 / nu2;
  PURIPSI_MEDIUM_LOG("sigma2 is {} ", sigma2);
  
    PURIPSI_HIGH_LOG("Calculating kappa");  
  auto const kappa = ((measurements_transform->adjoint() * uv_data.vis).real().maxCoeff() * 1e-3) / nu2;  
  PURIPSI_MEDIUM_LOG("kappa is {} ", kappa);
  
  Vector<> dimage = (measurements_transform->adjoint() * uv_data.vis).real();
  Vector<t_complex> initial_estimate = Vector<t_complex>::Zero(dimage.size());
  psi::utilities::write_tiff(Image<t_real>::Map(dimage.data(), imsizey, imsizex), dirty_image);
  pfitsio::write2d(Image<t_real>::Map(dimage.data(), imsizey, imsizex), dirty_image_fits);

  Vector<double> Ui = Vector<double>::Ones(uv_data.vis.size());
  puripsi::preconditioner<double>(Ui, uv_data.u, uv_data.v, imsizey, imsizex);

  PURIPSI_HIGH_LOG("Creating preconditioned primal-dual Functor");
  auto ppd
      = psi::algorithm::PreconditionedPrimalDual<t_complex>(uv_data.vis)
            .Ui(Ui)
            .itermax(300)
            .tau(tau)
            .kappa(kappa)
            .sigma1(sigma1)
            .sigma2(sigma2)
            .levels(nlevels)
            .l2ball_epsilon(epsilon)
            .nu(nu2)
            .relative_variation(1e-5)
            .positivity_constraint(true)
            .residual_convergence(epsilon * 1.001)
            .Psi(Psi)
            .Phi(*measurements_transform);

  PURIPSI_HIGH_LOG("Starting psi preconditioned primal dual");
  auto diagnostic = ppd();
  if(not diagnostic.good){
    PURIPSI_HIGH_LOG("preconditioned primal dual did not converge in {} iterations", diagnostic.niters);
  }else{
    PURIPSI_HIGH_LOG("preconditioned primal dual returned in {} iterations", diagnostic.niters);
  }
  assert(diagnostic.x.size() == M31.size());
  Image<t_complex> image
      = Image<t_complex>::Map(diagnostic.x.data(), imsizey, imsizex);
  pfitsio::write2d(image.real(), outfile_fits);
  Vector<t_complex> residuals = measurements_transform->adjoint() *
                                (uv_data.vis - ((*measurements_transform) * diagnostic.x));
  Image<t_complex> residual_image = Image<t_complex>::Map(residuals.data(), imsizey, imsizex);
  pfitsio::write2d(residual_image.real(), residual_fits);

};


int main(int, char **) {
  psi::logging::initialize();
  puripsi::logging::initialize();
  psi::logging::set_level("debug");
  puripsi::logging::set_level("debug");
  const std::string & name = "30dor_256";
  const t_real snr = 30;
  std::string const fitsfile = image_filename(name + ".fits");
  auto M31 = pfitsio::read2d(fitsfile);
  std::string const inputfile = output_filename(name + "_" + "input.fits");
  
  const t_real cellsize = 1;
  const bool wterm = false;
  const t_uint Jw = 6;

  t_real const max = M31.array().abs().maxCoeff();
  M31 = M31 * 1. / max;
  pfitsio::write2d(M31.real(), inputfile);
  
  t_int const number_of_pixels = M31.size();
 t_int const number_of_vis = std::floor( number_of_pixels * 2.);
  // Generating random uv(w) coverage
  t_real const sigma_m = constant::pi / 3;
  auto uv_data = utilities::random_sample_density(number_of_vis, 0, sigma_m, 0);
  PURIPSI_MEDIUM_LOG("Number of measurements / number of pixels: {} ",  uv_data.u.size() * 1. / number_of_pixels);
  auto const sky_measurements = measurementoperator::init_degrid_operator_2d<Vector<t_complex>>(
      uv_data, M31.rows(), M31.cols(), cellsize, cellsize, 2, 1000, 0.0001, 
      kernels::kernel_from_string.at("kb"), 8, 8, wterm);
  uv_data.vis = (*sky_measurements) * Image<t_complex>::Map(M31.data(), M31.size(), 1);
  Vector<t_complex> const y0 = uv_data.vis;
  // working out value of signal given SNR of 30
  t_real const sigma = utilities::SNR_to_standard_deviation(y0, snr);

  // adding noise to visibilities
  uv_data.vis = utilities::add_noise(y0, 0., sigma);
  
  uv_data.units = utilities::vis_units::radians;
 
  // adding noise to visibilities
  uv_data.vis = utilities::add_noise(y0, 0., sigma);
  pd(name + "30", M31, "kb", 4, uv_data, sigma, cellsize, wterm);
  return 0;
}
