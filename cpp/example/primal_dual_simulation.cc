#include "puripsi/config.h"
#include <array>
#include <ctime>
#include <memory>
#include <random>
#include <boost/math/special_functions/erf.hpp>
#include <psi/primal_dual.h>
#include <psi/relative_variation.h>
#include <psi/utilities.h>
#include <psi/wavelets.h>
#include <psi/wavelets/sara.h>
#include <psi/power_method.h>
#include "puripsi/operators.h"
#include "puripsi/directories.h"
#include "puripsi/logging.h"
#include "puripsi/pfitsio.h"
#include "puripsi/types.h"
#include "puripsi/utilities.h"

int main(int nargs, char const **args) {
  if(nargs != 9) {
    std::cerr << " Wrong number of arguments! " << '\n';
    return 1;
  }

  using namespace puripsi;
  using namespace puripsi::notinstalled;
  psi::logging::initialize();
  puripsi::logging::initialize();
  psi::logging::set_level("info");
  puripsi::logging::set_level("info");

  
  std::string const test_type = args[1];
  std::string const kernel = args[2];
  t_real const over_sample = std::stod(static_cast<std::string>(args[3]));
  t_int const J = static_cast<t_int>(std::stod(static_cast<std::string>(args[4])));
  t_real const m_over_n = std::stod(static_cast<std::string>(args[5]));
  std::string const test_number = static_cast<std::string>(args[6]);
  t_real const ISNR = std::stod(static_cast<std::string>(args[7]));
  std::string const name = static_cast<std::string>(args[8]);

  std::string const fitsfile = image_filename(name + ".fits");

  std::string const dirty_image_fits
      = output_filename(name + "_dirty_" + kernel + "_" + test_number + ".fits");
  std::string const results
      = output_filename(name + "_results_" + kernel + "_" + test_number + ".txt");

  PURIPSI_MEDIUM_LOG("Input file: {}", fitsfile);

  auto sky_model = pfitsio::read2d(fitsfile);
  auto sky_model_max = sky_model.array().abs().maxCoeff();
  sky_model = sky_model / sky_model_max;
  t_int const number_of_vis = std::floor(m_over_n * sky_model.size());
  t_real const sigma_m = constant::pi / 3;

  auto uv_data = utilities::random_sample_density(number_of_vis, 0, sigma_m);
  uv_data.units = utilities::vis_units::radians;
  PURIPSI_MEDIUM_LOG("Number of measurements: {}", uv_data.u.size());

  auto measurements_sky = *measurementoperator::init_degrid_operator_2d<Vector<t_complex>>(
      uv_data.u, uv_data.v, uv_data.w, uv_data.weights, sky_model.cols(), sky_model.rows(),
      over_sample, 100, 1e-4, kernels::kernel_from_string.at(kernel), 8, 8);
  uv_data.vis = measurements_sky * Vector<t_complex>::Map(sky_model.data(), sky_model.size());
  auto measurements_transform = *measurementoperator::init_degrid_operator_2d<Vector<t_complex>>(
      uv_data.u, uv_data.v, uv_data.w, uv_data.weights, sky_model.cols(), sky_model.rows(),
      over_sample, 100, 1e-4, kernels::kernel_from_string.at(kernel), J, J);

  std::vector<std::tuple<std::string, t_uint>> wavelets;
	
  if (test_type == "clean")
  	wavelets.push_back(std::make_tuple("Dirac" ,3u));
  if (test_type == "ms_clean")
  	wavelets.push_back(std::make_tuple("DB4" ,3u));
  if (test_type == "primaldual"){
  	wavelets.push_back(std::make_tuple("Dirac" ,3u));
  	wavelets.push_back(std::make_tuple("DB1" ,3u));
  	wavelets.push_back(std::make_tuple("DB2" ,3u));
  	wavelets.push_back(std::make_tuple("DB3" ,3u));
  	wavelets.push_back(std::make_tuple("DB4" ,3u));
  	wavelets.push_back(std::make_tuple("DB5" ,3u));
  	wavelets.push_back(std::make_tuple("DB6" ,3u));
  	wavelets.push_back(std::make_tuple("DB7" ,3u));
  	wavelets.push_back(std::make_tuple("DB8" ,3u));
  }
  psi::wavelets::SARA const sara(wavelets.begin(), wavelets.end());
  auto const Psi
    = psi::linear_transform<t_complex>(sara, sky_model.rows(), sky_model.cols());

  // working out value of sigma given SNR of 30
  t_real sigma = utilities::SNR_to_standard_deviation(uv_data.vis, ISNR);
  // adding noise to visibilities
  uv_data.vis = utilities::add_noise(uv_data.vis, 0., sigma);

  Vector<> dimage = (measurements_transform.adjoint() * uv_data.vis).real();
  t_real const max_val = dimage.array().abs().maxCoeff();
  dimage = dimage / max_val;
  Vector<t_complex> initial_estimate = Vector<t_complex>::Zero(dimage.size());

  auto const nlevels = sara.size();

  auto const epsilon = utilities::calculate_l2_radius(uv_data.vis, sigma, 2.0,"chi^2");

  t_int iters = 0;
  auto convergence_function = [&iters](const Vector<t_complex> &x) { iters = iters + 1; return true; };

  PURIPSI_MEDIUM_LOG("epsilon is {} ", epsilon);
  
  auto const tau = 0.49;
  PURIPSI_MEDIUM_LOG("tau is {} ", tau);

  psi::Vector<t_complex> rand = psi::Vector<t_complex>::Random(sky_model.rows() * sky_model.cols() * nlevels);
  PURIPSI_HIGH_LOG("Setting up power method to calculate sigma values for primal dual");
  auto const pm = psi::algorithm::PowerMethod<psi::t_complex>().tolerance(1e-6).itermax(10000);

  PURIPSI_HIGH_LOG("Calculating sigma1");
  auto const nu1data = pm.AtA(Psi, rand);
  auto const nu1 = nu1data.magnitude.real();
  auto sigma1 = 1e0 / nu1;
  PURIPSI_MEDIUM_LOG("sigma1 is {} ", sigma1);
  
  rand = psi::Vector<t_complex>::Random(sky_model.rows() * sky_model.cols() * (over_sample/2));

  PURIPSI_HIGH_LOG("Calculating sigma2");  
  auto const nu2data = pm.AtA(measurements_transform, rand);
  auto const nu2 = nu2data.magnitude.real();
  auto sigma2 = 1e0 / nu2;
  PURIPSI_MEDIUM_LOG("sigma2 is {} ", sigma2);
  
    PURIPSI_HIGH_LOG("Calculating kappa");  
  auto const kappa = ((measurements_transform.adjoint() * uv_data.vis).real().maxCoeff() * 1e-3) / nu2;  
  PURIPSI_MEDIUM_LOG("kappa is {} ", kappa);

  PURIPSI_HIGH_LOG("Starting psi!");
  auto pd = psi::algorithm::PrimalDual<t_complex>(uv_data.vis)
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
			 .itermax(500)
			 .is_converged(convergence_function)
                         .Phi(measurements_transform);

  // Timing reconstruction
  std::clock_t c_start = std::clock();
  auto diagnostic = pd();
  std::clock_t c_end = std::clock();

  // Reading if algo has converged
  t_int converged = 0;
  if (diagnostic.good) {
	converged = 1;
  }
  const t_uint maxiters = iters;

  Image<t_complex> image
    = Image<t_complex>::Map(diagnostic.x.data(), sky_model.rows(), sky_model.cols());

  Vector<t_complex> original = Vector<t_complex>::Map(sky_model.data(), sky_model.size(), 1);
  Image<t_complex> res = sky_model - image;
  Vector<t_complex> residual = Vector<t_complex>::Map(res.data(), image.size(), 1);

  auto snr = 20. * std::log10(original.norm() / residual.norm()); // SNR of reconstruction
  auto total_time = (c_end - c_start) / CLOCKS_PER_SEC; // total time for solver to run in seconds
  // writing snr and time to a file
  std::ofstream out(results);
  out.precision(13);
  out << snr << " " << total_time << " " << converged << " " << maxiters;
  out.close();

  return 0;
}
