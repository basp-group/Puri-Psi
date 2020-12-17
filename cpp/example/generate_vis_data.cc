#include "puripsi/config.h"
#include <array>
#include <memory>
#include <random>
#include "puripsi/operators.h"
#include "puripsi/directories.h"
#include "puripsi/logging.h"
#include "puripsi/pfitsio.h"
#include "puripsi/types.h"
#include "puripsi/utilities.h"

int main(int nargs, char const **args) {
  using namespace puripsi;
  using namespace puripsi::notinstalled;
  puripsi::logging::initialize();
  puripsi::logging::set_level(puripsi::default_logging_level());

  std::string const fitsfile = image_filename("M31.fits");
  std::string const inputfile = output_filename("M31_input.fits");

  std::string const kernel = "kb";
  t_real const over_sample = 5.;
  t_int const J = 24;

  t_real m_over_n;
  std::string test_number;
  if(nargs != 3){
    PURIPSI_CRITICAL("Expected two input parameters, m_over_n and test_number. Defaulting to 1 for both");
    m_over_n = 1;
    test_number = 1;
  }else{
    m_over_n = std::stod(static_cast<std::string>(args[1]));
    test_number = static_cast<std::string>(args[2]);
  }

  std::string const vis_file = output_filename("M31_vis_" + test_number + ".vis");

  auto M31 = pfitsio::read2d(fitsfile);
  t_real const max = M31.array().abs().maxCoeff();
  M31 = M31 * 1. / max;
  pfitsio::write2d(M31.real(), inputfile);
  // Following same formula in matlab example
  t_real const sigma_m = constant::pi / 3;
  // t_int const number_of_vis = std::floor(p * rho * M31.size());
  t_int const number_of_vis = std::floor(m_over_n * M31.size());
  // Generating random uv(w) coverage
  auto uv_data = utilities::random_sample_density(number_of_vis, 0, sigma_m);
  uv_data.units = utilities::vis_units::radians;

  PURIPSI_HIGH_LOG("Number of measurements: {}", uv_data.u.size());
  auto measurements_transform = std::make_shared<psi::LinearTransform<psi::Vector<psi::t_complex>>>(puripsi::operators::MeasurementOperator<Vector<t_complex>, t_complex>(
      uv_data.u, uv_data.v, uv_data.w, uv_data.weights, M31.cols(), M31.rows(), over_sample, 100,
      1e-4, kernels::kernel_from_string.at(kernel), J, J));
  uv_data.vis = *measurements_transform * Vector<t_complex>::Map(M31.data(), M31.size());
  utilities::write_visibility(uv_data, vis_file);
}
