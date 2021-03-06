#include "puripsi/logging.h"
#include "puripsi/types.h"
#include "puripsi/operators.h"
#include "puripsi/utilities.h"

using namespace puripsi;

int main(int nargs, char const **args) {
  puripsi::logging::initialize();
  puripsi::logging::set_level("debug");
  // Gridding example
  auto const oversample_ratio = 2;
  auto const power_iters = 0;
  auto const power_tol = 1e-4;
  auto const ft_plan = operators::fftw_plan::measure;
  auto const Ju = 4;
  auto const Jv = 4;
  auto const imsizex = 256;
  auto const imsizey = 256;

  auto const kernel = "kb";

  t_uint const number_of_pixels = imsizex * imsizey;
  t_uint const number_of_vis = 4e6; // std::floor(number_of_pixels * 2);
  // Generating random uv(w) coverage
  t_real const sigma_m = constant::pi / 3;
  auto uv_vis = utilities::random_sample_density(number_of_vis, 0, sigma_m);
  uv_vis.units = utilities::vis_units::radians;
  const Vector<t_complex> image = Vector<t_complex>::Random(number_of_pixels);
  const auto measure_op = std::make_shared<psi::LinearTransform<psi::Vector<psi::t_complex>>>(puripsi::operators::MeasurementOperator<Vector<t_complex>, t_complex>(
      uv_vis.u, uv_vis.v, uv_vis.w, uv_vis.weights, imsizey, imsizex, oversample_ratio, power_iters,
      power_tol, kernels::kernel_from_string.at(kernel), Ju, Jv));
  for(t_uint i = 0; i < 100; i++) {
    PURIPSI_LOW_LOG("Iteration: {}", i);
    Vector<t_complex> const measurements = *measure_op * image;
  }
}
