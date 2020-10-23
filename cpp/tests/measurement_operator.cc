#include <catch2/catch.hpp>
#include "puripsi/operators.h"
#include "puripsi/pfitsio.h"
#include "puripsi/utilities.h"
#include "puripsi/types.h"
#include "puripsi/logging.h"

using namespace puripsi;

TEST_CASE("Measurement operator [ADJOINT]", "[ADJOINT]") {

    // Image dimensions
    const t_int imsizey = 128;
    const t_int imsizex = 128;
    const t_real over_sample = 2;
    const t_uint J = 4;
    // const kernels::kernel kernel = kernels::kernel::kb;
    const std::string & kernel = "kb";
    t_int const number_of_pixels = imsizey*imsizex;
    t_int const number_of_vis = std::floor( number_of_pixels * 2.);
    const t_real cellsize = 1;
    const bool wterm = false;

    // Generating random uv(w) coverage
    t_real const sigma_m = constant::pi / 3;
    auto uv_data = utilities::random_sample_density(number_of_vis, 0, sigma_m, true);

    auto const measurements_transform =
      measurementoperator::init_degrid_operator_2d<Vector<t_complex>>(
	  uv_data, imsizey, imsizex, cellsize, cellsize, over_sample, 1000,
          0.0001, kernels::kernel_from_string.at(kernel), J, J, wterm);

    Vector<t_complex> x = Vector<t_complex>::Random(number_of_pixels);
    Vector<t_complex> y = Vector<t_complex>::Random(number_of_vis);

    Vector<t_complex> Phix = (*measurements_transform) * x;
    Vector<t_complex> Phity = measurements_transform->adjoint() * y; 
    
    // Test conjugacy of the operators fft2 and ifft2
    t_complex p0 = (Phix.array().conjugate() * y.array()).sum();
    t_complex p1 = (x.array().conjugate() * Phity.array()).sum();
    CHECK(std::abs(p0 - p1)/std::abs(p0) < 1e-13);
}
