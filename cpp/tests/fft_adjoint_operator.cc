#include <catch2/catch.hpp>
#include "puripsi/operators.h"
#include "puripsi/pfitsio.h"
#include "puripsi/utilities.h"
#include "puripsi/types.h"
#include "puripsi/logging.h"

using namespace puripsi;

TEST_CASE("FFT Operator 2 [CONJUGACY]", "[CONJUGACY]") {

    // Image dimensions
    const t_int imsizey = 128;
    const t_int imsizex = 128;
    const t_real oversample_ratio = 1;
    const t_uint ftsizev = std::floor(imsizey * oversample_ratio);
    const t_uint ftsizeu = std::floor(imsizex * oversample_ratio);

    // Define fft operator
    psi::OperatorFunction<Vector<t_complex>> directFFT, indirectFFT;
    std::tie(directFFT, indirectFFT)
        = operators::init_FFT_2d<Vector<t_complex>>(imsizey, imsizex, oversample_ratio);

    const Vector<t_complex> x0 = Vector<t_complex>::Random(ftsizev * ftsizeu);
    Vector<t_complex> Fx0;
    directFFT(Fx0, x0);

    CHECK(Fx0.size() == ftsizeu * ftsizev);
    
    const Vector<t_complex> y0 = Vector<t_complex>::Random(ftsizev * ftsizeu);
    Vector<t_complex> Fty0;
    indirectFFT(Fty0, y0);
    CHECK(Fty0.size() == ftsizev * ftsizeu);     
    
    // Test conjugacy of the operators fft2 and K*ifft2
    t_complex p0 = (Fx0.array().conjugate() * y0.array()).sum();
    t_complex p1 = (x0.array().conjugate() * Fty0.array()).sum();
    CHECK(std::abs(p0 - p1)/std::abs(p0) < 1e-13);
}
