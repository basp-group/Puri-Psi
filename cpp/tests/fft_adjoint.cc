#include <catch2/catch.hpp>
#include "puripsi/FFTOperator.h"
#include "puripsi/pfitsio.h"
#include "puripsi/utilities.h"
#include "puripsi/types.h"
#include "puripsi/logging.h"

using namespace puripsi;

TEST_CASE("FFT Operator [CONJUGACY]", "[CONJUGACY]") {

    // Image dimensions
    const t_int imsizey = 128;
    const t_int imsizex = 128;

    // Define fft operator
    // t_int fft_flag = (FFTW_PATIENT | FFTW_PRESERVE_INPUT);
    auto fft_operator = puripsi::FFTOperator(); //.fftw_flag(fft_flag);

    // Generate random images
    psi::Matrix<t_complex> x0 = psi::Matrix<t_complex>::Random(imsizey, imsizex);
    psi::Matrix<t_complex> x1 = psi::Matrix<t_complex>::Random(imsizey, imsizex);

    // Compute fft2(x0) anf ifft2(x1)
    Matrix<t_complex> Fx0 = fft_operator.forward(x0);
    Matrix<t_complex> Ftx1 = fft_operator.inverse(x1);
    
    // Test conjugacy of the operators fft2 and ifft2
    t_complex p0 = (Fx0.array().conjugate() * x1.array()).sum();
    t_complex p1 = static_cast<t_complex>(imsizey*imsizex)*(x0.array().conjugate() * Ftx1.array()).sum();
    CHECK(std::abs(p0 - p1)/std::abs(p0) < 1e-13);
}
