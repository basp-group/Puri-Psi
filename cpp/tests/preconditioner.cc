#include <catch2/catch.hpp>
#include <Eigen/Core>
// #include <limits>       // std::numeric_limits
#include <fstream>
#include <iostream>
#include "psi/sort_utils.h"
#include "puripsi/preconditioner.h"
#include "puripsi/utilities.h"

TEST_CASE("Test preconditioning matrix, [preconditioner]"){

    std::vector<Eigen::VectorXd> uvaW = puripsi::utilities::read_uvaW("../../../data/test/preconditioner/uvaW.vis"); // file name ok with ctest, needs to be changed for debugging
    // problem to find the appropriate file...
    // matlab generated preconditioning matrix for the u-v points specified in the text file "uvaW.vis"
    Eigen::VectorXd aW = Eigen::VectorXd::Ones(uvaW[0].size());
    const int Ny = 10;
    const int Nx = 10;
    puripsi::preconditioner<double>(aW, uvaW[0], uvaW[1], Ny, Nx);
    CHECK((aW - uvaW[2]).stableNorm() <= 1e-15);
}
