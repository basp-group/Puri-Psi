#ifndef PURIPSI_TYPES_H
#define PURIPSI_TYPES_H

#ifdef PURIPSI_CImg
#include <CImg.h>
#ifdef Success
#undef Success
#endif
#ifdef Complex
#undef Complex
#endif
#ifdef Bool
#undef Bool
#endif
#ifdef None
#undef None
#endif
#ifdef Status
#undef Status
#endif
#endif

#include "puripsi/config.h"
#include <complex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <psi/types.h>

namespace puripsi {
using psi::Array;
using psi::Image;
using psi::Matrix;
using psi::Vector;
using psi::t_complex;
using psi::t_int;
using psi::t_real;
using psi::t_uint;

typedef std::complex<float> t_complexf;
//! Root of the type hierarchy for triplet lists
typedef Eigen::Triplet<t_complex> t_tripletList;

//! \brief A matrix of a given type
//! \details Operates as mathematical sparse matrix.
template <class T = t_real> using Sparse = Eigen::SparseMatrix<T, Eigen::RowMajor>;
template <class T = t_real> using SparseVector = Eigen::SparseVector<T>;

#ifdef PURIPSI_CImg
//! Image type of CImg library
template <class T = t_real> using CImage = cimg_library::CImg<T>;
template <class T = t_real> using CImageList = cimg_library::CImgList<T>;
//! Display used to display CImg images
typedef cimg_library::CImgDisplay CDisplay;
#endif

namespace constant {
//! mathematical constant
const t_real pi = 3.14159265358979323846;
//! speed of light in vacuum
const t_real c = 299792458.0;
}
}
#endif
