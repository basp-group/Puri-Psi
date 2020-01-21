#ifndef PURI_PSI_PRECONDITIONER
#define PURI_PSI_PRECONDITIONER

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include "psi/types.h"
#include "psi/sort_utils.h"

namespace puripsi{

template<typename T>
void preconditioner(Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>> aW, const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>& u, const Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, 1>>& v, const int Ny, const int Nx){

    using psiVector = typename psi::Vector<T>;

    aW = psiVector::Ones(v.size());
    psiVector lsv = psiVector::LinSpaced(Ny + 1, -EIGEN_PI, EIGEN_PI);
    psiVector lsu = psiVector::LinSpaced(Nx + 1, -EIGEN_PI, EIGEN_PI); 

    psiVector v_ = v;
    psi::Vector<psi::t_uint> sv(v.size());
    std::sort(v_.data(), v_.data() + v_.size()); // get the sorted out vector
    psi::sort_indices(v, sv);                    // keep the indices from the original vector
    
    for(int k = 0; k < Ny; ++k){
        std::pair<int, int> indices_v;
        int sfv_l = psi::lower_bound_index(v_.data(), v_.data() + v_.size(), lsv(k));          // first index >= lsv(k)
        int sfv_h = psi::strict_upper_bound_index(v_.data(), v_.data() + v_.size(), lsv(k+1)); // last index < lsv(k+1)
        if(sfv_l >= 0 && sfv_h >= 0){ // sfv_h - sfv_l + 1 > 0 (no failure for the two preceding functions)
            psi::Vector<psi::t_uint> sfv = sv.segment(sfv_l, sfv_h - sfv_l + 1); // sv(sfv_l:sfv_h); [PA] extract the corresponding indices in sfv 
            psiVector u_(sfv.size()); 
            psi::Vector<psi::t_uint> su(sfv.size());
            for(int m = 0; m < sfv.size(); ++m){
                u_(m) = u(sfv(m));
            }
            psi::sort_indices(u_, su);
            std::sort(u_.data(), u_.data() + u_.size());
            for(int j = 0; j < Nx; ++j){
                int sfu_l = psi::lower_bound_index(u_.data(), u_.data() + u_.size(), lsu(j));          // first index >= lsu(j)
                int sfu_h = psi::strict_upper_bound_index(u_.data(), u_.data() + u_.size(), lsu(j+1)); // last index < lsu(j+1)
                if(sfu_l >= 0 && sfu_h >= 0){ // sfv_h - sfv_l + 1 > 0 (no failure for the two preceding functions)
                    psi::Vector<psi::t_uint> sfu = su.segment(sfu_l, sfu_h - sfu_l + 1);
                    for(int n = 0; n < sfu.size(); ++n){
                        aW(sfv(sfu(n))) = 1./static_cast<T>(sfu.size());
                    }
                }
            }
        }
    }
    return;
} // end preconditioner
} // end namespace puripsi
#endif
