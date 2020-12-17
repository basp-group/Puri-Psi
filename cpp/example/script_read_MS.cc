//
// Created by mjiang on 6/3/19.
//

#include <array>
#include <vector>
#include <memory>
#include <random>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip> // for setw
#include <cstdlib> // for exit()
#include <boost/math/special_functions/erf.hpp>
#include <random>

#include <unsupported/Eigen/SparseExtra> // for Eigen::saveMarket
#include <psi/maths.h>
#include <psi/forward_backward_nnls.h>
#include <psi/relative_variation.h>
#include <psi/positive_quadrant.h>
#include <psi/reweighted_time_blocking.h>
#include <psi/sampling.h>
#include <psi/types.h>
#include <psi/utilities.h>
#include <psi/wavelets.h>
#include <psi/wavelets/sara.h>
#include <psi/power_method.h>
#include <psi/power_method_blocking.h>
#include <psi/primal_dual_time_blocking.h>

#include "puripsi/casacore.h"
#include "puripsi/time_blocking.h"
#include "puripsi/directories.h"
#include "puripsi/pfitsio.h"
#include "puripsi/types.h"
#include "puripsi/utilities.h"
#include "puripsi/logging.h"
#include "puripsi/preconditioner.h"
#include "puripsi/astrodecomposition.h"

using namespace puripsi;
using namespace puripsi::notinstalled;

int main(int nargs, char const **args) {

    // from example at: http://casacore.github.io/casacore/classcasacore_1_1MSSelection.html

    // string msName = "../ms/CYG-C-6680-64.MS";
    string dataName = "/mnt/d/codes/c_cpp/data_puripsi/CYG-A-5-8M10S.MS"; //"/data/pthouvenin/basp_sharing/Ming/CYG-C-6680-64.MS";
    // Create full MeasurementSet
    ::casacore::MeasurementSet ms(dataName);
    // General info on the full ms
    ::casacore::MSMetaData meta_data_ms(&ms, 1.);
    auto nChannels_ms = meta_data_ms.nChans();
    int nScans_ms = meta_data_ms.nScans();
    int nRow = ms.nrow();
    std::cout << nScans_ms << std::endl;
    std::cout << nRow << std::endl;
    // Check spectral content of the full MS
    ::casacore::ArrayColumn<::casacore::Double> freqCols_ms(ms.spectralWindow(),::casacore::MSSpectralWindow::columnName(::casacore::MSSpectralWindow::CHAN_FREQ));
    Eigen::VectorXd frequencies_ms = Eigen::VectorXd::Map(freqCols_ms(0).data(), freqCols_ms(0).nelements(), 1);

    // Image parameters
    t_int imsizey = 1024;
    t_int imsizex = 1024;
    const std::string test_number = "1";

    // Gridding parameters
    t_int const J = 8;
    t_real const over_sample = 2;
    const string kernel = "kb";
    const t_int ftsizeu = imsizex*over_sample;
    const t_int ftsizev = imsizey*over_sample;
    t_real dl = 1.8;
    t_real pixel_size = -1;
    double tol_min = .8;
    double tol_max = 1.2;
    int block_size = 1.1e5;

    t_int n_measurements = 0;
    t_int field_id = 1;
    t_int n_blocks = 1;
    t_int fieldID = 2;
    int channel_index = 0; // define the channel index of interest, consider a loop of this index for hyperspectral (HS) data
    std::set<::casacore::uInt> spws_ids = meta_data_ms.getSpwsForField(fieldID); // get index of the spectral windows associated with the fieldID of interest
    std::vector<utilities::vis_params> uv_data;
    // uv_data = get_time_blocks(dataName, &dl, &pixel_size, &n_blocks, &n_measurements, channel_index, &tol_min, &tol_max, &block_size, field_id);
    
    // loop over the spectral windows (in fine: loop over both spectral window first, and then channel index within each window)
    for(auto it : spws_ids) {
        // for(int ch=0; ch < nChannels_ms[it]; ++ch) { // not activated here for the sake of debugging
        std::cout << it << std::endl;
        std::vector<utilities::vis_params> uv_data_tmp;
        uv_data_tmp = get_time_blocks(dataName, &dl, &pixel_size, &n_blocks, &n_measurements, it, channel_index, &tol_min, &tol_max, &block_size, field_id); // replace channel_index by ch in double loop
        uv_data.insert(uv_data.end(), uv_data_tmp.begin(), uv_data_tmp.end() ); // quite inefficient, needs to be modified later on
        // }
    }  

    return 0;

}
