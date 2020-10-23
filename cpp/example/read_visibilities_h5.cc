/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Board of Trustees of the University of Illinois.         *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://support.hdfgroup.org/ftp/HDF5/releases.  *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * This example shows how to create a compound datatype,
 * write an array which has the compound datatype to the file,
 * and read back fields' subsets.
 */

#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
using std::cout;
using std::endl;

#include <complex>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "H5Cpp.h"
#include <psi/types.h>
#include "puripsi/types.h"
#include "puripsi/utilities.h"
#include "puripsi/logging.h"

using namespace H5;
using namespace puripsi;

typedef struct complex_type{
    double real;
    double imag;
} complex_type;


int main(void)
{
	puripsi::logging::initialize();
	puripsi::logging::set_level("critical");

    // Try block to detect exceptions raised by any of the calls inside it
    try
    {
        H5File file( "y_N=256_L=60_p=1_snr=40.h5", H5F_ACC_RDONLY ); // /mnt/d/codes/matlab/hyper-sara/tests/uv_model_N=256_L=60_p=1.h5 /mnt/d/codes/matlab/hyper-sara/data/y_N=256_L=15_p=1_snr=40.h5

        DataSet dataset = file.openDataSet( "operator_norm" );
        double op_norm;
        dataset.read(&op_norm, PredType::NATIVE_DOUBLE);
        dataset.close();

        // extract number of blocks and channels from the size of the "epsilon" dataset
        dataset = file.openDataSet( "epsilon" );
        DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        hsize_t dims_epsilon[rank];
        int ndims = dataspace.getSimpleExtentDims( dims_epsilon, NULL);
        cout << "l2_ball_epsilon rank " << rank << ", dimensions " <<
                (unsigned long)(dims_epsilon[0]) << " x " <<
                (unsigned long)(dims_epsilon[1]) << endl;

        hsize_t band_number = dims_epsilon[1]; // 60
        hsize_t n_blocks = dims_epsilon[0];    // 4 
        
        // extract l2ball_epsilon[f][b]
        psi::Vector<psi::Vector<psi::t_real>> l2ball_epsilon(band_number); // Decomp.my_number_of_frequencies()

        for(hsize_t f = 0; f < band_number; f++){
            l2ball_epsilon[f] = psi::Vector<psi::t_real>::Zero(n_blocks);
            
            hsize_t col_dims[1] = {n_blocks};
            DataSpace memspace(1, col_dims);
            hsize_t offset_out[2] = {0, f};       // hyperslab offset in memory
            hsize_t count_out[2] = {n_blocks, 1}; // size of the hyperslab in memory
            dataspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out );
            dataset.read( l2ball_epsilon[f].data(), PredType::NATIVE_DOUBLE, memspace, dataspace );
            memspace.close(); // not necessary, memsapce released once out of scope

            for (hsize_t b = 0; b < n_blocks; b++)
            {
                PURIPSI_HIGH_LOG("l2ball_epsilon[{}]({}) = {}", f, b,  l2ball_epsilon[f](b) );
            }  
        }
        dataspace.close();
        dataset.close();

        std::vector<std::vector<psi::Image<psi::t_real>>> uv_model(band_number);
        // extract per-block uv model
        for(hsize_t f = 0; f < band_number; f++){
            uv_model[f].reserve(n_blocks);
            for(hsize_t b = 0; b < n_blocks; b++){
                // extract uvw for block [f][b]
                std::string datasetname = "uvw" + std::to_string(f*n_blocks+b+1);
                dataset = file.openDataSet( datasetname );
                dataspace = dataset.getSpace();
                rank = dataspace.getSimpleExtentNdims();
                hsize_t dims_uvw_block[rank];
                int ndims_block = dataspace.getSimpleExtentDims( dims_uvw_block, NULL);

                psi::Image<psi::t_real> temp_uvw = psi::Image<psi::t_real>::Zero(dims_uvw_block[1], dims_uvw_block[0]); //! inversion of the dimensions needed due to column major in Eigen, row major in HDF5, take results in rows afterwards
                dataset.read(temp_uvw.data(), PredType::NATIVE_DOUBLE);

                uv_model[f].emplace_back(temp_uvw);
                PURIPSI_HIGH_LOG("uv_model[{}][{}]: {} x {}", f, b, temp_uvw.rows(), temp_uvw.cols());
                dataspace.close();
                dataset.close(); 
            }
        }

        PURIPSI_HIGH_LOG("uv_model[0][0]: u[0] = {}",  uv_model[0][0](0,0));
        PURIPSI_HIGH_LOG("uv_model[0][0]: v[0] = {}",  uv_model[0][0](1,0));
        PURIPSI_HIGH_LOG("uv_model[0][0]: w[0] = {}",  uv_model[0][0](2,0));
       
        PURIPSI_HIGH_LOG("uv_model[0][0]: u = {}",  uv_model[0][0].row(0));
        PURIPSI_HIGH_LOG("uv_model[0][0]: v = {}",  uv_model[0][0].row(1));
        PURIPSI_HIGH_LOG("uv_model[0][0]: w = {}",  uv_model[0][0].row(2));

        // extract visibilities target[f][b]
        CompType complex_data_type(sizeof(complex_type));
        complex_data_type.insertMember( "real", 0, PredType::NATIVE_DOUBLE);
        complex_data_type.insertMember( "imag", sizeof(double), PredType::NATIVE_DOUBLE);
        std::vector<std::vector<psi::Vector<psi::t_complex>>> target(band_number); // Decomp.my_number_of_frequencies()
        
        for(hsize_t f=0; f<band_number; f++){            
            target[f].reserve(n_blocks); // Decomp.my_frequencies()[f].number_of_time_blocks

            for (hsize_t b=0; b<n_blocks; b++){

                // extract visibilities target[f][b]
                std::string datasetname = "y" + std::to_string(f*n_blocks+b+1);
                dataset = file.openDataSet( datasetname );
                dataspace = dataset.getSpace();
                rank = dataspace.getSimpleExtentNdims();
                hsize_t dims_target[rank];
                int ndims_target = dataspace.getSimpleExtentDims( dims_target, NULL);

                Eigen::VectorXcd y = Eigen::VectorXcd::Zero(dims_target[0]);
                dataset.read(y.data(), complex_data_type);
                target[f].emplace_back(y);

                PURIPSI_HIGH_LOG("target[{}][{}].size = {}", f, b, target[f][b].size());
                dataspace.close();
                dataset.close();
            }  
        }
        file.close();
        // PURIPSI_HIGH_LOG("target[{}][{}] = {}", 0, 0, target[0][0]);
    }  // end of try block
 
    // catch failure caused by the H5File operations
    catch(FileIException error)
    {
    error.printErrorStack();
    return -1;
    }

    // catch failure caused by the DataSet operations
    catch(DataSetIException error)
    {
    error.printErrorStack();
    return -1;
    }

    // catch failure caused by the DataSpace operations
    catch(DataSpaceIException error)
    {
    error.printErrorStack();
    return -1;
    }

    return 0;
}
