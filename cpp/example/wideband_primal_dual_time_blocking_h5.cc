
#include <array>
#include <vector>
#include <memory>
#include <random>
#include <string>
#include <fstream>
#include <iostream>
//#include <iomanip>
//#include <cstdlib>
#include <boost/math/special_functions/erf.hpp>
#include <mpi.h>
#include "H5Cpp.h"

#include <psi/maths.h>
#include <psi/forward_backward_nnls.h>
#include <psi/relative_variation.h>
#include <psi/positive_quadrant.h>
#include <psi/reweighted_wideband.h>
#include <psi/sampling.h>
#include <psi/types.h>
#include <psi/utilities.h>
#include <psi/wavelets.h>
#include <psi/wavelets/sara.h>
#include <psi/power_method_wideband.h>
#include <psi/primal_dual_wideband_blocking.h>

#include "puripsi/directories.h"
#include "puripsi/pfitsio.h"
#include "puripsi/types.h"
#include "puripsi/utilities.h"
#include "puripsi/logging.h"
#include "puripsi/MeasurementOperator.h"
#include "puripsi/preconditioner.h"
#include "puripsi/astrodecomposition.h"

using namespace H5;
using namespace puripsi;

typedef struct complex_type{
    double real;
    double imag;
} complex_type;

using namespace puripsi;
using namespace puripsi::notinstalled;

int main(int argc, const char **argv) {
	psi::logging::initialize();
	puripsi::logging::initialize();
	psi::logging::set_level("critical");
	puripsi::logging::set_level("critical");

	Eigen::initParallel();

	// Image parameters
	t_int imsizey = 256;
	t_int imsizex = 256;
	auto const input_snr = 40.; // in dB

	// Gridding parameters
	t_int const J = 8;
	t_real const over_sample = 2;
	const string kernel = "kb";
	t_real pixel_size = 1;

    // Visibility file (from Matlab)
    std::string visname = argc >= 1 ? argv[0] : "y_N=256_L=15_p=1_snr=40.h5"; // "/mnt/d/codes/matlab/hyper-sara/data/y_N=256_L=15_p=1_snr=40.h5"; //
	std::string modelName = argc >= 2 ? argv[1] : "W28_N=256_L=15.fits"; // "/mnt/d/codes/matlab/hyper-sara/data/W28_N=256_L=15.fits"; //
    std::string temp_load_visibilities = argc >= 3 ? argv[2] : "true";
    visname = argc >= 4 ? argv[3] : visname;
	std::string name = argc >= 5 ? argv[4] : "puripsi_output";
	imsizex = argc >= 6 ? static_cast<t_int>(std::stod(static_cast<std::string>(argv[5]))) : imsizex;
	imsizey = argc >= 7 ? static_cast<t_int>(std::stod(static_cast<std::string>(argv[6]))) : imsizey;
	std::string temp_only_dirty = argc >= 8 ? argv[7] : "false";
	std::string temp_restoring = argc >= 9 ? argv[8] : "false";

	t_int image_size = imsizey*imsizex;

	const t_int ftsizeu = imsizex*over_sample;
	const t_int ftsizev = imsizey*over_sample;
	const t_real nshiftx = static_cast<psi::t_real>(imsizex)/2.;
	const t_real nshifty = static_cast<psi::t_real>(imsizey)/2.;
	std::string const fits_ending = ".fits";
	std::string const clean_outfile_fits = name + "_clean_";
	std::string const outfile_fits = name + "_";
	std::string const dirtyoutfile_fits = name +  "_dirty_";
	std::string const dirtyresidualoutfile_fits = name +  "_residual" + fits_ending;

	bool only_dirty = false;
	bool preconditioning = true;
	bool restoring =  false;
	bool wavelet_parallelisation = true;
    bool load_visibilities = false;

	t_int band_number;
	t_int row_number;
	t_int n_blocks = 1;

	if(argc > 10) {
		std::cout << "Usage:\n"
				"$ "
				<< argv[0] << " [uv] [model] [load] [output] [imsizex] [imsizey] [only_dirty] [restoring]\n\n"
                "- uv: name of input visibility file\n\n"
				"- model: name of the model file\n\n"
				"- load: loading integer (0 or 1) or string (true/True/false/False) specifying whether the code loads the noisy visibilities from a .h5 file and does not generate them"
				"- output: name of output file\n\n"
				"- imsizex: integer specifying the x size of the image\n\n"
				"- imsizey: integer specifying the y size of the image\n\n"
				"- only_dirty: integer (0 or 1) or string (true/True/false/False) specifying whether the code only produces a dirty image and does not simulate the true image"
				"- restoring: integer (0 or 1) or string (true/True/false/False) specifying whether a restart file is being used to initialise the simulation";
		exit(0);
	}

    if(temp_load_visibilities == "1" || temp_load_visibilities == "true" || temp_load_visibilities == "True"){
		load_visibilities = true;
	}else if(temp_load_visibilities == "0" || temp_load_visibilities == "false" || temp_load_visibilities == "False"){
		load_visibilities = false;
	}else{
		std::cout << "Incorrect load_visibilities parameter. Should be\n\n"
				"- load_visibilities: 1 or true for load_visibilities, 0 or false for not, false by default";
		exit(0);
	}

	if(temp_only_dirty == "1" || temp_only_dirty == "true" || temp_only_dirty == "True"){
		only_dirty = true;
	}else if(temp_only_dirty == "0" || temp_only_dirty == "false" || temp_only_dirty == "False"){
		only_dirty = false;
	}else{
		std::cout << "Incorrect only_dirty parameter. Should be\n\n"
				"- only_dirty: 1 or true for only_dirty, 0 or false for not, false by default";
		exit(0);
	}

	if(temp_restoring == "1" || temp_restoring == "true" || temp_restoring == "True"){
		restoring = true;
	}else if(temp_restoring == "0" || temp_restoring == "false" || temp_restoring == "False"){
		restoring = false;
	}else{
		std::cout << "Incorrect restoring parameter. Should be\n\n"
				"- restoring: 1 or true for restoring, 0 or false for not, false by default";
		exit(0);
	}

	bool mpi_init_status = psi::mpi::init(argc, argv);

	if(!mpi_init_status){

		PURIPSI_ERROR("Problem initialising MPI. Quitting.");
		return 0;

	}else{

		auto const world = psi::mpi::Communicator::World();

		bool parallel = true;

		auto Decomp = AstroDecomposition(parallel, world);

		auto adaptive_epsilon_start = 200;
		if(restoring){
			adaptive_epsilon_start = 0;
		}

		/* Set parameters for the solver */
		// Set SARA dictionary
		psi::wavelets::SARA const sara{
			std::make_tuple("Dirac", 3u), std::make_tuple("DB1", 3u), std::make_tuple("DB2", 3u),
					std::make_tuple("DB3", 3u),   std::make_tuple("DB4", 3u), std::make_tuple("DB5", 3u),
					std::make_tuple("DB6", 3u),   std::make_tuple("DB7", 3u), std::make_tuple("DB8", 3u)};
		auto const nlevels = sara.size();
		auto const min_delta = 1e-5;

        // Set number of channels and data blocks from .h5 file
        //! use collective read operation? (see how to do this)
        if(load_visibilities){
            if(Decomp.global_comm().is_root()){
                H5File file( visname, H5F_ACC_RDONLY );
                // extract number of blocks and channels from the size of the "epsilon" dataset
                DataSet dataset = file.openDataSet( "epsilon" );
                DataSpace dataspace = dataset.getSpace();
                int rank = dataspace.getSimpleExtentNdims();
                hsize_t dims_epsilon[rank];
                int ndims = dataspace.getSimpleExtentDims( dims_epsilon, NULL);
				PURIPSI_HIGH_LOG("l2_ball_epsilon rank: {}, dimensions: {}x{}", rank, (unsigned long)(dims_epsilon[0]), dims_epsilon[1]);
                band_number = dims_epsilon[1]; // 60
                n_blocks = dims_epsilon[0];    // 4
            }

            n_blocks = Decomp.global_comm().broadcast(n_blocks, Decomp.global_comm().root_id());
        }

		Image<t_complex> local_X0;

		//! Local scoping added to enable global_X0 to be removed after the data loading has happened and free up memory
		{
			Image<t_complex> global_X0;


			if(Decomp.global_comm().is_root()){
				global_X0 = pfitsio::read2d(modelName);   // model cube should be row-major [L, N], X0[N, L] after reading
                if(load_visibilities and (global_X0.cols() != band_number)){
                    PURIPSI_ERROR("Number of channels in the image cube is not the same as in the visibility file");
                }
                else{
				    band_number = global_X0.cols();
                }
				row_number = global_X0.rows();
				PURIPSI_HIGH_LOG("Number of channels is {} ", band_number);
				PURIPSI_HIGH_LOG("Image size from file is {} ", row_number);
				PURIPSI_HIGH_LOG("Image size {} x {} = {}", imsizex, imsizey, imsizex*imsizey);
			}



			band_number = Decomp.global_comm().broadcast(band_number, Decomp.global_comm().root_id());
			row_number = Decomp.global_comm().broadcast(row_number, Decomp.global_comm().root_id());

			if(row_number != imsizex*imsizey){
				PURIPSI_ERROR("Image size is not the same as in the model file");
			}

            //! modify from here (read number of data blocks from file, idem for number of bands)
			std::vector<t_int> time_blocks = std::vector<t_int>(band_number);
			for(int b=0; b<band_number; b++){
				time_blocks[b] = n_blocks;
			}

			std::vector<t_int> wavelet_levels = std::vector<t_int>(band_number);
			for(int b=0; b<band_number; b++){
				wavelet_levels[b] = nlevels;
			}

			std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(band_number);
			for(int b=0; b<band_number; b++){
				sub_blocks[b] = std::vector<t_int>(n_blocks);
				for(int t=0; t<n_blocks; t++){
					sub_blocks[b][t] = 0;
				}
			}

			Decomp.decompose_primal_dual(true, true, false, wavelet_parallelisation, true, band_number, wavelet_levels, time_blocks, sub_blocks, true);

			Decomp.set_checkpointing(true);
			Decomp.set_checkpointing_frequency(100);
			Decomp.set_restoring(restoring);

			local_X0 = Image<t_complex>(row_number, Decomp.my_number_of_frequencies());

			Decomp.template distribute_frequency_data<Image<t_complex>, t_complex>(local_X0, global_X0, false);

		}

		std::vector<t_uint> local_nlevels(Decomp.my_number_of_frequencies());

		std::vector<psi::LinearTransform<psi::Vector<psi::t_complex>>> Psi;
		Psi.reserve(Decomp.my_number_of_frequencies());
		// TODO properly fix this so there are multiple distributed_sara used
		// because at the moment it assumes all wavelets are distributed the same way
		auto distributed_sara = psi::wavelets::distribute_sara(sara, Decomp.my_frequencies()[0].lower_wavelet, Decomp.my_frequencies()[0].number_of_wavelets, Decomp.frequencies()[0].number_of_wavelets);
		for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
			PURIPSI_LOW_LOG("Distributing wavelets {} {} {}",Decomp.global_comm().rank(), Decomp.my_frequencies()[f].lower_wavelet, Decomp.my_frequencies()[f].number_of_wavelets);
			Psi.emplace_back(psi::linear_transform<psi::t_complex>(distributed_sara, imsizey, imsizex));
			local_nlevels[f] = Decomp.my_frequencies()[f].number_of_wavelets;
		}

		psi::LinearTransform<psi::Vector<psi::t_complex>> Psi_Root = psi::linear_transform_identity<t_complex>();
		auto distributed_root_sara = psi::wavelets::distribute_sara(sara, Decomp.my_lower_root_wavelet(), Decomp.my_number_of_root_wavelets(), Decomp.global_number_of_root_wavelets());
		if(Decomp.my_number_of_root_wavelets()>0){
			PURIPSI_LOW_LOG("Distributing root wavelets {} {} {}",Decomp.global_comm().rank(), Decomp.my_lower_root_wavelet(), Decomp.my_number_of_root_wavelets());
			Psi_Root = psi::linear_transform<psi::t_complex>(distributed_root_sara, imsizey, imsizex);
		}

		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			if(Decomp.own_this_frequency(Decomp.my_frequencies()[f].freq_number)){
				Image<t_complex> out_image = Image<t_complex>::Map(local_X0.col(f).data(), imsizex, imsizey);
				out_image.transposeInPlace();
				pfitsio::write2d(out_image.real(), clean_outfile_fits + std::to_string(Decomp.my_frequencies()[f].freq_number) + fits_ending);
			}
		}

		/* For the issue of storage, instead of reading data directly from an MS, hyperspectral data are generated from
		 * a realistic monochromatic uv file and a model hyperspectral image */
		// 1.wide-band frequency vector
        psi::Vector<t_real> freq = psi::Vector<t_real>::LinSpaced(band_number, 1e9, 2e9);

		std::vector<std::vector<utilities::vis_params>> uv_data(Decomp.my_number_of_frequencies());

		if(load_visibilities){

			std::vector<std::vector<psi::Image<psi::t_real>>> global_uv_model(band_number);
			if(Decomp.global_comm().is_root()){
				
				// 2.wide-band uv generated from a monochromatic realistic uv-coverage
				PSI_HIGH_LOG("Reading uv data");         
				H5File file( visname, H5F_ACC_RDONLY );
				// extract uv model
				for(hsize_t f = 0; f < band_number; f++){
					global_uv_model[f].reserve(n_blocks);
					for(hsize_t b = 0; b < n_blocks; b++){
						std::string datasetname = "uvw" + std::to_string(f*n_blocks+b+1);
						DataSet dataset = file.openDataSet( datasetname );
						DataSpace dataspace = dataset.getSpace();
						int rank = dataspace.getSimpleExtentNdims();
						hsize_t dims_uvw_block[rank];
						int ndims_block = dataspace.getSimpleExtentDims( dims_uvw_block, NULL);
						psi::Image<psi::t_real> temp_uvw = psi::Image<psi::t_real>::Zero(dims_uvw_block[1], dims_uvw_block[0]); //! inversion of the dimensions needed due to column major in Eigen, row major in HDF5, take results in rows afterwards
						// temp_uvw.transposeInPlace(); //! needed when reading from h5 file, read by row when reading from .fits file
						// need to comment out the tranpose, since the distribution from global_uv_model to uv_model acts row by row
						dataset.read(temp_uvw.data(), PredType::NATIVE_DOUBLE);
						global_uv_model[f].emplace_back(temp_uvw);
						dataspace.close();
						dataset.close(); 
					}
				}
				file.close();

				PURIPSI_HIGH_LOG("Size global_uv_model[0][0]: {} x {}",  global_uv_model[0][0].rows(), global_uv_model[0][0].cols());
				PURIPSI_HIGH_LOG("global_uv_model[0][0]: u = {}",  global_uv_model[0][0](0));
				PURIPSI_HIGH_LOG("global_uv_model[0][0]: v = {}",  global_uv_model[0][0](1));
				PURIPSI_HIGH_LOG("global_uv_model[0][0]: w = {}",  global_uv_model[0][0](2));
			}

			std::vector<std::vector<psi::Image<t_real>>> uv_model(Decomp.my_number_of_frequencies());
			for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
				uv_model[f] = std::vector<psi::Image<t_real>>(Decomp.my_frequencies()[f].number_of_time_blocks);
			}
			// distribute global_uv_model into uv_model
			Decomp.template distribute_uv_data<std::vector<std::vector<psi::Image<t_real>>>, psi::Image<t_real>>(global_uv_model, uv_model); //! problem transmission here!! (format)

			for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
				uv_data[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
				for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){
					utilities::vis_params vis_tmp;
					// uv_model[f][t] = (uv_model[f][t] * freq[Decomp.my_frequencies()[f].freq_number]/freq[0]).real().eval();    // scaling of uv-coverage in terms of frequency
					vis_tmp.u = uv_model[f][t].row(0);
					vis_tmp.v = uv_model[f][t].row(1);
					vis_tmp.w = uv_model[f][t].row(2);
					vis_tmp.weights = Vector<t_complex>::Constant(uv_model[f][t].row(0).size(), 1);
					vis_tmp.vis = Vector<t_complex>::Constant(uv_model[f][t].row(0).size(), 1);
					uv_data[f].emplace_back(vis_tmp);
					uv_data[f][t].units = puripsi::utilities::vis_units::radians;
				}
			}

		} else {
			//! generate and save synth coverage
			t_real const sigma_m = constant::pi / 3;
			t_int const number_of_vis = std::floor(image_size * 2.);
			t_int const number_of_vis_per_block = number_of_vis/n_blocks;

			// generate random coverage in parallel
			for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
				uv_data[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
				// utilities::vis_params vis_tmp = utilities::random_sample_density(number_of_vis_per_block, 0, sigma_m, 0); // use in testing mode, set rng to 1000
				for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){
					utilities::vis_params vis_tmp2 = utilities::random_sample_density(number_of_vis, 0, sigma_m, 0);
					vis_tmp2.units = puripsi::utilities::vis_units::radians;
					uv_data[f].emplace_back(vis_tmp2);
				}
			} 
		}	

		// //! AJ Temporary until we calculate here
		// pixel_size = 0.64;

		// 3.Compute the preconditioning matrix
		std::vector<std::vector<psi::Vector<t_real>>> Ui(Decomp.my_number_of_frequencies());
		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			Ui[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
			for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
				Ui[f].emplace_back(psi::Vector<psi::t_real>::Ones(uv_data[f][t].u.size()));
				puripsi::preconditioner<t_real>(Ui[f][t], uv_data[f][t].u, uv_data[f][t].v, ftsizev, ftsizeu);
			}
		}


		t_real nu2;
		if(not restoring){
			if(load_visibilities){
				if(Decomp.global_comm().is_root()){
				H5File file( visname, H5F_ACC_RDONLY );
				DataSet dataset = file.openDataSet( "operator_norm" );
				dataset.read(&nu2, PredType::NATIVE_DOUBLE);
				dataset.close();
				file.close();
				PURIPSI_HIGH_LOG("nu2 is {} ", nu2);
				}
			} else {
				std::vector<std::vector<std::shared_ptr<const psi::LinearTransform<psi::Vector<psi::t_complex>>>>> Phi2(Decomp.my_number_of_frequencies());
				for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
					Phi2[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
					for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
						Phi2[f].emplace_back(std::make_shared<const MeasurementOperator>(uv_data[f][t], Ui[f][t], J, J, kernel, imsizex, imsizey, 100, over_sample, pixel_size, pixel_size, "none", 0, false, 1, "none", false, nshiftx, nshifty));
					}
				}

				// Compute global operator norm
				auto const pm = psi::algorithm::PowerMethodWideband<psi::t_complex>().tolerance(1e-6).decomp(Decomp);
				auto const result = pm.AtA(Phi2, psi::Matrix<psi::t_complex>::Random(imsizey*imsizex, Decomp.my_number_of_frequencies()));
				nu2 = result.magnitude.real();
				//	nu2 = 1638468.8841572138;

				// Manually delete the Phi2 measurement operator to reduce memory here (it doesn't seem to be free'd quick enough to let the second set of measurement
				// operators get built successfully below in large image size cases.
				for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
					for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
						Phi2[f][t].reset();
					}
				}
			}
			nu2 = Decomp.global_comm().broadcast(nu2, Decomp.global_comm().root_id());
		}

		// 4.Generate measurement operators from the available uv_data
		std::vector<std::vector<std::shared_ptr<const psi::LinearTransform<psi::Vector<psi::t_complex>>>>> Phi(Decomp.my_number_of_frequencies());
		for(int f=0; f< Decomp.my_number_of_frequencies(); ++f){
			Phi[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
			for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
				Phi[f].emplace_back(std::make_shared<const MeasurementOperator>(uv_data[f][t], J, J, kernel, imsizex, imsizey, 100, over_sample, pixel_size, pixel_size, "none", 0, false, 1, "none", false, nshiftx, nshifty));
			}
		}

        // 5.Generate the ground truth measurements y0, or load visibilities
		std::vector<std::vector<psi::Vector<t_complex>>> target(Decomp.my_number_of_frequencies());
        psi::Vector<psi::Vector<t_real>> l2ball_epsilon(Decomp.my_number_of_frequencies());

		//! use collective read operation?
        if(load_visibilities){
			
			CompType complex_data_type(sizeof(complex_type));
			complex_data_type.insertMember( "real", 0, PredType::NATIVE_DOUBLE);
			complex_data_type.insertMember( "imag", sizeof(double), PredType::NATIVE_DOUBLE);
			std::vector<std::vector<psi::Vector<t_complex>>> global_target(band_number);
			psi::Vector<psi::Vector<t_real>> global_l2ball_epsilon(band_number);

			for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
				target[f] = std::vector<Vector<t_complex>>(Decomp.my_frequencies()[f].number_of_time_blocks);
				l2ball_epsilon[f] = psi::Vector<t_real>::Zero(Decomp.my_frequencies()[f].number_of_time_blocks);
			}

			if(Decomp.global_comm().is_root()){

				H5File file( visname, H5F_ACC_RDONLY );
				
				// extract visibilities target[f][b]
				for(hsize_t f=0; f<band_number; f++){            
					global_target[f].reserve(n_blocks);
					for (hsize_t b=0; b<n_blocks; b++){
						std::string datasetname = "y" + std::to_string(f*n_blocks+b+1);
						DataSet dataset = file.openDataSet( datasetname );
						DataSpace dataspace = dataset.getSpace();
						int rank = dataspace.getSimpleExtentNdims();
						hsize_t dims_target[rank];
						int ndims_target = dataspace.getSimpleExtentDims( dims_target, NULL);

						psi::Vector<t_complex> y = psi::Vector<t_complex>::Zero(dims_target[0]);
						dataset.read(y.data(), complex_data_type);
						global_target[f].emplace_back(y);
						dataspace.close();
						dataset.close();
					}  
				}

				// extract number of blocks and channels from the size of the "epsilon" dataset
				DataSet dataset = file.openDataSet( "epsilon" );
				DataSpace dataspace = dataset.getSpace();

				for(hsize_t f = 0; f < band_number; f++){
					global_l2ball_epsilon[f] = psi::Vector<psi::t_real>::Zero(n_blocks);
					
					hsize_t col_dims[1] = {(hsize_t)n_blocks};
					DataSpace memspace(1, col_dims);
					hsize_t offset_out[2] = {0, f};       // hyperslab offset in memory
					hsize_t count_out[2] = {(hsize_t)n_blocks, 1}; // size of the hyperslab in memory
					dataspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out );
					dataset.read( global_l2ball_epsilon[f].data(), PredType::NATIVE_DOUBLE, memspace, dataspace );
					memspace.close(); // not necessary, memsapce released once out of scope
				}
				dataspace.close();
				dataset.close();

				PURIPSI_HIGH_LOG("Loaded target and epsilon");
				
			}
			// Distributed global_l2ball_epsilon
			Decomp.template distribute_epsilons_wideband_blocking<psi::Vector<psi::Vector<t_real>>>(l2ball_epsilon, global_l2ball_epsilon);
			// Distribute global_target
			Decomp.template distribute_target_data<std::vector<std::vector<psi::Vector<t_complex>>>, psi::Vector<t_complex>>(global_target, target);

        }
        else{
            std::vector<std::vector<psi::Vector<t_complex>>> y0(Decomp.my_number_of_frequencies());
            t_real normy0 = 0.;
            t_int Nm = 0; // total number of measurements
            for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
                y0[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
                for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
                    auto tmp = (*Phi[f][t]) * local_X0.col(f);
                    y0[f].emplace_back(tmp);
                    normy0 += y0[f][t].squaredNorm();
                    Nm += y0[f][t].size();
                }
            }

            if(Decomp.global_comm().is_root()){
                PURIPSI_HIGH_LOG("Constructed y0");
				PURIPSI_HIGH_LOG("y0[{}][{}] = {}", 0, 0, y0[0][0]);
            }

            // TODO: Does this need to be globally reduced?
            auto sigma_noise = std::sqrt(normy0) / std::sqrt(Nm) * std::pow(10.0, -(input_snr / 20.0));

            // 6.Add noise to the measurements
            for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
                target[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
                for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
                    // The *true* in the add noise below ensures that the same seed for the random number generator is always used. This should
                    // make benchmarking more consistent but should *never* be used for production simulations.
                    uv_data[f][t].vis = utilities::add_noise(y0[f][t], 0., sigma_noise, true);
                    target[f].emplace_back(uv_data[f][t].vis);
                }
            }

            if(Decomp.global_comm().is_root()){
                PURIPSI_HIGH_LOG("Constructed target");
            }

            for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
                l2ball_epsilon[f] = psi::Vector<t_real>::Zero(Decomp.my_frequencies()[f].number_of_time_blocks);
            }

			// Compute global epsilon bound (from Abdullah's code)
			auto global_epsilon = std::sqrt(Nm + 2*std::sqrt(2*Nm)) * sigma_noise;
			// Compute epsilon for each spectral band (from Abdullah's code)
			for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
				for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
					l2ball_epsilon[f](t) = std::sqrt(static_cast<t_real>(target[f][t].size())/static_cast<t_real>(Nm)) * global_epsilon;
				}
			}
        }

		if(Decomp.global_comm().is_root()){
			PURIPSI_HIGH_LOG("Calculated epsilon");
		}


		// Algorithm parameters
		auto const tau = 0.99/3.; // 3 terms involved.
		if(Decomp.global_comm().is_root()){
			PURIPSI_HIGH_LOG("tau is {} ", tau);
		}

		auto kappa1 = 1.0; // spectral norm of the identity operator
		if(Decomp.global_comm().is_root()){
			PURIPSI_HIGH_LOG("kappa1 is {} ", kappa1);
		}

		auto kappa2 = 1.0; // Daubechies wavelets are orthogonal, and only the identity is considered in addition to these dictionaries -> operator norm equal to 1
		if(Decomp.global_comm().is_root()){
			PURIPSI_HIGH_LOG("kappa2 is {} ", kappa2);
		}

		t_real kappa3;
		if(not restoring){
			kappa3 = 1./nu2; // inverse of the norm of the full measurement operator Phi (single value)
			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("kappa3 is {} ", kappa3);
			}
		}

		psi::Vector<t_real> eps_lambdas(3);
		eps_lambdas << 0.99, 1.01, 0.5*(sqrt(5)-1);
		if(Decomp.global_comm().is_root()){
			PURIPSI_HIGH_LOG("adaptive epsilon parameters: tol_in {}, tol_out {}, percentage {} ", eps_lambdas(0), eps_lambdas(1), eps_lambdas(2));
		}


		std::vector<Vector<t_complex>> dirty(Decomp.my_number_of_frequencies());
		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			dirty[f] = Vector<t_complex>::Zero(imsizey*imsizex);
		}
		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){
				dirty[f] = dirty[f] +  (Phi[f][t]->adjoint() * uv_data[f][t].vis);
			}
			Decomp.my_frequencies()[f].freq_comm.distributed_sum(dirty[f], Decomp.my_frequencies()[f].freq_comm.root_id());
		}
		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			if(Decomp.own_this_frequency(Decomp.my_frequencies()[f].freq_number)){
				Image<t_complex> dirty_image = Image<t_complex>::Map(dirty[f].data(), imsizex, imsizey);
				dirty_image.transposeInPlace();
				pfitsio::write2d(dirty_image.real(), dirtyoutfile_fits + std::to_string(Decomp.my_frequencies()[f].freq_number) + fits_ending);
			}
		}


		if(not only_dirty){


			psi::mpi::Scalapack scalapack = psi::mpi::Scalapack(true);
			scalapack.setupBlacs(Decomp, std::min((int)std::min(imsizey*imsizex, band_number),(int)std::min((int)60,  (int)Decomp.global_comm().size())), imsizey*imsizex, Decomp.global_number_of_frequencies()); // triggers warning/error message when setup not run already

	                if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("Using {} processes for the SVD",std::min((int)std::min(imsizey*imsizex, band_number),(int)std::min((int)60,  (int)Decomp.global_comm().size())));
			}

			// Compute mu (nuclear_norm / l21_norm) in parallel
			//TODO: avoid using global_dirty_image_matrix (there should be a more efficient way to do this operation)
			t_real mu = 0.;
			{
				std::vector<Vector<t_complex>> global_dirty;
				Matrix<t_complex> global_dirty_matrix;

				if(Decomp.global_comm().is_root()){
					global_dirty = std::vector<Vector<t_complex>>(Decomp.global_number_of_frequencies());
					global_dirty_matrix = Matrix<t_complex>(image_size, band_number);
					for(int f=0; f<Decomp.global_number_of_frequencies(); ++f){
						global_dirty[f] = Vector<t_complex>::Zero(imsizey*imsizex);
					}
				}

				Decomp.collect_dirty_image(dirty, global_dirty);


				if(Decomp.global_comm().is_root()){
					for(int f=0; f<band_number; ++f){
						global_dirty_matrix.col(f) = global_dirty[f];
					}
				}

				Matrix<t_complex> partial;
				t_real nuclear_norm;
				t_real l21_norm;
				int mpa;
				int npa;
				int mpu;
				int npu;
				int mpvt;
				int npvt;
				psi::Vector<t_real> A;
				psi::Vector<t_real> U;
				psi::Vector<t_real> VT;
				psi::Vector<t_real> data_svd;
				psi::Vector<t_real> sigma = psi::Vector<t_real>(std::min(image_size, band_number));

				if(scalapack.involvedInSVD()){
					// set protected variables for scalapack
					mpa = scalapack.getmpa();
					npa = scalapack.getnpa();
					mpu = scalapack.getmpu();
					npu = scalapack.getnpu();
					mpvt = scalapack.getmpvt();
					npvt = scalapack.getnpvt();
					A = psi::Vector<t_real>(mpa*npa);
					U = psi::Vector<t_real>(mpu*npu);
					VT = psi::Vector<t_real>(mpvt*npvt);
				}

				if(Decomp.global_comm().is_root()) {
					data_svd = psi::Vector<t_real>(image_size*band_number);
				}

				if(scalapack.involvedInSVD()){
					scalapack.setupSVD(A, sigma, U, VT);
				}

				if(!Decomp.parallel_mpi() or not scalapack.usingScalapack()){
					if(Decomp.global_comm().is_root())
					{
						PURIPSI_LOW_LOG("Compute nuclear norm in serial");
						nuclear_norm = psi::nuclear_norm(global_dirty_matrix);
					}
				} else {
					if(Decomp.global_comm().is_root()){
						PURIPSI_LOW_LOG("Compute nuclear norm in parallel");
						for(int l=0; l<band_number ; ++l){
							for(int n=0; n<image_size; ++n){
								data_svd[l*image_size+n] = real(global_dirty_matrix(n,l)); //global_dirty[l](n)
							}
						}
					}

					if(scalapack.involvedInSVD()){ //! already checked in the functions, so not needed here
						scalapack.scatter(Decomp, A, data_svd, image_size, band_number, mpa, npa);
						scalapack.runSVD(A, sigma, U, VT);
					}

					if (Decomp.global_comm().is_root()){
						nuclear_norm = psi::l1_norm(sigma);
						PURIPSI_LOW_LOG("Parallel nuclear norm: {}", nuclear_norm);
					}
				}

				// Comptute l21 norm in parallel
				auto wavelet_regularization = [](psi::LinearTransform<Vector<t_complex>> psi, const Matrix<t_complex> &X, const t_uint rows) {
					Matrix<t_complex> Y(rows, X.cols());
#ifdef PSI_OPENMP
#pragma omp parallel for default(shared)
#endif
					for (int l = 0; l < X.cols(); ++l){
						Y.col(l) = static_cast<Vector<t_complex>>(psi.adjoint() * X.col(l));
					}
					return Y;
				};

				if(!Decomp.parallel_mpi() or Decomp.my_number_of_root_wavelets() != 0){
					Matrix<t_complex> local_dirty;
					if(Decomp.my_root_wavelet_comm().size() != 1){
						local_dirty = Decomp.my_root_wavelet_comm().broadcast(global_dirty_matrix, Decomp.global_comm().root_id());
						partial = wavelet_regularization(Psi_Root, local_dirty, image_size*Decomp.my_number_of_root_wavelets());
					}else{
						partial = wavelet_regularization(Psi_Root, global_dirty_matrix, image_size*Decomp.my_number_of_root_wavelets());
					}
					l21_norm = psi::l21_norm(partial);

					if(Decomp.parallel_mpi() and Decomp.my_root_wavelet_comm().size() != 1){
						Decomp.my_root_wavelet_comm().distributed_sum(&l21_norm, Decomp.global_comm().root_id());
					}
				}

				if(Decomp.global_comm().is_root()){
					mu = nuclear_norm/l21_norm; //! need to make sure this value is available everywhere (just in case)
				}
			}

			mu = Decomp.global_comm().broadcast(mu, Decomp.global_comm().root_id());

			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("mu = {}", mu);
			}

			//AJ Hack for matlab comparison
			mu = 1e-2;

			// Instantiate algorithm
			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("Creating wideband primal-dual functor");
			}

			auto ppd = psi::algorithm::PrimalDualWidebandBlocking<t_complex>(target, imsizey*imsizex, l2ball_epsilon, Phi, Ui)
																					.itermax(2000)
																					.itermin(300)
																					.mu(mu)
																					.tau(tau)
																					.kappa1(kappa1)
																					.kappa2(kappa2)
																					.kappa3(kappa3)
																					.Psi(Psi)
																					.Psi_Root(Psi_Root)
																					.levels(local_nlevels)
																					.global_levels(nlevels)
																					.n_channels(Decomp.global_number_of_frequencies())
																					.l21_proximal_weights(psi::Vector<t_real>::Ones(imsizex*imsizey*Decomp.my_number_of_root_wavelets()))
																					.nuclear_proximal_weights(psi::Vector<t_real>::Ones(Decomp.global_number_of_frequencies()))
																					.positivity_constraint(true)
																					.relative_variation(5e-4)
																					.residual_convergence(1.001)
																					.update_epsilon(false)
																					.relative_variation_x(5e-4)
																					.lambdas(eps_lambdas)
																					.P(100)
																					.decomp(Decomp)
																					.adaptive_epsilon_start(adaptive_epsilon_start)
																					.itermax_fb(20)
																					.preconditioning(true)
																					.relative_variation_fb(1e-8)
																					.objective_check_frequency(1)
																					.scalapack(scalapack);

			// Sets weight after each pd iteration.
			//PURIPSI_HIGH_LOG("Creating reweighting-scheme functor");
			auto reweighted = psi::algorithm::reweighted(ppd)
			.itermax(3)
			.min_delta(min_delta)
			.update_delta([](t_real delta) { return 0.5 * delta; })
			.is_converged(psi::RelativeVariation<psi::t_complex>(1e-4));

			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("Starting re-weighted primal dual from psi library");
			}
			std::clock_t c_start = std::clock();
			auto diagnostic = reweighted();
			std::clock_t c_end = std::clock();

			auto total_time = (c_end - c_start) / CLOCKS_PER_SEC;

			if(Decomp.global_comm().is_root()){
				if(not diagnostic.good){
					PURIPSI_HIGH_LOG("re-weighted primal dual did not converge in {} iterations", diagnostic.niters);
				}else{
					PURIPSI_HIGH_LOG("re-weighted primal dual returned in {} iterations", diagnostic.niters);
				}
				PURIPSI_HIGH_LOG("Total computing time: {}", total_time);

				// Write estimated image to a .fits file
				//assert(diagnostic.x.size() == band_number*imsizey*imsizex);
				//Image<t_complex> image_save = Image<t_complex>::Map(diagnostic.x.data(), imsizey*imsizex, band_number);
				//pfitsio::write2d(image_save.real(), outfile_fits);
				for(int f=0; f<Decomp.global_number_of_frequencies(); ++f){
					Image<t_complex> out_image = Image<t_complex>::Map(diagnostic.algo.x.col(f).data(), imsizex, imsizey);
					out_image.transposeInPlace();
					pfitsio::write2d(out_image.real(), outfile_fits + std::to_string(f) + fits_ending);
				}
			}
		}

        // MPI_Info_free(&info);
		psi::mpi::finalize();

	}

	return 0;
}
