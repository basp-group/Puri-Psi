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

#include <mpi.h>

#include <unsupported/Eigen/SparseExtra> // for Eigen::saveMarket
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
#include <psi/power_method.h>
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
#include "puripsi/astroio.h"
#include "puripsi/time_blocking.h"

using namespace puripsi;
using namespace puripsi::notinstalled;

int main(int argc, const char **argv) {
	psi::logging::initialize();
	puripsi::logging::initialize();
	psi::logging::set_level("critical");
	puripsi::logging::set_level("critical");

	// Image parameters
	t_int imsizey = 2048;
	t_int imsizex = 2048;
	auto const input_snr = 40.; // in dB

	// Gridding parameters
	t_int const J = 8;
	t_real const over_sample = 2;
	const string kernel = "kb";
	t_real dl = 2.5059;
	t_real pixel_size = 0.04;

	std::vector<std::vector<std::string>> dataName(2);
	dataName[0] = std::vector<std::string>(2);
	dataName[1] = std::vector<std::string>(2);
	dataName[0][0] = "/home/nx01/nx01/adrianj/compress/puri-psi-dev/build/cpp/example/wideband_ms/CYG-A-5-8M10S.MS";
	dataName[0][1] = "/home/nx01/nx01/adrianj/compress/puri-psi-dev/build/cpp/example/wideband_ms/CYG-C-5-8M10S.MS";
	dataName[1][0] = "/home/nx01/nx01/adrianj/compress/puri-psi-dev/build/cpp/example/wideband_ms/CYG-A-7-8M10S.MS";
	dataName[1][1] = "/home/nx01/nx01/adrianj/compress/puri-psi-dev/build/cpp/example/wideband_ms/CYG-C-7-8M10S.MS";

	t_int n_datasets = dataName.size();

	std::string name = argc >= 2 ? argv[1] : "puripsi_output";
	imsizex = argc >= 3 ? static_cast<t_int>(std::stod(static_cast<std::string>(argv[2]))) : imsizex;
	imsizey = argc >= 4 ? static_cast<t_int>(std::stod(static_cast<std::string>(argv[3]))) : imsizey;
	std::string temp_only_dirty = argc >= 5 ? argv[4] : "false";
	std::string temp_restoring = argc >= 6 ? argv[5] : "false";
	std::string temp_restoring_uv_data = argc >= 7 ? argv[6] : "false";
	std::string uv_data_filename = argc >= 8 ? argv[7] : "uv_data.dat";

	const t_int ftsizeu = imsizex*over_sample;
	const t_int ftsizev = imsizey*over_sample;
	std::string const fits_ending = ".fits";
	std::string const clean_outfile_fits = name + "_clean_";
	std::string const outfile_fits = name + "_";
	std::string const dirtyoutfile_fits = name +  "_dirty_";
	std::string const dirtyresidualoutfile_fits = name +  "_residual" + fits_ending;

	bool only_dirty = false;
	bool preconditioning = true;
	bool restoring =  false;
	bool restoring_uv_data = false;
	bool wavelet_parallelisation = true;

	t_int band_number = 0;
	t_int row_number;
	t_int n_blocks = 0;
	t_int n_measurements = 0;
	t_int field_id = 2;

	if(argc > 8) {
		std::cout << "Usage:\n"
				"$ "
				<< argv[0] <<
				"- output: name of output file\n\n"
				"- imsizex: integer specifying the x size of the image\n\n"
				"- imsizey: integer specifying the y size of the image\n\n"
				"- only_dirty: integer (0 or 1) or string (true/True/false/False) specifying whether the code only produces a dirty image and does not simulate the true image"
				"- restoring: integer (0 or 1) or string (true/True/false/False) specifying whether a restart file is being used to initialise the simulation";
		"- restoring_uv_data: integer (0 or 1) or string (true/True/false/False) specifying whether a uv_data file is being used to initialise the simulation";
		"- uv_data: name of the uv_data file if one is being used\n\n";
		exit(0);
	}

	only_dirty = utilities::parse_true_false_parameter(temp_only_dirty, "only_dirty");
	restoring = utilities::parse_true_false_parameter(temp_restoring, "restoring");
	restoring_uv_data = utilities::parse_true_false_parameter(temp_restoring_uv_data, "restoring_uv_data");

	bool mpi_init_status = psi::mpi::init(argc, argv);

	if(!mpi_init_status){

		PURIPSI_ERROR("Problem initialising MPI. Quitting.");
		return 0;

	}else{

		auto const world = psi::mpi::Communicator::World();

		bool parallel = true;

		auto Decomp = AstroDecomposition(parallel, world);

		/* Set parameters for the solver */
		// Set SARA dictionary
		psi::wavelets::SARA const sara{
			std::make_tuple("Dirac", 3u), std::make_tuple("DB1", 3u), std::make_tuple("DB2", 3u),
					std::make_tuple("DB3", 3u),   std::make_tuple("DB4", 3u), std::make_tuple("DB5", 3u),
					std::make_tuple("DB6", 3u),   std::make_tuple("DB7", 3u), std::make_tuple("DB8", 3u)};
		auto const nlevels = sara.size();
		auto const min_delta = 1e-5;


		if(Decomp.global_comm().is_root()){
			PURIPSI_HIGH_LOG("Only create the dirty image {}",only_dirty);
			PURIPSI_HIGH_LOG("Restoring from checkpoint {}",restoring);
			PURIPSI_HIGH_LOG("Restoring uv_data from file {}",restoring_uv_data);
		}

		std::vector<std::vector<utilities::vis_params>> uv_data;
		t_real field_of_view;
		//! Encapsulate below in brackets to make global_uv_data go out of scope when we have finished with it and
		//! thereby free up memory.
		{
			std::vector<std::vector<puripsi::utilities::vis_params>> global_uv_data;

			Vector<t_int> frequencies_per_dataset(n_datasets);
			Vector<t_int> n_blocks_per_dataset(n_datasets);
			Vector<t_int> blocks_per_frequency;
			Vector<std::set<::casacore::uInt>> spws_ids(n_datasets);

			if(not restoring_uv_data){

				if(Decomp.global_comm().is_root()){
					for(int i=0; i<n_datasets; i++){
						extract_number_of_channels_and_spectal_windows(dataName[i][0], field_id, frequencies_per_dataset[i], spws_ids[i]);
						band_number += frequencies_per_dataset[i];
						PURIPSI_HIGH_LOG("Number of Spectral Windows {}",spws_ids[i].size());
						for(auto f : spws_ids[i]) {
							PURIPSI_HIGH_LOG("Spectral Windows {}",f);
						}
					}
					blocks_per_frequency = Vector<t_int>(band_number);

					global_uv_data = std::vector<std::vector<puripsi::utilities::vis_params>>(band_number);
					int freq_number = 0;
					for(int i=0; i<n_datasets; i++){
						PURIPSI_HIGH_LOG("Reading in file {} {}",dataName[i][0], dataName[i][1]);
						for(auto f : spws_ids[i]) {
							PURIPSI_HIGH_LOG("Calculated number of frequencies {}",frequencies_per_dataset[i]/spws_ids[i].size());
							for(int j=0; j<frequencies_per_dataset[i]/spws_ids[i].size(); j++){
								PURIPSI_HIGH_LOG("Spectral Window {} Frequency {}",f,j);
								std::vector<std::string> temp_file(1);
								global_uv_data[freq_number] = get_time_blocks_multi_file_spectral_window(dataName[i], &dl, &pixel_size, &n_blocks, &n_measurements, f, j, n_blocks_per_dataset, field_id);
								PURIPSI_HIGH_LOG("Final number of blocks for channel {} is {} and measurements is {}",freq_number,n_blocks,n_measurements);
								blocks_per_frequency[freq_number] = n_blocks;
								freq_number++;
							}
						}
					}
					PURIPSI_HIGH_LOG("Number of channels is {} ", band_number);
					for(int f=0; f<band_number; f++){
						PURIPSI_HIGH_LOG("Channel[{}] has {} blocks", f, blocks_per_frequency[f]);
					}
				}
			}else{
				if(Decomp.global_comm().is_root()){
					auto io = astroio::AstroIO();

					psi::io::IOStatus io_status = io.load_uv_data_header(band_number, blocks_per_frequency, dl, pixel_size, uv_data_filename);
					if(io_status != psi::io::IOStatus::Success){
						PSI_ERROR("Problem reading uv_data header");
					}

					global_uv_data = std::vector<std::vector<puripsi::utilities::vis_params>>(band_number);

					for(int f=0; f<band_number; f++){
						global_uv_data[f] = std::vector<utilities::vis_params>(blocks_per_frequency[f]);
					}

					io_status = io.load_uv_data(global_uv_data, uv_data_filename);
					if(io_status != psi::io::IOStatus::Success){
						PSI_ERROR("Problem reading uv_data");
					}
				}
			}

			Decomp.distribute_parameters_int_real(&band_number, &pixel_size);
			Decomp.distribute_parameters_wideband(blocks_per_frequency);

			field_of_view = pixel_size*imsizex;

			std::vector<t_int> time_blocks = std::vector<t_int>(band_number);
			for(int b=0; b<band_number; b++){
				time_blocks[b] = blocks_per_frequency[b];
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

			Decomp.decompose_primal_dual(true, true, false, wavelet_parallelisation, true, band_number, wavelet_levels, time_blocks, sub_blocks, false);

			Decomp.set_checkpointing(true);
			Decomp.set_checkpointing_frequency(100);
			Decomp.set_restoring(restoring);

			uv_data = std::vector<std::vector<utilities::vis_params>> (Decomp.my_number_of_frequencies());
			for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
				uv_data[f] = std::vector<puripsi::utilities::vis_params>(Decomp.my_frequencies()[f].number_of_time_blocks);
			}

			Decomp.distribute_uv_data(global_uv_data, uv_data);
		}

		std::vector<t_uint> local_nlevels(Decomp.my_number_of_frequencies());

		std::vector<psi::LinearTransform<psi::Vector<psi::t_complex>>> Psi;
		Psi.reserve(Decomp.my_number_of_frequencies());
		// TODO properly fix this so there are multiple distributed_sara used
		auto distributed_sara = psi::wavelets::distribute_sara(sara, Decomp.my_frequencies()[0].lower_wavelet, Decomp.my_frequencies()[0].number_of_wavelets, Decomp.frequencies()[0].number_of_wavelets);
		for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
			PURIPSI_LOW_LOG("Distributing wavelets {} {} {}",Decomp.global_comm().rank(), Decomp.my_frequencies()[f].lower_wavelet, Decomp.my_frequencies()[f].number_of_wavelets);
			Psi.emplace_back(psi::linear_transform<psi::t_complex>(distributed_sara, imsizey, imsizex));
			local_nlevels[f] = Decomp.my_frequencies()[f].number_of_wavelets;
		}

		psi::LinearTransform<psi::Vector<psi::t_complex>> Psi_Root = psi::linear_transform_identity<t_complex>();
		if(Decomp.my_number_of_root_wavelets()>0){
			PURIPSI_HIGH_LOG("Distributing root wavelets {} {} {}",Decomp.global_comm().rank(), Decomp.my_lower_root_wavelet(), Decomp.my_number_of_root_wavelets());
			Psi_Root = psi::linear_transform<psi::t_complex>(psi::wavelets::distribute_sara(sara, Decomp.my_lower_root_wavelet(), Decomp.my_number_of_root_wavelets(), Decomp.global_number_of_root_wavelets()), imsizey, imsizex);
		}


		std::vector<std::vector<psi::Vector<psi::t_complex>>> target(Decomp.my_number_of_frequencies());
		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			target[f] = std::vector<psi::Vector<psi::t_complex>>(Decomp.my_frequencies()[f].number_of_time_blocks);
		}
		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){
				target[f][t] = uv_data[f][t].vis.array() * uv_data[f][t].weights.array();
			}
		}


		std::vector<std::vector<psi::Vector<t_real>>> Ui;
		if(preconditioning){
			Ui = std::vector<std::vector<psi::Vector<t_real>>>(Decomp.my_number_of_frequencies());
			for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
				Ui[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
				for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
					Ui[f].emplace_back(psi::Vector<psi::t_real>::Ones(uv_data[f][t].u.size()));
					puripsi::preconditioner<t_real>(Ui[f][t], uv_data[f][t].u, uv_data[f][t].v, ftsizev, ftsizeu);
				}
			}
		}

		t_real nu2 = 1.0;
		if(not restoring){
			std::vector<std::vector<std::shared_ptr<const psi::LinearTransform<psi::Vector<psi::t_complex>>>>> Phi2(Decomp.my_number_of_frequencies());
			for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
				Phi2[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
				for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
					if(preconditioning){
						Phi2[f].emplace_back(std::make_shared<const MeasurementOperator>(uv_data[f][t], Ui[f][t], J, J, kernel, imsizex, imsizey, 100, over_sample, pixel_size, pixel_size, "natural", 0, "false", 1, "none", true));
					}else{
						Phi2[f].emplace_back(std::make_shared<const MeasurementOperator>(uv_data[f][t], J, J, kernel, imsizex, imsizey, 100, over_sample, pixel_size, pixel_size, "natural", 0, "false", 1, "none", true));
					}
				}
			}
			// Compute global operator norm
			auto const pm = psi::algorithm::PowerMethodWideband<psi::t_complex>().tolerance(1e-6).decomp(Decomp);
			auto const result = pm.AtA(Phi2, psi::Matrix<psi::t_complex>::Random(imsizey*imsizex, Decomp.my_number_of_frequencies()));
			PURIPSI_HIGH_LOG("Calculated power method {}",Decomp.global_comm().rank());
			nu2 = result.magnitude.real();
			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("nu2 is {} ", nu2);
			}
		}

		// 4.Generate measurement operators from the available uv_data
		std::vector<std::vector<std::shared_ptr<const psi::LinearTransform<psi::Vector<psi::t_complex>>>>> Phi(Decomp.my_number_of_frequencies());
		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			Phi[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
			for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
				// No preconditioning (normal operator)
				Phi[f].emplace_back(std::make_shared<const MeasurementOperator>(uv_data[f][t], J, J, kernel, imsizex, imsizey, 100, over_sample, pixel_size, pixel_size, "natural", 0, "false", 1, "none", true));
			}
		}

		psi::Vector<psi::Vector<t_real>> l2ball_epsilon(Decomp.my_number_of_frequencies());

		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			l2ball_epsilon[f] = psi::Vector<t_real>::Zero(Decomp.my_frequencies()[f].number_of_time_blocks);
			for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){
				if(uv_data[f][t].u.size() == 0){
					PURIPSI_HIGH_LOG("Frequency {} Block {} on rank {} is zero in size",f,t,Decomp.global_comm().rank());
				}

				//! If we are reading in a checkpoint from file we will read in the epsilon rather than calculate it.
				if(!restoring){
					auto pixel_size_config = field_of_view / imsizey;
					std::shared_ptr<const psi::LinearTransform<psi::Vector<psi::t_complex>>> Phi_nnls = std::make_shared<const MeasurementOperator>(uv_data[f][t], J, J, kernel, imsizex, imsizey, 100, over_sample, pixel_size_config, pixel_size_config, "natural", 0, "false", 1, "none", true);
					auto const pm = psi::algorithm::PowerMethod<psi::t_complex>().tolerance(1e-6);
					auto const nu1data = pm.AtA(Phi_nnls, psi::Vector<psi::t_complex>::Random(imsizey*imsizex));
					auto nu = nu1data.magnitude.real();

					auto forwardbackward_nnls_fista = psi::algorithm::ForwardBackward_nnls<t_complex>(target[f][t])
                                                        																				.itermax(500)
																																		.Phi(Phi_nnls)
																																		.mu(1./nu)
																																		.FISTA(true)
																																		.relative_variation(5e-5);

					auto diagnostic_fista = forwardbackward_nnls_fista();
					l2ball_epsilon[f](t) = diagnostic_fista.residual.norm();

					PURIPSI_HIGH_LOG("Estimated l2 ball bound for frequency {} block {}: {}", Decomp.my_frequencies()[f].freq_number, Decomp.my_frequencies()[f].time_blocks[t].time_block_number, l2ball_epsilon[f](t));
				}
			}

		}
		/*
				// Compute global epsilon bound (from Abdullah's code)
				auto global_epsilon = std::sqrt(Nm + 2*std::sqrt(2*Nm)) * sigma_noise;
				// Compute epsilon for each spectral band (from Abdullah's code)
				psi::Vector<psi::Vector<t_real>> l2ball_epsilon(Decomp.my_number_of_frequencies());
				for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
					l2ball_epsilon[f] = psi::Vector<t_real>::Zero(Decomp.my_frequencies()[f].number_of_time_blocks);
					for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
						l2ball_epsilon[f](t) = std::sqrt(static_cast<t_real>(target[f][t].size())/static_cast<t_real>(Nm)) * global_epsilon;
					}
				}*/

		if(Decomp.global_comm().is_root()){
			PURIPSI_HIGH_LOG("Calculated epsilon");
		}

		auto const mu = 1e-4; // hyperparameter related to the l21 norm

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

		auto kappa3 = 1./nu2; // inverse of the norm of the full measurement operator Phi (single value)
		if(Decomp.global_comm().is_root()){
			PURIPSI_HIGH_LOG("kappa3 is {} ", kappa3);
		}

		psi::Vector<t_real> eps_lambdas(3);
		eps_lambdas << 0.99, 1.01, 0.5*(sqrt(5)-1);
		if(Decomp.global_comm().is_root()){
			PURIPSI_HIGH_LOG("adaptive epsilon parameters: tol_in {}, tol_out {}, pecentage {} ", eps_lambdas(0), eps_lambdas(1), eps_lambdas(2));
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
			Image<t_complex> dirty_image = Image<t_complex>::Map(dirty[f].data(), imsizey, imsizex);
			pfitsio::write2d(dirty_image.real(), dirtyoutfile_fits + std::to_string(Decomp.my_frequencies()[f].freq_number) + fits_ending);
		}

		if(not only_dirty){
			// Instantiate algorithm
			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("Creating wideband primal-dual functor");
			}
			auto ppd = psi::algorithm::PrimalDualWidebandBlocking<t_complex>(target, imsizey*imsizex, l2ball_epsilon, Phi, Ui)
									.itermax(1000)
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
									.update_epsilon(true)
									.relative_variation_x(1e-4)
									.lambdas(eps_lambdas)
									.P(20)
									.decomp(Decomp)
									.adaptive_epsilon_start(200)
									.itermax_fb(20)
									.preconditioning(preconditioning)
									.relative_variation_fb(1e-8);

			// Sets weight after each pd iteration.
			//PURIPSI_HIGH_LOG("Creating reweighting-scheme functor");

			// !error of reweighted
			auto reweighted = psi::algorithm::reweighted(ppd)
			.itermax(10)
			.min_delta(min_delta)
			.decomp(Decomp)
			.is_converged(psi::RelativeVariation<psi::t_complex>(1e-5));

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
					Image<t_complex> out_image = Image<t_complex>::Map(diagnostic.algo.x.col(f).data(), imsizey, imsizex);
					pfitsio::write2d(out_image.real(), outfile_fits + std::to_string(f) + fits_ending);
				}
			}
		}


		psi::mpi::finalize();

	}

	return 0;
}
