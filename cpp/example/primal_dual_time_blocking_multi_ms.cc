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

#include "puripsi/operators.h"
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
using namespace puripsi::operators;

int main(int argc, const char **argv) {
	psi::logging::initialize();
	puripsi::logging::initialize();
	psi::logging::set_level("critical");
	puripsi::logging::set_level("critical");

	// Image parameters
	const t_int imsizey = 4096;
	const t_int imsizex = 4096;
	const t_real nshiftx = static_cast<psi::t_real>(imsizex)/2.;
	const t_real nshifty = static_cast<psi::t_real>(imsizey)/2.;
	const std::string test_number = "1";

	// Gridding parameters
	t_int const J = 8;
	t_real const over_sample = 2;
	const string kernel = "kb";
	const t_int ftsizeu = imsizex*over_sample;
	const t_int ftsizev = imsizey*over_sample;
	t_real dl = 2.5059; // keep 1.8 as a generic value for other simulations
	t_real pixel_size = 0.04; // fixed here...

	std::string temp_only_dirty = argc >= 2 ? argv[1] : "false";
	std::string temp_restoring = argc >= 3 ? argv[2] : "false";


	bool only_dirty = false;
	bool preconditioning = true;
	bool restoring =  false;
	bool wavelet_parallelisation = true;

	std::vector<std::string> dataName{
		"/lustre/home/shared/sc004/cpp_ms/CYG-8422-D-1X2MHZ-10S.MS",
		"/lustre/home/shared/sc004/cpp_ms/CYG-8422-C-1X2MHZ-10S.MS",
		"/lustre/home/shared/sc004/cpp_ms/CYG-8422-B-1X2MHZ-10S.MS",
		"/lustre/home/shared/sc004/cpp_ms/CYG-8422-A1-1X2MHZ-10S.MS",
		"/lustre/home/shared/sc004/cpp_ms/CYG-8422-A2-1X2MHZ-10S.MS",
	};

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


	t_int n_datasets = dataName.size();

	Vector<t_int> imsizey_config(dataName.size());
	imsizey_config << 128, 256, 512, 2560, 2560;
	Vector<t_int> imsizex_config = imsizey_config;

	std::string name = "puripsi_cygA";

	std::string const outfile_fits = name + ".fits";
	std::string const dirtyoutfile_fits = name +  "_dirty" + ".fits";
	std::string const dirtyresidualoutfile_fits = name +  "_residual" + ".fits";

	bool mpi_init_status = psi::mpi::init(argc, argv);

	if(!mpi_init_status){

		PURIPSI_ERROR("Problem initialising MPI. Quitting.");
		return 0;

	}else{

		auto const world = psi::mpi::Communicator::World();
		bool parallel = true;

		auto Decomp = AstroDecomposition(parallel, world);

		t_int n_blocks = 1;
		int channel_index = 0;
		t_int n_measurements = 0;
		auto adaptive_epsilon_start = 500;
		if(restoring){
			adaptive_epsilon_start = 0;
		}

		// Set SARA dictionary
		psi::wavelets::SARA const total_sara{
			std::make_tuple("Dirac", 3u), std::make_tuple("DB1", 3u), std::make_tuple("DB2", 3u),
					std::make_tuple("DB3", 3u),   std::make_tuple("DB4", 3u), std::make_tuple("DB5", 3u),
					std::make_tuple("DB6", 3u),   std::make_tuple("DB7", 3u), std::make_tuple("DB8", 3u)};
		auto const nlevels = total_sara.size();
		auto const min_delta = 1e-6;

		std::vector<std::vector<puripsi::utilities::vis_params>> uv_data(1);
		Vector<t_int> n_blocks_per_dataset(n_datasets);

		//! Read in the data on the root process
		if(!Decomp.parallel_mpi() or Decomp.global_comm().is_root()){
			uv_data[0] =  puripsi::get_time_blocks_multi_file(dataName, &dl, &pixel_size, &n_blocks, &n_measurements, channel_index, n_blocks_per_dataset);
		}

		//! Send global parameters from the process that did all the read of the data
		//! and therefore calculated/obtained these global values, to all the other
		//! processes
		Decomp.distribute_parameters_int_real(&channel_index, &pixel_size);
		Decomp.distribute_parameters_int_int(&n_blocks, &n_measurements);
		t_real field_of_view = pixel_size*imsizex;

		// This test case only has one frequency
		t_int frequencies = 1;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = n_blocks;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Decomp.decompose_primal_dual(false, true, false, wavelet_parallelisation, true, frequencies, wavelet_levels, time_blocks, sub_blocks);

		if(!Decomp.parallel_mpi() or Decomp.global_comm().is_root()){
			PURIPSI_HIGH_LOG("Channel index: {} Pixel size: {} Time blocks: {} Number of Measurements: {}",channel_index, pixel_size, n_blocks, n_measurements);
		}

		n_blocks_per_dataset = Decomp.global_comm().broadcast(n_blocks_per_dataset, Decomp.global_comm().root_id());

		if(!Decomp.parallel_mpi() or Decomp.global_comm().is_root()){
			PURIPSI_HIGH_LOG("Running with preconditioning: {}",preconditioning);
			PURIPSI_HIGH_LOG("Output file name: {}", outfile_fits);
		}

		Decomp.set_checkpointing(true);
		Decomp.set_checkpointing_frequency(100);
		Decomp.set_restoring(restoring);

		PURIPSI_HIGH_LOG("Distributing wavelets {} {} {}",Decomp.global_comm().rank(),Decomp.my_frequencies()[0].lower_wavelet, Decomp.my_frequencies()[0].number_of_wavelets);
		psi::LinearTransform<psi::Vector<t_complex>> Psi = psi::linear_transform_identity<t_complex>();
		psi::wavelets::SARA distributed_sara =  psi::wavelets::distribute_sara(total_sara, Decomp.my_frequencies()[0].lower_wavelet, Decomp.my_frequencies()[0].number_of_wavelets,Decomp.frequencies()[0].number_of_wavelets);
		Psi = psi::linear_transform<t_complex>(distributed_sara, imsizey, imsizex);
		auto const local_nlevels = Decomp.my_frequencies()[0].number_of_wavelets;


		if(Decomp.my_number_of_frequencies() > 0){

			std::vector<std::vector<puripsi::utilities::vis_params>> my_uv_data(1);
			my_uv_data[0] = std::vector<puripsi::utilities::vis_params>(Decomp.my_frequencies()[0].number_of_time_blocks);

			Decomp.distribute_uv_data(uv_data, my_uv_data);

			std::vector<psi::Vector<psi::t_complex>> target(Decomp.my_frequencies()[0].number_of_time_blocks);
			for(int n = 0; n < Decomp.my_frequencies()[0].number_of_time_blocks; ++n){
				target[n] = my_uv_data[0][n].vis.array() * my_uv_data[0][n].weights.array();
			}


			// Generate measurement operators from the available uv_data
			psi::Vector<psi::t_real> l2ball_epsilon(Decomp.my_frequencies()[0].number_of_time_blocks);
			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("Creating Measurement Transforms for the blocks");
			}

			std::vector<psi::Vector<t_real>> Ui;

			if(preconditioning){
				Ui = std::vector<psi::Vector<t_real>>(Decomp.my_frequencies()[0].number_of_time_blocks);
				for(int l = 0; l < Decomp.my_frequencies()[0].number_of_time_blocks; ++l){
					Ui[l] = psi::Vector<psi::t_real>::Ones(my_uv_data[0][l].u.size());
				}
			}

			t_real kappa;
			t_real nu2;

			std::vector<std::shared_ptr<psi::LinearTransform<psi::Vector<psi::t_complex>>>> Phi(Decomp.my_frequencies()[0].number_of_time_blocks);

			for(int t = 0; t < Decomp.my_frequencies()[0].number_of_time_blocks; ++t){

				if(preconditioning){
					puripsi::preconditioner<t_real>(Ui[t], my_uv_data[0][t].u, my_uv_data[0][t].v, ftsizev, ftsizeu);
					Phi[t]= std::make_shared<MeasurementOperator<Vector<t_complex>, t_complex>>(
							my_uv_data[0][t], Ui[t], imsizey, imsizex, pixel_size, pixel_size, over_sample, 100,
							0.0001, kernels::kernel::kb, nshifty, nshiftx, J, J, false);
				}else{
					Phi[t] = std::make_shared<MeasurementOperator<Vector<t_complex>, t_complex>>(
							my_uv_data[0][t], imsizey, imsizex, pixel_size, pixel_size, over_sample, 100,
							0.0001, kernels::kernel::kb, nshifty, nshiftx, J, J, false);
				}

			}

			if(!restoring){

				// Compute global operator norm (adapt power method to the wideband setting)
				// instanciate the power method
				auto const pm = psi::algorithm::PowerMethodBlocking<psi::t_complex>().tolerance(1e-6).decomp(Decomp);

				auto const result = pm.AtA(Phi, psi::Vector<psi::t_complex>::Random(imsizey*imsizex));

				nu2 = result.magnitude.real();
				if(Decomp.global_comm().is_root()){
					PURIPSI_HIGH_LOG("nu2 {}", nu2);
				}

				if(Decomp.global_comm().is_root()){
					PURIPSI_HIGH_LOG("Calculating kappa");
				}
				kappa = 2e-5;
				// Vector<t_real> tmp_dirty = Vector<t_real>::Zero(imsizex*imsizey);
				// for(int l = 0; l < Decomp.my_frequencies()[0].number_of_time_blocks; ++l){
				// 	tmp_dirty = tmp_dirty + ((Phi[l]->adjoint() * my_uv_data[0][l].vis).real()).eval();
				// }
				// Decomp.reduce_kappa(tmp_dirty, &kappa, nu2);

				if(Decomp.global_comm().is_root()){
					PURIPSI_HIGH_LOG("kappa is {} ", kappa);
				}

			}

			for(int l = 0; l < Decomp.my_frequencies()[0].number_of_time_blocks; ++l){

				// Calculate which data_id this block is from so we can work out later what imsize_x and imsize_y to apply to it
				int block_number = Decomp.my_frequencies()[0].time_blocks[l].time_block_number;
				int data_id = 0;
				int blocks = n_blocks_per_dataset(data_id);
				for(int k=-1; k<n_datasets-1; k++){
					if(block_number <= blocks){
						data_id = k+1;
						break;
					}else{
						blocks = blocks + n_blocks_per_dataset(k+2);
					}
				}

				if(my_uv_data[0][l].u.size() == 0){
					PURIPSI_HIGH_LOG("Block {} on rank {} is zero in size",block_number,Decomp.global_comm().rank());

				}

				// None preconditioning operator for blocking power method.
				Phi[l] = std::make_shared<MeasurementOperator<Vector<t_complex>, t_complex>>(
						my_uv_data[0][l], imsizey, imsizex, pixel_size, pixel_size, over_sample, 100,
						0.0001, kernels::kernel::kb, nshifty, nshiftx, J, J, false);
				//! If we are reading in a checkpoint from file we will read in the epsilon rather than calculate it.
				if(!restoring){

					auto pixel_size_config = field_of_view / imsizey_config(data_id);
					std::shared_ptr<psi::LinearTransform<psi::Vector<psi::t_complex>>> Phi_nnls = std::make_shared<MeasurementOperator<Vector<t_complex>, t_complex>>(
							my_uv_data[0][l], imsizey_config(data_id), imsizex_config(data_id), pixel_size_config, pixel_size_config, over_sample, 100,
							0.0001, kernels::kernel::kb, nshifty, nshiftx, J, J, false);
					auto const pm = psi::algorithm::PowerMethod<psi::t_complex>().tolerance(1e-6);
					auto const nu1data = pm.AtA(Phi_nnls, psi::Vector<psi::t_complex>::Random(imsizey_config(data_id)*imsizex_config(data_id)));
					auto nu = nu1data.magnitude.real();

					auto forwardbackward_nnls_fista = psi::algorithm::ForwardBackward_nnls<t_complex>(target[l])
                                                        																.itermax(500)
																														.Phi(Phi_nnls)
																														.mu(1./nu)
																														.FISTA(true)
																														.relative_variation(5e-5);

					auto diagnostic_fista = forwardbackward_nnls_fista();
					l2ball_epsilon(l) = diagnostic_fista.residual.norm();

					PURIPSI_HIGH_LOG("Estimated l2 ball bound for block {}: {}", Decomp.my_frequencies()[0].time_blocks[l].time_block_number, l2ball_epsilon(l));
				}

			}

			// Deactivate measurement operator preconditioning as it's only required for the nu2 calculation
			for(int t = 0; t < Decomp.my_frequencies()[0].number_of_time_blocks; ++t){
				(*Phi[t]).disable_preconditioning();
			}


			// Algorithm parameters
			auto const tau = 0.49; // 2 terms involved.
			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("tau is {} ", tau);
			}
			auto sigma1 = 1.0;    // Daubechies wavelets are orthogonal, and only the identity is considered in addition to these dictionaries -> operator norm equal to 1
			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("sigma1 is {} ", sigma1);
			}
			t_real sigma2;
			if(!restoring){
				sigma2 = 1./nu2; // inverse of the norm of the full measurement operator Phi (single value), to be loaded
				if(Decomp.global_comm().is_root()){
					PURIPSI_HIGH_LOG("sigma2 is {} ", sigma2);
				}
			}
			psi::Vector<t_real> eps_lambdas(3);
			eps_lambdas << 0.999, 1.001, 0.5*(sqrt(5)-1); // to be verified (take a look at the ppd)


			Vector<t_complex> dirty = Vector<t_complex>::Zero(imsizey*imsizex);
			for(int n=0; n<Decomp.my_frequencies()[0].number_of_time_blocks; ++n){
				dirty = dirty +  (Phi[n]->adjoint() * my_uv_data[0][n].vis);
			}

			Decomp.my_frequencies()[0].freq_comm.distributed_sum(dirty, Decomp.my_frequencies()[0].freq_comm.root_id());

			if(Decomp.global_comm().is_root()){
				Image<t_complex> dirty_image = Image<t_complex>::Map(dirty.data(), imsizey, imsizex);
				pfitsio::write2d(dirty_image.real(), dirtyoutfile_fits);
			}

			if(not only_dirty){

				auto pd
				= psi::algorithm::PrimalDualTimeBlocking<t_complex>(target, imsizey*imsizex, l2ball_epsilon, Phi, Ui)
				.itermax(2000)
				.tau(tau)
				.sigma1(sigma1)
				.sigma2(sigma2)
				.lambdas(eps_lambdas)
				.kappa(kappa)
				.levels(local_nlevels)
				.l1_proximal_weights(psi::Vector<t_real>::Ones(imsizex*imsizey*local_nlevels))
				.positivity_constraint(true)
				.relative_variation(1e-6) // relative variation of the solution in the Matlab code, not the objective function
				.residual_convergence(1.001)
				.update_epsilon(true)
				.adaptive_epsilon_start(adaptive_epsilon_start)
				.P(100)
				.Psi(Psi)
				.decomp(Decomp)
				.relative_variation_x(1e-4) // criterion for the espilon update (relative variation of the iterate)
				.preconditioning(preconditioning)
				.itermax_fb(20) 
				.relative_variation_fb(1e-8);

				// Sets weight after each pd iteration.
				if(Decomp.global_comm().is_root()){
					PURIPSI_HIGH_LOG("Creating re-weighting-scheme functor");
				}
				auto reweighted = psi::algorithm::reweighted(pd)
				.itermax(30)
				.min_delta(min_delta)
				.is_converged(psi::RelativeVariation<psi::t_complex>(5e-5))
				.decomp(Decomp);

				if(Decomp.global_comm().is_root()){
					PURIPSI_HIGH_LOG("Starting re-weighted primal dual from psi library");
				}
				std::clock_t c_start = std::clock();
				auto diagnostic = reweighted();
				std::clock_t c_end = std::clock();

				if(Decomp.global_comm().is_root()){

					auto total_time = (c_end - c_start) / CLOCKS_PER_SEC;

					if(not diagnostic.algo.good){
						PURIPSI_HIGH_LOG("reweighted primal dual did not converge in {} iterations", diagnostic.algo.niters);
					}else{
						PURIPSI_HIGH_LOG("reweighted primal dual returned in {} iterations", diagnostic.algo.niters);
					}


					PURIPSI_HIGH_LOG("Total computing time: {}", total_time);


					// Write estimated image to a .fits file
					assert(diagnostic.algo.x.size() == imsizey*imsizex);
					Image<t_complex> image_save
					= Image<t_complex>::Map(diagnostic.algo.x.data(), imsizey, imsizex);
					pfitsio::write2d(image_save.real(), outfile_fits);
				}
				// Compute rescaling factor (peak value of the PSF)
				Vector<t_complex> dirac = Vector<t_complex>::Zero(imsizey*imsizex);
				t_int index = std::floor(imsizex/2)*imsizey + std::floor(imsizey/2); // see if ok for the position
				dirac(index) = 1.;

				Vector<t_complex> psf = Vector<t_complex>::Zero(imsizey*imsizex);
				for(int n=0; n<Decomp.my_frequencies()[0].number_of_time_blocks; ++n){
					Vector<t_complex> temp = (*Phi[n]) * dirac; // error here !!
					psf = psf +  (Phi[n]->adjoint() * temp);
				}

				Decomp.my_frequencies()[0].freq_comm.distributed_sum(psf, Decomp.my_frequencies()[0].freq_comm.root_id());

				Vector<t_complex> dirty_residual = Vector<t_complex>::Zero(imsizey*imsizex);
				for(int n=0; n<Decomp.my_frequencies()[0].number_of_time_blocks; ++n){
					dirty_residual = dirty_residual +  (Phi[n]->adjoint() * diagnostic.algo.residual[n]);
				}

				Decomp.my_frequencies()[0].freq_comm.distributed_sum(dirty_residual, Decomp.my_frequencies()[0].freq_comm.root_id());


				if(Decomp.global_comm().is_root()){
					PURIPSI_HIGH_LOG("Writing residual image to {}", dirtyresidualoutfile_fits);
					auto psf_peak = psf.real().maxCoeff();
					dirty_residual = dirty_residual / psf_peak;
					Image<t_complex> dirty_image = Image<t_complex>::Map(dirty_residual.data(), imsizey, imsizex);
					pfitsio::write2d(dirty_image.real(), dirtyresidualoutfile_fits);
				}

			}

		}

		psi::mpi::finalize();

	}


	return 0;
}
