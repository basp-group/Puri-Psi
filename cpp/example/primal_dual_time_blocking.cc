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

#include "puripsi/MeasurementOperator.h"
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

int main(int argc, const char **argv) {
	psi::logging::initialize();
	puripsi::logging::initialize();
	psi::logging::set_level("critical");
	puripsi::logging::set_level("critical");

	// Image parameters
	//	t_int imsizey = 256;
	//	t_int imsizex = 256;
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
	//t_real dl = 1.;
	t_real pixel_size = -1;

	double tol_min = .8;
	double tol_max = 1.2;
	int block_size = 1.1e5;
	t_int field_id = 0;
	//int block_size = 2e4;


	bool preconditioning = false;
	bool only_dirty = false;
	bool restoring = false;
	bool wavelet_decomp = false;

	std::string dataName = argc >= 2 ? argv[1] : "data/ms/test.MS";
	std::string name = argc >= 3 ? argv[2] : "puripsi_output";
	std::string temp_preconditioning = argc >= 4 ? argv[3] : "false";
	imsizex = argc >= 5 ? static_cast<t_int>(std::stod(static_cast<std::string>(argv[4]))) : imsizex;
	imsizey = argc >= 6 ? static_cast<t_int>(std::stod(static_cast<std::string>(argv[5]))) : imsizey;
	dl = argc >= 7 ? static_cast<t_real>(std::stod(static_cast<std::string>(argv[6]))) : dl;
	std::string temp_only_dirty = argc >= 8 ? argv[7] : "false";
	std::string temp_restoring = argc >= 9 ? argv[8] : "false";
	std::string temp_wavelet_decomp = argc >= 10 ? argv[9] : "false";
    field_id = argc >= 11? static_cast<t_int>(std::stod(static_cast<std::string>(argv[10]))) : field_id;


	std::string const outfile_fits = name + ".fits";
	std::string const dirtyoutfile_fits = name +  "_dirty" + ".fits";
	std::string const dirtyresidualoutfile_fits = name +  "_residual" + ".fits";


	if(temp_preconditioning == "1" || temp_preconditioning == "true" || temp_preconditioning == "True"){
		preconditioning = true;
	}else if(temp_preconditioning == "0" || temp_preconditioning == "false" || temp_preconditioning == "False"){
		preconditioning = false;
	}else{
		std::cout << "Incorrect preconditioning parameter. Should be\n\n"
				"- preconditioning: 1 or true for preconditioning, 0 or false for not, false by default";
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

	if(temp_wavelet_decomp == "1" || temp_wavelet_decomp == "true" || temp_wavelet_decomp == "True"){
		wavelet_decomp = true;
	}else if(temp_wavelet_decomp == "0" || temp_wavelet_decomp == "false" || temp_wavelet_decomp == "False"){
		wavelet_decomp = false;
	}else{
		std::cout << "Incorrect wavelet decomp parameter. Should be\n\n"
				"- wavelet_decomp: 1 or true for using wavelet decompositionn, 0 or false for not, false by default";
		exit(0);
	}


	if(argc > 11) {
		std::cout << "Usage:\n"
				"$ "
				<< argv[0] << " [input ] [output] [preconditioning]\n\n"
				"- input: path to the image to clean (or name of standard PSI image)\n\n"
				"- output: name of output file\n\n"
				"- preconditioning: 1 or true for preconditioning, 0 or false for not, false by default\n\n";
		"- imsizex: integer specifying the x size of the output image\n\n";
		"- imsizey: integer specifying the y size of the output image\n\n";
		"- dl: real specifying the dl value used in pixel size calculations\n\n";
		"- only_dirty: 1 or true for only_dirty, 0 or false for not, false by default\n\n";
		"- restoring: 1 or true for restoring, 0 or false for not, false by default\n\n";
		"- wavelet_decomp: 1 or true for using wavelet decomposition, 0 or false for not, false by default.";






		exit(0);
	}

	bool mpi_init_status = psi::mpi::init(argc, argv);

	if(!mpi_init_status){

		PURIPSI_ERROR("Problem initialising MPI. Quitting.");
		return 0;

	}else{

		auto adaptive_epsilon_start = 500;
		if(restoring){
			adaptive_epsilon_start = 0;
		}

		auto const world = psi::mpi::Communicator::World();
		bool parallel = true;

		auto Decomp = AstroDecomposition(parallel, world);

		t_int n_blocks = 1;
		int channel_index = 0; //17 define the channel index of interest, consider a loop of this index for hyperspectral (HS) data
		t_int n_measurements = 0;


		if(!Decomp.parallel_mpi() or Decomp.global_comm().is_root()){
			PURIPSI_HIGH_LOG("Running with preconditioning: {}",preconditioning);
			PURIPSI_HIGH_LOG("Output file name: {}", outfile_fits);
		}

		std::vector<std::vector<puripsi::utilities::vis_params>> uv_data;

		//! Read in the data on the root process
		if(!Decomp.parallel_mpi() or Decomp.global_comm().is_root()){
			uv_data = std::vector<std::vector<puripsi::utilities::vis_params>> (1);
				uv_data[0] = get_time_blocks(dataName, &dl, &pixel_size, &n_blocks, &n_measurements, channel_index, &tol_min, &tol_max, &block_size, field_id);
		}

		//! Send global parameters from the process that did all the read of the data
		//! and therefore calculated/obtained these global values, to all the other
		//! processes
		Decomp.distribute_parameters_int_real(&channel_index, &pixel_size);
		Decomp.distribute_parameters_int_int(&n_blocks, &n_measurements);

		if(!Decomp.parallel_mpi() || Decomp.global_comm().is_root()){
			PURIPSI_HIGH_LOG("Channel index: {} Pixel size: {} Time blocks: {} Number of Measurements: {}",channel_index, pixel_size, n_blocks, n_measurements);
		}


		// Set SARA dictionary
		psi::wavelets::SARA const total_sara{
			std::make_tuple("Dirac", 3u), std::make_tuple("DB1", 3u), std::make_tuple("DB2", 3u),
					std::make_tuple("DB3", 3u),   std::make_tuple("DB4", 3u), std::make_tuple("DB5", 3u),
					std::make_tuple("DB6", 3u),   std::make_tuple("DB7", 3u), std::make_tuple("DB8", 3u)};
		auto const nlevels = total_sara.size();
		auto const min_delta = 1e-5;


		// This test case only has one frequency
		t_int frequencies = 1;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = n_blocks;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Decomp.decompose_primal_dual(false, true, false, wavelet_decomp, true, frequencies, wavelet_levels, time_blocks, sub_blocks);

		Decomp.set_checkpointing(true);
		Decomp.set_checkpointing_frequency(100);
		Decomp.set_restoring(restoring);
		psi::LinearTransform<psi::Vector<t_complex>> Psi = psi::linear_transform_identity<t_complex>();
		psi::wavelets::SARA distributed_sara =  psi::wavelets::distribute_sara(total_sara, Decomp.my_frequencies()[0].lower_wavelet, Decomp.my_frequencies()[0].number_of_wavelets, Decomp.frequencies()[0].number_of_wavelets);
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
			std::vector<std::shared_ptr<const psi::LinearTransform<psi::Vector<psi::t_complex>>>> Phi(Decomp.my_frequencies()[0].number_of_time_blocks);
			// Needed for no preconditioning (in the case that preconditioning is happening)
			std::vector<std::shared_ptr<const psi::LinearTransform<psi::Vector<psi::t_complex>>>> Phi2(Decomp.my_frequencies()[0].number_of_time_blocks);

			psi::Vector<psi::t_real> l2ball_epsilon(Decomp.my_frequencies()[0].number_of_time_blocks);
			psi::Vector<t_real> nu(Decomp.my_frequencies()[0].number_of_time_blocks);
			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("Creating Measurement Transforms for the blocks");
			}

			std::vector<psi::Vector<t_real>> Ui;

			if(preconditioning){
				Ui = std::vector<psi::Vector<t_real>>(Decomp.my_frequencies()[0].number_of_time_blocks);
			}

			for(int l = 0; l < Decomp.my_frequencies()[0].number_of_time_blocks; ++l){
				if(preconditioning){
					Ui[l] = psi::Vector<psi::t_real>::Ones(my_uv_data[0][l].u.size());
					puripsi::preconditioner<t_real>(Ui[l], my_uv_data[0][l].u, my_uv_data[0][l].v, ftsizev, ftsizeu);
					Phi2[l] = std::make_shared<const MeasurementOperator>(my_uv_data[0][l], Ui[l], J, J, "kb", imsizex, imsizey, 100, over_sample, pixel_size, pixel_size, "natural", 0, "false", 1, "none", true);
				}else{
					Phi2[l] = std::make_shared<const MeasurementOperator>(my_uv_data[0][l], J, J, "kb", imsizex, imsizey, 100, over_sample, pixel_size, pixel_size, "natural", 0, "false", 1, "none", true);
				}

				// Non-preconditioning operator for blocking power method.
				Phi[l] = 	std::make_shared<const MeasurementOperator>(my_uv_data[0][l], J, J, "kb", imsizex, imsizey, 100, over_sample, pixel_size, pixel_size, "natural", 0, "false", 1, "none", true);
				//! If we are reading in a checkpoint from file we will read in the epsilon rather than calculate it.
				if(!restoring){
					auto const pm = psi::algorithm::PowerMethod<psi::t_complex>().tolerance(1e-6);
					auto const nu1data = pm.AtA(Phi[l], psi::Vector<psi::t_complex>::Random(imsizey*imsizex));
					nu(l) = nu1data.magnitude.real();

					auto forwardbackward_nnls_fista = psi::algorithm::ForwardBackward_nnls<t_complex>(target[l])
		                																										.itermax(20)
																																.Phi(Phi[l])
																																.mu(1./nu(l))
																																.FISTA(true)
																																.relative_variation(5e-5);

					auto diagnostic_fista = forwardbackward_nnls_fista();
					l2ball_epsilon(l) = diagnostic_fista.residual.norm();

					PURIPSI_HIGH_LOG("Estimated l2 ball bound for block {}: {}", Decomp.my_frequencies()[0].time_blocks[l].time_block_number, l2ball_epsilon(l));
				}
			}

			t_real kappa;
			t_real nu2;

			if(!restoring){
				if(Decomp.global_comm().is_root()){
					PURIPSI_HIGH_LOG("Setting up Power Method");
				}
				// Compute global operator norm (adapt power method to the wideband setting)
				auto const pm = psi::algorithm::PowerMethodBlocking<psi::t_complex>().tolerance(1e-6).decomp(Decomp);

				auto const result = pm.AtA(Phi2, psi::Vector<psi::t_complex>::Random(imsizey*imsizex));

				nu2 = result.magnitude.real();

				if(Decomp.global_comm().is_root()){
					PURIPSI_HIGH_LOG("Calculating kappa");
				}

				Vector<t_real> tmp_dirty = Vector<t_real>::Zero(imsizex*imsizey);
				for(int l = 0; l < Decomp.my_frequencies()[0].number_of_time_blocks; ++l){
					tmp_dirty = tmp_dirty + ((Phi[l]->adjoint() * my_uv_data[0][l].vis).real()).eval();
				}
				Decomp.reduce_kappa(tmp_dirty, &kappa, nu2);

				if(Decomp.global_comm().is_root()){
					PURIPSI_HIGH_LOG("kappa is {} ", kappa);
				}
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
				sigma2 = 1./nu2; // inverse of the norm of the full measurement operator Phi (single value)
				if(Decomp.global_comm().is_root()){
					PURIPSI_HIGH_LOG("sigma2 is {} ", sigma2);
				}
			}
			psi::Vector<t_real> eps_lambdas(3);
			eps_lambdas << 0.99, 1.01, 0.5*(sqrt(5)-1);
			// Instantiate algorithm
			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("Creating wideband primal-dual functor");
			}

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
				.relative_variation(1e-5)
				.residual_convergence(1.005)
				.update_epsilon(true)
				.adaptive_epsilon_start(adaptive_epsilon_start)
				.P(100)
				.Psi(Psi)
				.decomp(Decomp)
				.relative_variation_x(1e-4)
				.preconditioning(preconditioning);

				// Sets weight after each pd iteration.
				if(Decomp.global_comm().is_root()){
					PURIPSI_HIGH_LOG("Creating reweighting-scheme functor");
				}
				auto reweighted = psi::algorithm::reweighted(pd)
				.itermax(10)
				.min_delta(min_delta)
				.is_converged(psi::RelativeVariation<psi::t_complex>(1e-5))
				.decomp(Decomp);

				if(Decomp.global_comm().is_root()){
					PURIPSI_HIGH_LOG("Starting reweighted primal dual from psi library");
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
