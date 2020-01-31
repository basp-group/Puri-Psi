
#include <array>
#include <vector>
#include <memory>
#include <random>
#include <string>
#include <fstream>
#include <iostream>
//#include <iomanip>
#//include <cstdlib>
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
#include <psi/power_method_wideband.h>
#include <psi/primal_dual_wideband_blocking.h>

//#include "puripsi/casacore.h"
#include "puripsi/directories.h"
#include "puripsi/pfitsio.h"
#include "puripsi/types.h"
#include "puripsi/utilities.h"
#include "puripsi/logging.h"
#include "puripsi/MeasurementOperator.h"
#include "puripsi/preconditioner.h"
#include "puripsi/astrodecomposition.h"

using namespace puripsi;
using namespace puripsi::notinstalled;

int main(int argc, const char **argv) {
	psi::logging::initialize();
	puripsi::logging::initialize();
	psi::logging::set_level("critical");
	puripsi::logging::set_level("critical");

    Eigen::initParallel();
    PURIPSI_HIGH_LOG("Using {} Eigen threads",Eigen::nbThreads());

	// Image parameters
	t_int imsizey = 2048;
	t_int imsizex = 2048;
	auto const input_snr = 40.; // in dB

	// Gridding parameters
	t_int const J = 8;
	t_real const over_sample = 2;
	const string kernel = "kb";
	t_real pixel_size = -1;

	std::string uvdataName = argc >= 2 ? argv[1] : "/data/mjiang/MeerKAT_UVW_norm.fits";
	std::string modelName = argc >= 3 ? argv[2] : "/data/mjiang/XF_mat_C.fits";
	std::string name = argc >= 4 ? argv[3] : "puripsi_output";
	imsizex = argc >= 5 ? static_cast<t_int>(std::stod(static_cast<std::string>(argv[4]))) : imsizex;
	imsizey = argc >= 6 ? static_cast<t_int>(std::stod(static_cast<std::string>(argv[5]))) : imsizey;
	std::string temp_only_dirty = argc >= 6 ? argv[6] : "false";
	std::string temp_restoring = argc >= 7 ? argv[7] : "false";


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
	bool wavelet_parallelisation = true;

	t_int band_number;
	t_int row_number;
	t_int n_blocks = 1;

	if(argc > 9) {
		std::cout << "Usage:\n"
				"$ "
				<< argv[0] << " [uv] [model] [output]\n\n"
				"- uv: name of the sampling pattern file\n\n"
				"- model: name of the model file\n\n"
				"- output: name of output file\n\n"
				"- imsizex: integer specifying the x size of the image\n\n"
				"- imsizey: integer specifying the y size of the image\n\n"
				"- only_dirty: integer (0 or 1) or string (true/True/false/False) specifying whether the code only produces a dirty image and does not simulate the true image"
				"- restoring: integer (0 or 1) or string (true/True/false/False) specifying whether a restart file is being used to initialise the simulation";
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

		Image<t_complex> local_X0;

		//! Local scoping added to enable global_X0 to be removed after the data loading has happened and free up memory
		{
			Image<t_complex> global_X0;


			if(Decomp.global_comm().is_root()){
				global_X0 = pfitsio::read2d(modelName);   // model cube should be row-major [L, N], X0[N, L] after reading
				band_number = global_X0.cols();
				row_number = global_X0.rows();
				PURIPSI_HIGH_LOG("Number of channels is {} ", band_number);
			}

			// Reducing the number of channels used to fit in memory.
			band_number = 4;

			band_number = Decomp.global_comm().broadcast(band_number, Decomp.global_comm().root_id());
			row_number = Decomp.global_comm().broadcast(row_number, Decomp.global_comm().root_id());

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

			Decomp.decompose_primal_dual(true, true, false, wavelet_parallelisation, true, band_number, wavelet_levels, time_blocks, sub_blocks, false);

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
			PURIPSI_HIGH_LOG("Distributing wavelets {} {} {}",Decomp.global_comm().rank(), Decomp.my_frequencies()[f].lower_wavelet, Decomp.my_frequencies()[f].number_of_wavelets);
			Psi.emplace_back(psi::linear_transform<psi::t_complex>(distributed_sara, imsizey, imsizex));
			local_nlevels[f] = Decomp.my_frequencies()[f].number_of_wavelets;
		}

		psi::LinearTransform<psi::Vector<psi::t_complex>> Psi_Root = psi::linear_transform_identity<t_complex>();
		auto distributed_root_sara = psi::wavelets::distribute_sara(sara, Decomp.my_lower_root_wavelet(), Decomp.my_number_of_root_wavelets(), Decomp.global_number_of_root_wavelets());
		if(Decomp.my_number_of_root_wavelets()>0){
			PURIPSI_HIGH_LOG("Distributing root wavelets {} {} {}",Decomp.global_comm().rank(), Decomp.my_lower_root_wavelet(), Decomp.my_number_of_root_wavelets());
			Psi_Root = psi::linear_transform<psi::t_complex>(distributed_root_sara, imsizey, imsizex);
		}

		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			if(Decomp.own_this_frequency(Decomp.my_frequencies()[f].freq_number)){
				Image<t_complex> out_image = Image<t_complex>::Map(local_X0.col(f).data(), imsizey, imsizex);
				pfitsio::write2d(out_image.real(), clean_outfile_fits + std::to_string(Decomp.my_frequencies()[f].freq_number) + fits_ending);
			}
		}

		/* For the issue of storage, instead of reading data directly from an MS, hyperspectral data are generated from
		 * a realistic monochromatic uv file and a model hyperspectral image */
		// 1.wide-band frequency vector
		psi::Vector<t_real> freq(band_number);
		double freq0 = 1000.e6;
		double stepFreq = 1.e6;
		for(int l = 0; l < band_number; ++l){
			freq[l] = freq0 + stepFreq * l;
		}

		std::vector<std::vector<utilities::vis_params>> uv_data(Decomp.my_number_of_frequencies());

		{

			Image<t_complex> uv_model;
			if(Decomp.global_comm().is_root()){
				// 2.wide-band uv generated from a monochromatic realistic uv-coverage
				uv_model = pfitsio::read2d(uvdataName);
				uv_model.transposeInPlace();
				uv_model /= freq[band_number-1]/freq[0];            // avoid uv points outside [-pi, pi] for all freq channels
			}

			uv_model = Decomp.global_comm().broadcast(uv_model, Decomp.global_comm().root_id());

			for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
				uv_data[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
				for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
					utilities::vis_params vis_tmp;
					Image<t_real> uv_tmp;
					uv_tmp = (uv_model * freq[Decomp.my_frequencies()[f].freq_number]/freq[0]).real().eval();    // scaling of uv-coverage in terms of frequency
					vis_tmp.u = uv_tmp.row(0);
					vis_tmp.v = uv_tmp.row(1);
					vis_tmp.w = uv_tmp.row(2);
					vis_tmp.weights = Vector<t_complex>::Constant(uv_tmp.row(0).size(), 1);
					vis_tmp.vis = Vector<t_complex>::Constant(uv_tmp.row(0).size(), 1);
					uv_data[f].emplace_back(vis_tmp);
					uv_data[f][t].units = puripsi::utilities::vis_units::radians;
				}
			}

		}

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
			std::vector<std::vector<std::shared_ptr<const psi::LinearTransform<psi::Vector<psi::t_complex>>>>> Phi2(Decomp.my_number_of_frequencies());
			for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
				Phi2[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
				for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
					PURIPSI_HIGH_LOG("{} doing phi2 {} {}",Decomp.global_comm().rank(),f,t);
					Phi2[f].emplace_back(std::make_shared<const MeasurementOperator>(uv_data[f][t], Ui[f][t], J, J, kernel, imsizex, imsizey, 100, over_sample, pixel_size, pixel_size, "natural", 0, "false", 1, "none", true));
					PURIPSI_HIGH_LOG("{} done phi2 {} {}",Decomp.global_comm().rank(),f,t);
				}
			}

			// Compute global operator norm
			auto const pm = psi::algorithm::PowerMethodWideband<psi::t_complex>().tolerance(1e-6).decomp(Decomp);
			auto const result = pm.AtA(Phi2, psi::Matrix<psi::t_complex>::Random(imsizey*imsizex, Decomp.my_number_of_frequencies()));
			nu2 = result.magnitude.real();

			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("nu2 is {} ", nu2);
			}
		}

		// 4.Generate measurement operators from the available uv_data
		std::vector<std::vector<std::shared_ptr<const psi::LinearTransform<psi::Vector<psi::t_complex>>>>> Phi(Decomp.my_number_of_frequencies());
		for(int f=0; f< Decomp.my_number_of_frequencies(); ++f){
			Phi[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
			for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
				Phi[f].emplace_back(std::make_shared<const MeasurementOperator>(uv_data[f][t], J, J, "kb", imsizex, imsizey, 100, over_sample, pixel_size, pixel_size, "natural", 0, "false", 1, "none", true));
			}
		}

		// 5.Generate the ground truth measurements y0
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
		}

		// TODO: Does this need to be globally reduced?
		auto sigma_noise = std::sqrt(normy0) / std::sqrt(Nm) * std::pow(10.0, -(input_snr / 20.0));

		// 6.Add noise to the measurements
		std::vector<std::vector<psi::Vector<t_complex>>> target(Decomp.my_number_of_frequencies());
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

		psi::Vector<psi::Vector<t_real>> l2ball_epsilon(Decomp.my_number_of_frequencies());
		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			l2ball_epsilon[f] = psi::Vector<t_real>::Zero(Decomp.my_frequencies()[f].number_of_time_blocks);
		}

		if(not restoring){
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
			PURIPSI_HIGH_LOG("adaptive epsilon parameters: tol_in {}, tol_out {}, pecentage {} ", eps_lambdas(0), eps_lambdas(1), eps_lambdas(2));
		}

		/*	std::vector<Vector<t_complex>> global_dirty;
		if(Decomp.global_comm().is_root()){
			global_dirty = std::vector<Vector<t_complex>>(Decomp.global_number_of_frequencies());
			for(int f=0; f<Decomp.global_number_of_frequencies(); ++f){
				global_dirty[f] = Vector<t_complex>::Zero(imsizey*imsizex);
			}
		}*/

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

		//Decomp.collect_dirty_image(dirty, global_dirty);

		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			if(Decomp.own_this_frequency(f)){
				Image<t_complex> dirty_image = Image<t_complex>::Map(dirty[f].data(), imsizey, imsizex);
				pfitsio::write2d(dirty_image.real(), dirtyoutfile_fits + std::to_string(Decomp.my_frequencies()[f].freq_number) + "_" + fits_ending);
			}
		}

		if(not only_dirty){
			// Instantiate algorithm
			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("Creating wideband primal-dual functor");
			}

			auto ppd = psi::algorithm::PrimalDualWidebandBlocking<t_complex>(target, imsizey*imsizex, l2ball_epsilon, Phi, Ui)
																	.itermax(10)
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
																	.l21_proximal_weights(psi::Vector<t_real>::Ones(imsizex*imsizey*local_nlevels[0]))
																	.nuclear_proximal_weights(psi::Vector<t_real>::Ones(Decomp.global_number_of_frequencies()))
																	.positivity_constraint(true)
																	.relative_variation(5e-4)
																	.residual_convergence(1.001)
																	.update_epsilon(true)
																	.relative_variation_x(1e-4)
																	.lambdas(eps_lambdas)
																	.P(20)
																	.decomp(Decomp)
																	.adaptive_epsilon_start(adaptive_epsilon_start)
																	.itermax_fb(20)
																	.preconditioning(true)
																	.relative_variation_fb(1e-8);

			// Sets weight after each pd iteration.
			//PURIPSI_HIGH_LOG("Creating reweighting-scheme functor");

			// !error of reweighted
			auto reweighted = psi::algorithm::reweighted(ppd)
			.itermax(1)
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
