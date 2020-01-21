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
	psi::logging::set_level("debug");
	puripsi::logging::set_level("debug");

	// Set numeric display parameter (std::cout)
	std::cout << std::scientific;
	std::cout.precision(4);

	// Data parameters
	auto const input_snr = 40;

	// Image parameters
	t_int block_number = 1;
	const t_int imsizey = 256;
	const t_int imsizex = 256;
	const std::string & name = "M31";
	const std::string test_number = "1";
	bool generate_data = true;

	// Gridding parameters
	t_int const J = 8;
	t_real const over_sample = 2;
	const t_int ftsizeu = imsizex*over_sample;
	const t_int ftsizev = imsizey*over_sample;

	bool preconditioning = true;
	int nlevels = 1;

	bool mpi_init_status = psi::mpi::init(argc, argv);

	if(!mpi_init_status){

		PURIPSI_ERROR("Problem initialising MPI. Quitting.");
		return 0;

	}else{

		auto const world = psi::mpi::Communicator::World();
		bool parallel = true;

		block_number = world.size();

		auto Decomp = AstroDecomposition(parallel, world);


		std::string const fitsfile = image_filename(name + ".fits");
		auto M31 = pfitsio::read2d(fitsfile);
		std::string const inputfile = output_filename(name + "_" + "input.fits");

		// This test case only has one frequency
		t_int frequencies = 1;

		std::vector<t_int> time_blocks = std::vector<t_int>(frequencies);
		time_blocks[0] = block_number;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(1);
		sub_blocks[0] = std::vector<t_int>(1);
		sub_blocks[0][0] = 0;

		Decomp.decompose_primal_dual(false, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks);

		if(Decomp.my_number_of_frequencies() > 0){

			// Reading u-v coordinates and visibilities
			std::vector<std::vector<utilities::vis_params>> uv_data;
			t_real const sigma_m = constant::pi / 3;
			t_int const number_of_vis = std::floor(M31.size() * 2.);

			if(Decomp.global_comm().is_root()){
				uv_data = std::vector<std::vector<puripsi::utilities::vis_params>>(1);
				uv_data[0] = std::vector<puripsi::utilities::vis_params>(block_number);
				for(int l = 0; l < block_number; ++l){
					uv_data[0][l] = utilities::random_sample_density(number_of_vis, 0, sigma_m);
					uv_data[0][l].units = puripsi::utilities::vis_units::radians;
				}
			}

			std::vector<std::vector<puripsi::utilities::vis_params>> my_uv_data(1);
			my_uv_data[0] = std::vector<puripsi::utilities::vis_params>(Decomp.my_frequencies()[0].number_of_time_blocks);

			Decomp.distribute_uv_data(uv_data, my_uv_data);

			// Generate measurement operators from the available uv_data
			std::vector<std::shared_ptr<const psi::LinearTransform<psi::Vector<psi::t_complex>>>> Phi(Decomp.my_frequencies()[0].number_of_time_blocks);
			std::vector<std::shared_ptr<const psi::LinearTransform<psi::Vector<psi::t_complex>>>> Phi2(Decomp.my_frequencies()[0].number_of_time_blocks);

			std::vector<std::shared_ptr<const puripsi::MeasurementOperator>> Phitemp(Decomp.my_frequencies()[0].number_of_time_blocks);

			std::vector<psi::Vector<t_real>> Ui(Decomp.my_frequencies()[0].number_of_time_blocks);

			for(int l = 0; l < Decomp.my_frequencies()[0].number_of_time_blocks; ++l){
				// preconditioner
				Ui[l] = psi::Vector<psi::t_real>::Ones(my_uv_data[0][l].u.size());
				puripsi::preconditioner<t_real>(Ui[l], my_uv_data[0][l].u, my_uv_data[0][l].v, ftsizev, ftsizeu);

				//Phi[l] = measurement_operator;
				Phi[l] = std::make_shared<const MeasurementOperator>(my_uv_data[0][l], J, J, "kb", imsizex, imsizey, 100, over_sample, 1, 1, "none", 0, "false", 1, "none", true); // true (wrong result with "false" option for the moment...)
				Phi2[l] = std::make_shared<const MeasurementOperator>(my_uv_data[0][l], Ui[l], J, J, "kb", imsizex, imsizey, 100, over_sample, 1, 1, "none", 0, "false", 1, "none", true); // true (wrong result with "false" option for the moment...)

				Phitemp[l] = std::make_shared<const MeasurementOperator>(my_uv_data[0][l], J, J, "kb", imsizex, imsizey, 100, over_sample, 1, 1, "none", 0, "false", 1, "none", true); // true (wrong result with "false" option for the moment...)
			}
			// Compute global operator norm (adapt power method to the time blocking setting)
			auto const pm = psi::algorithm::PowerMethodBlocking<psi::t_complex>().tolerance(1e-6).decomp(Decomp);
			auto const result = pm.AtA(Phi2, psi::Vector<psi::t_complex>::Random(imsizey*imsizex));
			auto const nu2 = result.magnitude.real(); // compute the norm of the global operator (i.e., with all the blocks)

			PURIPSI_HIGH_LOG("Calculating kappa");
			auto kappa = 0.;
			for(int l=0; l<Decomp.my_frequencies()[0].number_of_time_blocks; ++l){
				kappa += (Phitemp[l]->grid(my_uv_data[0][l].vis).real().maxCoeff());
			}
			kappa = kappa*1e-3/nu2;
			PURIPSI_MEDIUM_LOG("kappa is {} ", kappa);

			// Reading ground truth image x0 (from MATLAB generated .fits file)
			psi::Vector<psi::t_complex> x0 = Vector<psi::t_complex>::Map(M31.data(), M31.size(), 1);

			// Generate the ground truth measurements (y0)
			std::vector<psi::Vector<t_complex>> y0(Decomp.my_frequencies()[0].number_of_time_blocks);
			t_real normy0 = 0.;
			for(int l = 0; l < Decomp.my_frequencies()[0].number_of_time_blocks; ++l){
				y0[l] = Phitemp[l]->degrid(M31);
				normy0 += y0[l].squaredNorm();
			}
			t_int Nm = Decomp.my_frequencies()[0].number_of_time_blocks*my_uv_data[0][0].u.size(); // total number of measurements (assume same number of measurements for each block)
			auto sigma_noise = std::sqrt(normy0) / std::sqrt(Nm) * std::pow(10.0, -(input_snr / 20.0));

			// Add noise to the measurements
			std::vector<psi::Vector<t_complex>> target(Decomp.my_frequencies()[0].number_of_time_blocks);
			for(int l = 0; l < Decomp.my_frequencies()[0].number_of_time_blocks; ++l){
				my_uv_data[0][l].vis = utilities::add_noise(y0[l], 0., sigma_noise);
				target[l] = my_uv_data[0][l].vis;
			}

			// Compute global epsilon bound
			auto global_epsilon = std::sqrt(Nm + 2*std::sqrt(2*Nm)) * sigma_noise;
			// Compute epsilon for each spectral band, same procedure when blocking is considered
			psi::Vector<t_real> l2ball_epsilon(Decomp.my_frequencies()[0].number_of_time_blocks);
			for(int l = 0; l < Decomp.my_frequencies()[0].number_of_time_blocks; ++l){
				l2ball_epsilon(l) = std::sqrt(static_cast<t_real>(target[l].size())/static_cast<t_real>(Nm)) * global_epsilon;
			}

			// Set SARA dictionary
			psi::wavelets::SARA const sara{
				std::make_tuple("Dirac", 3u), std::make_tuple("DB1", 3u), std::make_tuple("DB2", 3u),
						std::make_tuple("DB3", 3u),   std::make_tuple("DB4", 3u), std::make_tuple("DB5", 3u),
						std::make_tuple("DB6", 3u),   std::make_tuple("DB7", 3u), std::make_tuple("DB8", 3u)};
			auto const Psi
			= psi::linear_transform<t_complex>(sara, imsizey, imsizex);
			auto const nlevels = sara.size();
			auto const min_delta = 1e-5;

			// Algorithm parameters
			auto const tau = 0.49; // 2 terms involved.
			PURIPSI_MEDIUM_LOG("tau is {} ", tau);
			auto sigma1 = 1.0;    // Daubechies wavelets are orthogonal, and only the identity is considered in addition to these dictionaries -> operator norm equal to 1
			PURIPSI_MEDIUM_LOG("sigma1 is {} ", sigma1);
			auto sigma2 = 1./nu2; // inverse of the norm of the full measurement operator Phi (single value)
			PURIPSI_MEDIUM_LOG("sigma2 is {} ", sigma2);
			psi::Vector<t_real> eps_lambdas(3);
			eps_lambdas << 0.99, 1.01, 0.5*(sqrt(5)-1);
			PURIPSI_MEDIUM_LOG("adaptive epsilon parameters: tol_in {}, tol_out {}, pecentage {} ", eps_lambdas(0), eps_lambdas(1), eps_lambdas(2));

			// Instantiate algorithm
			PURIPSI_HIGH_LOG("Creating time-blocking primal-dual functor");
			auto pd
			= psi::algorithm::PrimalDualTimeBlocking<t_complex>(target, imsizey*imsizex, l2ball_epsilon, Phi, Ui)
			.itermax(100)
			.tau(tau)
			.sigma1(sigma1)
			.sigma2(sigma2)
			.kappa(kappa)
			.levels(nlevels)
			.l1_proximal_weights(psi::Vector<t_real>::Ones(imsizex*imsizey*nlevels))
			.positivity_constraint(true)
			.relative_variation(5e-4)
			.residual_convergence(1.001)
			.Psi(Psi)
			.update_epsilon(true)
			.lambdas(eps_lambdas)
			.adaptive_epsilon_start(100)
			.P(20)
			.relative_variation_x(1e-2)
			.decomp(Decomp)
			.preconditioning(preconditioning);

			// Sets weight after each pd iteration.
			PURIPSI_HIGH_LOG("Creating reweighting-scheme functor");
			auto reweighted = psi::algorithm::reweighted(pd)
			.itermax(5)
			.min_delta(min_delta)
			.is_converged(psi::RelativeVariation<psi::t_complex>(1e-5))
			.decomp(Decomp);

			PURIPSI_HIGH_LOG("Starting reweighted primal dual from psi library");
			std::clock_t c_start = std::clock();
			auto diagnostic = reweighted();
			std::clock_t c_end = std::clock();

			if(Decomp.global_comm().is_root()){


				// Compute SNRs (for each block)
				Vector<t_complex> error_image = x0 - diagnostic.algo.x;
				t_real snr = 20. * std::log10(x0.norm() / error_image.norm());
				Vector<t_real> snr_data(Decomp.my_frequencies()[0].number_of_time_blocks);
				for(psi::t_uint l = 0; l < Decomp.my_frequencies()[0].number_of_time_blocks; ++l){
					snr_data(l) = 20. * std::log10(y0[l].norm() / diagnostic.algo.residual[l].norm());
				}
				auto total_time = (c_end - c_start) / CLOCKS_PER_SEC; // total runtime for the solver [in seconds]



				PURIPSI_HIGH_LOG("total computing time: {}", total_time);

				if(not diagnostic.algo.good){
					PURIPSI_HIGH_LOG("reweighted primal dual did not converge in {} iterations", diagnostic.algo.niters);
				}else{
					PURIPSI_HIGH_LOG("reweighted primal dual returned in {} iterations", diagnostic.algo.niters);
				}

				// Writing snr and time to a .txt file
				std::string const results
				= output_filename(name + "_results_kb_" + test_number + ".txt");
				std::ofstream out(results);
				out.precision(10);
				out << std::left << "Data: [" << name << "]\n";
				out << std::left << "Total computing time: " << total_time << " s\n";
				//				out << "====================================\n";
				//				out << std::setw(16) << std::left << "SNR (data) " << "\t" << std::setw(16) << std::left << "SNR (image)" << "\n";
				//				out << "------------------------------------\n";
				//				for(int l = 0; l < Decomp.frequencies()[0].number_of_time_blocks; ++l){
				//					out << std::setw(16) << std::left << snr << "\t" << std::setw(16) << std::left << snr_data(l) << "\n";
				//				}
				//				out << "====================================\n";
				out.close();

				// Write estimated image to a .fits file
				std::string const outfile_fits = output_filename(name +  "_kb" + "_");
				assert(diagnostic.algo.x.size() == imsizey*imsizex);
				Image<t_complex> image_save
				= Image<t_complex>::Map(diagnostic.algo.x.data(), imsizey, imsizex);
				pfitsio::write2d(image_save.real(), outfile_fits + std::to_string(1) + ".fits");
			}
		}
		psi::mpi::finalize();

	}

	return 0;
}
