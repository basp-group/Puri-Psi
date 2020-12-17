
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
#include "puripsi/operators.h"
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

	Eigen::initParallel();

	// Image parameters
	t_int imsizey = 2048;
	t_int imsizex = 2048;
	t_real nshiftx = static_cast<psi::t_real>(imsizex)/2.;
	t_real nshifty = static_cast<psi::t_real>(imsizey)/2.;
	auto const input_snr = 40.; // in dB
	t_int svd_size = 2;
	t_int padding_size  = 0;

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
	svd_size = argc >= 8 ? static_cast<t_int>(std::stod(static_cast<std::string>(argv[8]))) : svd_size;
	padding_size = argc >= 9 ? static_cast<t_int>(std::stod(static_cast<std::string>(argv[9]))) : padding_size;

	if(padding_size < 0){
		PURIPSI_HIGH_LOG("Error with the padding_size input variable ({}) setting it to zero", padding_size);
		padding_size = 0;
	}

	t_int image_size = imsizey*imsizex;

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
	t_int n_blocks = 4;

	if(argc > 11) {
		std::cout << "Usage:\n"
				"$ "
				<< argv[0] << " [uv] [model] [output] [imsizex] [imsizey] [only_dirty] [restoring]\n\n"
				"- uv: name of the sampling pattern file\n\n"
				"- model: name of the model file\n\n"
				"- output: name of output file\n\n"
				"- imsizex: integer specifying the x size of the image\n\n"
				"- imsizey: integer specifying the y size of the image\n\n"
				"- only_dirty: integer (0 or 1) or string (true/True/false/False) specifying whether the code only produces a dirty image and does not simulate the true image\n\n"
				"- restoring: integer (0 or 1) or string (true/True/false/False) specifying whether a restart file is being used to initialise the simulation\n\n"
				"- svd_size: integer specifying how many processes to use for the svd\n\n"
				"- padding_size: integer specifying how many processes to leave free to give the global root more memory space";
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


		auto const world = psi::mpi::Communicator::World(padding_size);

		bool parallel = true;

		auto Decomp = AstroDecomposition(parallel, world);

		auto adaptive_epsilon_start = 250;
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

			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("Using {} padding processes", padding_size);
			}

			Image<t_complex> global_X0;


			if(Decomp.global_comm().is_root()){
				global_X0 = pfitsio::read2d(modelName);   // model cube should be row-major [L, N], X0[N, L] after reading
				band_number = global_X0.cols();
				row_number = global_X0.rows();
				PURIPSI_HIGH_LOG("Number of channels is {} ", band_number);
				PURIPSI_HIGH_LOG("Image sise from file is {} ", row_number);
				PURIPSI_HIGH_LOG("Image size {} x {} = {}", imsizex, imsizey, imsizex*imsizey);
			}



			band_number = Decomp.global_comm().broadcast(band_number, Decomp.global_comm().root_id());
			row_number = Decomp.global_comm().broadcast(row_number, Decomp.global_comm().root_id());

			if(row_number != imsizex*imsizey){
				PURIPSI_ERROR("Image size is not the same as in the model file");
			}

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
		int lower_wavelet = 0;
		int number_of_wavelets = 0;
		if(Decomp.my_number_of_frequencies() != 0){
			lower_wavelet = Decomp.my_frequencies()[0].lower_wavelet;
			number_of_wavelets = Decomp.my_frequencies()[0].number_of_wavelets;
		}
		// TODO properly fix this so there are multiple distributed_sara used
		// because at the moment it assumes all wavelets are distributed the same way
		psi::wavelets::SARA distributed_sara = psi::wavelets::distribute_sara(sara, lower_wavelet, number_of_wavelets, number_of_wavelets);
		if(Decomp.my_number_of_frequencies() != 0){
			for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
				PURIPSI_LOW_LOG("Distributing wavelets {} {} {}",Decomp.global_comm().rank(), Decomp.my_frequencies()[f].lower_wavelet, Decomp.my_frequencies()[f].number_of_wavelets);
				Psi.emplace_back(psi::linear_transform<psi::t_complex>(distributed_sara, imsizey, imsizex));
				local_nlevels[f] = Decomp.my_frequencies()[f].number_of_wavelets;
			}
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
		std::vector<std::vector<utilities::vis_params>> uv_data(Decomp.my_number_of_frequencies());

		{
			// 1.wide-band frequency vector
			psi::Vector<t_real> freq(band_number);
			double freq0 = 1000.e6;
			double stepFreq = 1.e6;
			for(int l = 0; l < band_number; ++l){
				freq[l] = freq0 + stepFreq * l;
			}

			Image<t_complex> uv_model;
			if(Decomp.global_comm().is_root()){
				// 2.wide-band uv generated from a monochromatic realistic uv-coverage
				PURIPSI_HIGH_LOG("Reading uv data {}", uvdataName);
				uv_model = pfitsio::read2d(uvdataName);
				//		uv_model.transposeInPlace(); //! [P.-A.] keep it in backup for now, Ming's file may need this convention (if they are not modified beforehand)
				uv_model /= freq[band_number-1]/freq[0];            // avoid uv points outside [-pi, pi] for all freq channels
			}

			//! TODO: If we are padding the global communicator here we are currently sending this data to too many processes (i.e. the padding processes).
			//! This could be optimised to use a non-global comm that didn't include the padding processes.
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

		//! AJ Temporary until we calculate here
		pixel_size = 1;

		// 3.Compute the preconditioning matrix
		std::vector<std::vector<psi::Vector<t_real>>> Ui(Decomp.my_number_of_frequencies());
		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			Ui[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
			for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
				Ui[f].emplace_back(psi::Vector<psi::t_real>::Ones(uv_data[f][t].u.size()));
				puripsi::preconditioner<t_real>(Ui[f][t], uv_data[f][t].u, uv_data[f][t].v, ftsizev, ftsizeu);
			}
		}


		std::vector<std::vector<std::shared_ptr<psi::LinearTransform<psi::Vector<psi::t_complex>>>>> Phi(Decomp.my_number_of_frequencies());
		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			Phi[f].reserve(Decomp.my_frequencies()[f].number_of_time_blocks);
			for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){   // assume the data are order per blocks per channel
				Phi[f].emplace_back(std::make_shared<MeasurementOperator<Vector<t_complex>, t_complex>>(
						uv_data[f][t], Ui[f][t], imsizey, imsizex, pixel_size, pixel_size, over_sample, 100,
						0.0001, kernels::kernel::kb, nshifty, nshiftx, J, J, false));
			}
		}

		t_real nu2;
		if(not restoring){
			// Compute global operator norm
			auto const pm = psi::algorithm::PowerMethodWideband<psi::t_complex>().tolerance(1e-6).decomp(Decomp);
			auto const result = pm.AtA(Phi, psi::Matrix<psi::t_complex>::Random(imsizey*imsizex, Decomp.my_number_of_frequencies()));
			nu2 = result.magnitude.real();

			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("nu2 is {} ", nu2);
			}
		}

		// Deactivate measurement operator preconditioning as it's only required for the nu2 calculation
		for(int f=0; f<Decomp.my_number_of_frequencies(); ++f){
			for(int t=0; t<Decomp.my_frequencies()[f].number_of_time_blocks; ++t){
				(*Phi[f][t]).disable_preconditioning();
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


			auto svd_process_count = std::min((int)std::min(imsizey*imsizex, band_number),(int)std::min((int)svd_size, (int)Decomp.global_comm().size()));


			psi::mpi::Scalapack scalapack = psi::mpi::Scalapack(true);
			scalapack.setupBlacs(Decomp, svd_process_count, imsizey*imsizex, Decomp.global_number_of_frequencies()); // triggers warning/error message when setup not run already

			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("Using {} processes for the SVD",svd_process_count);
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
				psi::Vector<t_real> sigma;
				if(scalapack.involvedInSVD() or Decomp.global_comm().is_root()){
					sigma = psi::Vector<t_real>(std::min(image_size, band_number));
				}

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

				if(scalapack.involvedInSVD()){
					scalapack.setupSVD(A, sigma, U, VT);
				}

				if(Decomp.global_comm().is_root() or scalapack.scalapack_comm().is_root()) {
					data_svd = psi::Vector<t_real>(image_size*band_number);
				}

				if(!Decomp.parallel_mpi() or not scalapack.usingScalapack()){
					if(Decomp.global_comm().is_root())
					{
						PURIPSI_HIGH_LOG("Compute nuclear norm in serial");
						nuclear_norm = psi::nuclear_norm(global_dirty_matrix);
					}
				} else {
					if(Decomp.global_comm().is_root()){
						PURIPSI_HIGH_LOG("Compute nuclear norm in parallel");
						for(int l=0; l<band_number ; ++l){
							for(int n=0; n<image_size; ++n){
								data_svd[l*image_size+n] = real(global_dirty_matrix(n,l)); //global_dirty[l](n)
							}
						}
					}

					//! Send the data from the global_root to the root of the SVD operation. Because of process padding
					//! this might not be the same process.
					if(Decomp.global_comm().is_root() or (scalapack.involvedInSVD() and scalapack.scalapack_comm().is_root())){
						scalapack.sendToScalapackRoot(Decomp, data_svd);
					}

					if(scalapack.involvedInSVD()){ //! already checked in the functions, so not needed here
						scalapack.scatter(Decomp, A, data_svd, image_size, band_number, mpa, npa);
						scalapack.runSVD(A, sigma, U, VT);
					}

					//! Send the data from the root of the SVD operation to the global_root. Because of process padding
					//! this might not be the same process.
					//! Send the calculated sigma, U, and VT vectors from the scalapack root to the global root.
					if(Decomp.global_comm().is_root() or (scalapack.involvedInSVD() and scalapack.scalapack_comm().is_root())){
						scalapack.recvFromScalapackRoot(Decomp, sigma);
					}
					if (Decomp.global_comm().is_root()){
						nuclear_norm = psi::l1_norm(sigma);
						PURIPSI_HIGH_LOG("Parallel nuclear norm: {}", nuclear_norm);
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
						local_dirty = Decomp.my_root_wavelet_comm().broadcast(global_dirty_matrix, Decomp.my_root_wavelet_comm().root_id());
						partial = wavelet_regularization(Psi_Root, local_dirty, image_size*Decomp.my_number_of_root_wavelets());
					}else{
						partial = wavelet_regularization(Psi_Root, global_dirty_matrix, image_size*Decomp.my_number_of_root_wavelets());
					}
					l21_norm = psi::l21_norm(partial);

					if(Decomp.parallel_mpi() and Decomp.my_root_wavelet_comm().size() != 1){
						Decomp.my_root_wavelet_comm().distributed_sum(&l21_norm, Decomp.my_root_wavelet_comm().root_id());
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

			// Instantiate algorithm
			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("Creating wideband primal-dual functor");
			}

			auto ppd = psi::algorithm::PrimalDualWidebandBlocking<t_complex>(target, imsizey*imsizex, l2ball_epsilon, Phi, Ui)
																						.itermax(20)
																						.itermin(10)
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
																						.relative_variation(1e-6)
																						.residual_convergence(1.01)
																						.update_epsilon(false)
																						.relative_variation_x(1e-4)
																						.lambdas(eps_lambdas)
																						.P(100)
																						.decomp(Decomp)
																						.adaptive_epsilon_start(adaptive_epsilon_start)
																						.itermax_fb(20)
																						.preconditioning(true)
																						.relative_variation_fb(1e-8)
																						.objective_check_frequency(10)
																						.scalapack(scalapack);

			auto reweighted = psi::algorithm::reweighted(ppd)
			.itermax(1)
			.min_delta(min_delta)
			.update_delta([](t_real delta) { return 0.5 * delta; })
			.is_converged(psi::RelativeVariation<psi::t_complex>(1e-6));

			if(Decomp.global_comm().is_root()){
				PURIPSI_HIGH_LOG("Starting re-weighted primal dual from psi library");
			}
			double c_start = Decomp.global_comm().time();
			auto diagnostic = reweighted();
			double c_end = Decomp.global_comm().time();

			auto total_time = (c_end - c_start);

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


		psi::mpi::finalize();

	}

	return 0;
}
