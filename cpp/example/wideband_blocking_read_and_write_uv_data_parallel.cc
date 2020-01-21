//
// Created by mjiang on 6/6/19.
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

//#include "puripsi/casacore.h"
#include "puripsi/directories.h"
#include "puripsi/pfitsio.h"
#include "puripsi/types.h"
#include "puripsi/utilities.h"
#include "puripsi/logging.h"
#include "puripsi/astroio.h"
#include "puripsi/time_blocking.h"
#include "puripsi/astrodecomposition.h"

using namespace puripsi;
using namespace puripsi::notinstalled;

int main(int argc, const char **argv) {
	psi::logging::initialize();
	puripsi::logging::initialize();
	psi::logging::set_level("critical");
	puripsi::logging::set_level("critical");

	t_real dl = 1.75;
	t_real pixel_size = 0.04;

	std::vector<std::vector<std::string>> dataName(2);
	for(int i=0; i< dataName.size(); i++){
		dataName[i] = std::vector<std::string>(2);
	}
	dataName[0][0] = "/home/nx01/nx01/adrianj/compress/puri-psi-dev/build/cpp/example/wideband_ms/CYG-A-5-8M10S.MS";
	dataName[0][1] = "/home/nx01/nx01/adrianj/compress/puri-psi-dev/build/cpp/example/wideband_ms/CYG-C-5-8M10S.MS";
	dataName[1][0] = "/home/nx01/nx01/adrianj/compress/puri-psi-dev/build/cpp/example/wideband_ms/CYG-A-7-8M10S.MS";
	dataName[1][1] = "/home/nx01/nx01/adrianj/compress/puri-psi-dev/build/cpp/example/wideband_ms/CYG-C-7-8M10S.MS";

	t_int n_datasets = dataName.size();

	std::string name = argc >= 2 ? argv[1] : "uv_data.dat";

	t_int band_number = 0;
	t_int row_number;
	t_int n_blocks = 0;
	t_int n_measurements = 0;
	t_int field_id = 2;

	if(argc > 2) {
		std::cout << "Usage:\n"
				"$ "
				<< argv[0] <<
				"- output: name of output file\n\n";
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

		std::vector<std::vector<puripsi::utilities::vis_params>> global_uv_data;

		Vector<t_int> frequencies_per_dataset(n_datasets);
		Vector<t_int> n_blocks_per_dataset(n_datasets);
		Vector<t_int> blocks_per_frequency;
		Vector<t_int> measurements_per_frequency;
		Vector<std::set<::casacore::uInt>> spws_ids(n_datasets);
		Vector<Vector<t_int>> spws_ids_int(n_datasets);

		if(Decomp.global_comm().is_root()){

			for(int i=0; i<n_datasets; i++){
				puripsi::extract_number_of_channels_and_spectal_windows(dataName[i][0], field_id, frequencies_per_dataset[i], spws_ids[i]);
				band_number += frequencies_per_dataset[i];
				PURIPSI_LOW_LOG("Number of Spectral Windows {}",spws_ids[i].size());
				spws_ids_int[i] = Vector<t_int>(spws_ids[i].size());
				int window = 0;
				for(auto f : spws_ids[i]) {
					spws_ids_int[i][window] = f;
					PURIPSI_LOW_LOG("Spectral Windows {}",spws_ids_int[i][window]);
					window++;
				}
			}

		}

		band_number = Decomp.global_comm().broadcast(band_number, Decomp.global_comm().root_id());
		frequencies_per_dataset = Decomp.global_comm().broadcast(frequencies_per_dataset, Decomp.global_comm().root_id());
		for(int i=0; i<n_datasets; i++){
			spws_ids_int[i] = Decomp.global_comm().broadcast(spws_ids_int[i], Decomp.global_comm().root_id());
		}

		//! Create a frequency decomposition to split the frequencies over available processes. Ignore wavelet, time block, and
		//! sub block levels because we do not have information on these aspects of the data set yet.

		std::vector<t_int> time_blocks = std::vector<t_int>(band_number);
		for(int b=0; b<band_number; b++){
			time_blocks[b] = 1;
		}

		std::vector<t_int> wavelet_levels = std::vector<t_int>(band_number);
		for(int b=0; b<band_number; b++){
			wavelet_levels[b] = 1;
		}

		std::vector<std::vector<t_int>> sub_blocks = std::vector<std::vector<t_int>>(band_number);
		for(int b=0; b<band_number; b++){
			sub_blocks[b] = std::vector<t_int>(1);
			for(int t=0; t<1; t++){
				sub_blocks[b][t] = 0;
			}
		}

		//! Create a frequency only decomposition. Here we have no knowledge yet of the number of blocks per frequency, so we
		//! are assuming there is only one block per frequency (as defined in the block loop above.
		Decomp.decompose_primal_dual(true, false, false, false, false, band_number, wavelet_levels, time_blocks, sub_blocks, true);

		Vector<t_int> global_blocks_per_frequency;
		Vector<t_int> global_measurements_per_frequency;

		blocks_per_frequency = Vector<t_int>(Decomp.my_number_of_frequencies());
		measurements_per_frequency = Vector<t_int>(Decomp.my_number_of_frequencies());
		global_blocks_per_frequency =  Vector<t_int>(Decomp.global_number_of_frequencies());
		if(Decomp.global_comm().is_root()){
			global_measurements_per_frequency =  Vector<t_int>(Decomp.global_number_of_frequencies());
		}
		std::vector<std::vector<puripsi::utilities::vis_params>> uv_data(Decomp.my_number_of_frequencies());
		std::vector<int> frequency_index(Decomp.my_number_of_frequencies());

		int uv_number = 0;
		int freq_number = 0;
		for(int i=0; i<n_datasets; i++){
			PURIPSI_HIGH_LOG("{} Reading in files {} {}",Decomp.global_comm().rank(), dataName[i][0], dataName[i][1]);
			for(int f=0; f<spws_ids_int[i].size(); f++) {
				for(int j=0; j<frequencies_per_dataset[i]/spws_ids_int[i].size(); j++){
					if(Decomp.own_this_frequency(freq_number)){
						PURIPSI_HIGH_LOG("{} Processing frequency {}",Decomp.global_comm().rank(),freq_number);
						uv_data[uv_number] = get_time_blocks_multi_file_spectral_window(dataName[i], &dl, &pixel_size, &n_blocks, &n_measurements, spws_ids_int[i][f], j, n_blocks_per_dataset, field_id);
						frequency_index[uv_number] = freq_number;
						blocks_per_frequency[uv_number] = n_blocks;
						measurements_per_frequency[uv_number] = n_measurements;
						uv_number++;
					}
					freq_number++;
				}
			}
		}

		Decomp.gather_frequency_local_vector_int(global_blocks_per_frequency, blocks_per_frequency);
		global_blocks_per_frequency = Decomp.global_comm().broadcast(global_blocks_per_frequency, Decomp.global_comm().root_id());
		Decomp.gather_frequency_local_vector_int(global_measurements_per_frequency, measurements_per_frequency);

		for(int b=0; b<band_number; b++){
			time_blocks[b] = global_blocks_per_frequency[b];
		}

		auto NewDecomp = AstroDecomposition(parallel, world);
		//! Create a frequency only decomposition using actual number of blocks per frequency. This we can then use to
		//! gather all the blocks back to the root for saving to file.
		NewDecomp.decompose_primal_dual(true, false, false, false, false, band_number, wavelet_levels, time_blocks, sub_blocks, true);


		if(NewDecomp.global_comm().is_root()){
			global_uv_data = std::vector<std::vector<puripsi::utilities::vis_params>>(band_number);
			for(int f=0; f<band_number; f++){
				global_uv_data[f] = std::vector<utilities::vis_params>(global_blocks_per_frequency[f]);
			}
		}


		NewDecomp.gather_uv_data(global_uv_data, uv_data);

		if(NewDecomp.global_comm().is_root()){

			PURIPSI_HIGH_LOG("Number of channels is {} ", band_number);
			for(int f=0; f<band_number; f++){
				PURIPSI_HIGH_LOG("Channel[{}] has {} blocks and {} measurements", f, global_blocks_per_frequency[f], global_measurements_per_frequency[f]);
				for(int t=0; t<global_blocks_per_frequency[f]; t++){
				}
			}

			auto io = astroio::AstroIO();
			psi::io::IOStatus io_status = io.save_uv_data(global_uv_data, dl, pixel_size, name);

			if(io_status != psi::io::IOStatus::Success){
				PURIPSI_HIGH_LOG("Problem outputting UV data to file.");
			}
		}

		psi::mpi::finalize();

	}



	return 0;
}
