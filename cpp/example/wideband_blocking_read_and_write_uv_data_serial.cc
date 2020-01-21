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

using namespace puripsi;
using namespace puripsi::notinstalled;

int main(int argc, const char **argv) {
	psi::logging::initialize();
	puripsi::logging::initialize();
	psi::logging::set_level("critical");
	puripsi::logging::set_level("critical");

	t_real dl = 1.75;
	t_real pixel_size = 0.04;



	std::vector<std::string> dataName{
		//	"/home/nx01/nx01/adrianj/compress/puri-psi-dev/build/cpp/example/wideband_ms/CYG-A-5-8M10S.MS",
		//	"/home/nx01/nx01/adrianj/compress/puri-psi-dev/build/cpp/example/wideband_ms/CYG-A-7-8M10S.MS",
		"/home/nx01/nx01/adrianj/compress/puri-psi-dev/build/cpp/example/wideband_ms/CYG-C-5-8M10S.MS",
		"/home/nx01/nx01/adrianj/compress/puri-psi-dev/build/cpp/example/wideband_ms/CYG-C-7-8M10S.MS",
	};

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



	std::vector<std::vector<puripsi::utilities::vis_params>> global_uv_data;

	Vector<t_int> frequencies_per_dataset(n_datasets);
	//! We are passing in a data set at a time to to the file read function so we only want to have n_blocks_per_dataset to be a single
	//! element vector.
	Vector<t_int> n_blocks_per_dataset(1);
	Vector<t_int> blocks_per_frequency;
	Vector<t_int> measurements_per_frequency;
	Vector<std::set<::casacore::uInt>> spws_ids(n_datasets);

	for(int i=0; i<n_datasets; i++){
		puripsi::extract_number_of_channels_and_spectal_windows(dataName[i], field_id, frequencies_per_dataset[i], spws_ids[i]);
		band_number += frequencies_per_dataset[i];
		PURIPSI_HIGH_LOG("Number of Spectral Windows {}",spws_ids[i].size());
		for(auto f : spws_ids[i]) {
			PURIPSI_HIGH_LOG("Spectral Windows {}",f);
		}
	}
	blocks_per_frequency = Vector<t_int>(band_number);
	measurements_per_frequency = Vector<t_int>(band_number);

	global_uv_data = std::vector<std::vector<puripsi::utilities::vis_params>>(band_number);
	int freq_number = 0;
	for(int i=0; i<n_datasets; i++){
		PURIPSI_HIGH_LOG("Reading in file {}",dataName[i]);
		for(auto f : spws_ids[i]) {
			PURIPSI_HIGH_LOG("Calculated number of frequencies {}",frequencies_per_dataset[i]/spws_ids[i].size());
			for(int j=0; j<frequencies_per_dataset[i]/spws_ids[i].size(); j++){
				PURIPSI_HIGH_LOG("Spectral Window {} Frequency {}",f,j);
				std::vector<std::string> temp_file(1);
				temp_file[0] = dataName[i];
				global_uv_data[freq_number] = get_time_blocks_multi_file_spectral_window(temp_file, &dl, &pixel_size, &n_blocks, &n_measurements, f, j, n_blocks_per_dataset, field_id);
				PURIPSI_HIGH_LOG("Final number of blocks for channel {} is {} and measurements is {}",freq_number,n_blocks,n_measurements);
				blocks_per_frequency[freq_number] = n_blocks;
				measurements_per_frequency[freq_number] = n_measurements;
				freq_number++;
			}
		}
	}
	PURIPSI_HIGH_LOG("Number of channels is {} ", band_number);
	for(int f=0; f<band_number; f++){
		PURIPSI_HIGH_LOG("Channel[{}] has {} blocks", f, blocks_per_frequency[f]);
	}

	auto io = astroio::AstroIO();
	psi::io::IOStatus io_status = io.save_uv_data(global_uv_data, dl, pixel_size, name);

	if(io_status != psi::io::IOStatus::Success){
		PURIPSI_HIGH_LOG("Problem outputting UV data to file.");
	}

	return 0;
}
