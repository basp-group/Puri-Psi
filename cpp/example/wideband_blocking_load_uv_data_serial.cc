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

	t_real dl;
	t_real pixel_size;
	t_int band_number;
	Vector<t_int> blocks;

	std::string name = argc >= 2 ? argv[1] : "uv_data.dat";

	if(argc > 2) {
		std::cout << "Usage:\n"
				"$ "
				<< argv[0] <<
				"- input: name of input file\n\n";
		exit(0);
	}

	auto io = astroio::AstroIO();

	psi::io::IOStatus io_status = io.load_uv_data_header(band_number, blocks, dl, pixel_size, name);
	if(io_status != psi::io::IOStatus::Success){
		PURIPSI_ERROR("Problem loading uv data file header information, quitting");
	}else{
		PURIPSI_HIGH_LOG("Frequencies {} dl {} pixel size {}", band_number, dl, pixel_size);
		for(int f=0; f<band_number; f++){
			PURIPSI_HIGH_LOG("Channel[{}] has {} blocks", f, blocks[f]);
		}
	}

	std::vector<std::vector<utilities::vis_params>> uv_data(band_number);

	for(int f=0; f<band_number; f++){
		uv_data[f] = std::vector<utilities::vis_params>(blocks[f]);
	}

	io_status = io.load_uv_data(uv_data, name);

	if(io_status != psi::io::IOStatus::Success){
		PURIPSI_HIGH_LOG("Problem loading UV data from file.");
	}else{
		for(int f=0; f<band_number; f++){
			for(int t=0; t<blocks[f]; t++){
				PURIPSI_HIGH_LOG("Block[{}][{}] has {} measurements", f, t,  uv_data[f][t].u.size());
			}
		}
	}

	return 0;
}
