#include <catch2/catch.hpp>
#include <Eigen/Core>
// #include <limits>       // std::numeric_limits
#include <fstream>
#include <iostream>
#include <psi/types.h>
#include <psi/utilities.h>

#include "puripsi/types.h"
#include "psi/sort_utils.h"
#include "puripsi/astroio.h"
#include "puripsi/utilities.h"


using namespace puripsi;

TEST_CASE("Test astroio, [astroio]"){

	SECTION("Load and save uv_data"){

		auto io = astroio::AstroIO();
		std::string filename = "restart.dat";

		int frequencies = 17;
		int blocks = 6;
		int array_size = 43;
		int new_frequencies = -1;
		Vector<t_int> new_blocks;
		t_real ra = 2.7;
		t_real dec = 5.9;
		t_real average_frequency = 75.25;
		t_int phase_centre_x = 8;
		t_int phase_centre_y = 4;
		t_real pixel_size = 2.1494;
		t_real dl = 4.0;
		t_real new_pixel_size = -1.0;
		t_real new_dl = -1.0;

		std::vector<std::vector<utilities::vis_params>> uv_data(frequencies);
		for(int f=0; f<frequencies; f++){
			uv_data[f] = std::vector<utilities::vis_params>(blocks);
			for(int t=0; t<blocks; t++){
				uv_data[f][t].u = Vector<t_real>(array_size);
				uv_data[f][t].u.fill(std::real(f+t));
				uv_data[f][t].v = Vector<t_real>(array_size);
				uv_data[f][t].v.fill(std::real(f+t));
				uv_data[f][t].w = Vector<t_real>(array_size);
				uv_data[f][t].w.fill(std::real(f+t));
				uv_data[f][t].vis = Vector<t_complex>(array_size);
				uv_data[f][t].vis.fill(std::complex<t_real>(f,t));
				uv_data[f][t].weights = Vector<t_complex>(array_size);
				uv_data[f][t].weights.fill(std::complex<t_real>(f,t));
				uv_data[f][t].time = Vector<t_real>(array_size);
				uv_data[f][t].time.fill(std::real(f*t));
				uv_data[f][t].baseline = Vector<t_uint>(array_size);
				uv_data[f][t].baseline.fill(f+t);
				uv_data[f][t].frequencies = Vector<t_real>(array_size);
				uv_data[f][t].frequencies.fill((f+t)*2);
				uv_data[f][t].units = utilities::vis_units::pixels;
				uv_data[f][t].ra = ra;
				uv_data[f][t].dec = dec;
				uv_data[f][t].average_frequency = average_frequency;
				uv_data[f][t].phase_centre_x = phase_centre_x;
				uv_data[f][t].phase_centre_y = phase_centre_y;
			}
		}


		psi::io::IOStatus io_status = io.save_uv_data(uv_data, dl, pixel_size, filename);

		CHECK(io_status ==  psi::io::IOStatus::Success);

		io_status = io.load_uv_data_header(new_frequencies, new_blocks, new_dl, new_pixel_size, filename);

		CHECK(io_status ==  psi::io::IOStatus::Success);
		CHECK(new_frequencies == frequencies);
		CHECK(new_blocks.size() == frequencies);
		for(int f=0; f<frequencies; f++){
			CHECK(new_blocks[f] == blocks);
		}
		CHECK(new_dl == dl);
		CHECK(new_pixel_size == pixel_size);

		std::vector<std::vector<utilities::vis_params>> new_uv_data(frequencies);

		for(int f=0; f<frequencies; f++){
			new_uv_data[f] = std::vector<utilities::vis_params>(blocks);
		}

		io_status = io.load_uv_data(new_uv_data, filename);

		CHECK(io_status ==  psi::io::IOStatus::Success);

		for(int f=0; f<frequencies; f++){
			for(int t=0; t<blocks; t++){
				CHECK(new_uv_data[f][t].u.size() != 0);
				CHECK(uv_data[f][t].u.size() != 0);
				CHECK(new_uv_data[f][t].u.size() == uv_data[f][t].u.size());
				for(int i=0; i<uv_data[f][t].u.size(); i++){
					CHECK(uv_data[f][t].u[i] == new_uv_data[f][t].u[i]);
				}
				CHECK(new_uv_data[f][t].v.size() != 0);
				CHECK(uv_data[f][t].v.size() != 0);
				CHECK(new_uv_data[f][t].v.size() == uv_data[f][t].v.size());
				for(int i=0; i<uv_data[f][t].v.size(); i++){
					CHECK(uv_data[f][t].v[i] == new_uv_data[f][t].v[i]);
				}
				CHECK(new_uv_data[f][t].w.size() != 0);
				CHECK(uv_data[f][t].w.size() != 0);
				CHECK(new_uv_data[f][t].w.size() == uv_data[f][t].w.size());
				for(int i=0; i<uv_data[f][t].w.size(); i++){
					CHECK(uv_data[f][t].w[i] == new_uv_data[f][t].w[i]);
				}
				CHECK(new_uv_data[f][t].vis.size() != 0);
				CHECK(uv_data[f][t].vis.size() != 0);
				CHECK(new_uv_data[f][t].vis.size() == uv_data[f][t].vis.size());
				for(int i=0; i<uv_data[f][t].vis.size(); i++){
					CHECK(uv_data[f][t].vis[i] == new_uv_data[f][t].vis[i]);
				}
				CHECK(new_uv_data[f][t].weights.size() != 0);
				CHECK(uv_data[f][t].weights.size() != 0);
				CHECK(new_uv_data[f][t].weights.size() == uv_data[f][t].weights.size());
				for(int i=0; i<uv_data[f][t].weights.size(); i++){
					CHECK(uv_data[f][t].weights[i] == new_uv_data[f][t].weights[i]);
				}
				CHECK(new_uv_data[f][t].time.size() != 0);
				CHECK(uv_data[f][t].time.size() != 0);
				CHECK(new_uv_data[f][t].time.size() == uv_data[f][t].time.size());
				for(int i=0; i<uv_data[f][t].time.size(); i++){
					CHECK(uv_data[f][t].time[i] == new_uv_data[f][t].time[i]);
				}
				CHECK(new_uv_data[f][t].baseline.size() != 0);
				CHECK(uv_data[f][t].baseline.size() != 0);
				CHECK(new_uv_data[f][t].baseline.size() == uv_data[f][t].baseline.size());
				for(int i=0; i<uv_data[f][t].baseline.size(); i++){
					CHECK(uv_data[f][t].baseline[i] == new_uv_data[f][t].baseline[i]);
				}
				CHECK(new_uv_data[f][t].frequencies.size() != 0);
				CHECK(uv_data[f][t].frequencies.size() != 0);
				CHECK(new_uv_data[f][t].frequencies.size() == uv_data[f][t].frequencies.size());
				for(int i=0; i<uv_data[f][t].frequencies.size(); i++){
					CHECK(uv_data[f][t].frequencies[i] == new_uv_data[f][t].frequencies[i]);
				}
				CHECK(uv_data[f][t].units == new_uv_data[f][t].units);
				CHECK(uv_data[f][t].ra == new_uv_data[f][t].ra);
				CHECK(uv_data[f][t].dec == new_uv_data[f][t].dec);
				CHECK(uv_data[f][t].average_frequency == new_uv_data[f][t].average_frequency);
				CHECK(uv_data[f][t].phase_centre_x == new_uv_data[f][t].phase_centre_x);
				CHECK(uv_data[f][t].phase_centre_y == new_uv_data[f][t].phase_centre_y);
			}
		}

		char char_filename[filename.size()+1];
		filename.copy(char_filename, filename.size()+1);
		// Delete the checkpoint test file
		auto status = remove(char_filename);
		CHECK(status == 0);

	}
}

