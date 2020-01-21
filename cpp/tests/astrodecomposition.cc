#include <catch2/catch.hpp>
#include <Eigen/Core>
// #include <limits>       // std::numeric_limits
#include <fstream>
#include <iostream>
#include <psi/types.h>
#include <psi/utilities.h>

#include "puripsi/types.h"
#include "psi/sort_utils.h"
#include "puripsi/astrodecomposition.h"
#include "puripsi/utilities.h"


using namespace puripsi;

TEST_CASE("Test astrodecomposition, [astrodecomposition]"){

	auto const world = psi::mpi::Communicator::World();

	SECTION("Distribute parameters"){

		bool parallel = true;

		int block_number = world.size();

		auto Decomp = puripsi::AstroDecomposition(parallel, world);

		int global_frequencies = 17;
		int global_blocks = 6;
		int global_measurements = 567;
		t_real global_pixel_size = 6.5;
		int frequencies = 0;
		int nlevels = 1;
		t_real pixel_size;
		int n_measurements;
		int n_blocks;


		if(Decomp.global_comm().is_root()){
			pixel_size = global_pixel_size;
			frequencies = global_frequencies;
			n_measurements = global_measurements;
			n_blocks = global_blocks;
		}

		Decomp.distribute_parameters_int_real(&frequencies, &pixel_size);
		Decomp.distribute_parameters_int_int(&n_blocks, &n_measurements);


		CHECK(frequencies == global_frequencies);
		CHECK(pixel_size == global_pixel_size);
		CHECK(n_blocks == global_blocks);
		CHECK(n_measurements == global_measurements);

	}


	SECTION("Distribute wideband parameters"){

		bool parallel = true;

		int block_number = world.size();

		auto Decomp = puripsi::AstroDecomposition(parallel, world);

		int global_frequencies = 17;
		int global_blocks = 6;
		int global_measurements = 567;
		t_real global_pixel_size = 6.5;
		int frequencies = 0;
		int nlevels = 1;
		t_real pixel_size;
		int n_measurements;
		int n_blocks;

		Vector<t_int> blocks_per_channel;


		if(Decomp.global_comm().is_root()){
			pixel_size = global_pixel_size;
			frequencies = global_frequencies;
			n_measurements = global_measurements;
		}

		Decomp.distribute_parameters_int_real(&frequencies, &pixel_size);

		blocks_per_channel = Vector<t_int>(frequencies);

		if(Decomp.global_comm().is_root()){
			for(int f=0; f<frequencies; f++){
				blocks_per_channel[f] = global_blocks;
			}
		}

		Decomp.distribute_parameters_wideband(blocks_per_channel);

		CHECK(frequencies == global_frequencies);
		CHECK(pixel_size == global_pixel_size);
		for(int f=0; f<frequencies; f++){
			CHECK(blocks_per_channel[f] == global_blocks);
		}

	}

	SECTION("Single Frequency"){

		bool parallel = true;

		int block_number = world.size();

		auto Decomp = puripsi::AstroDecomposition(parallel, world);

		int frequencies = 1;
		int nlevels = 1;

		std::vector<int> time_blocks = std::vector<int>(frequencies);
		time_blocks[0] = block_number;

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		wavelet_levels[0] = nlevels;

		std::vector<std::vector<int>> sub_blocks = std::vector<std::vector<int>>(1);
		sub_blocks[0] = std::vector<int>(1);
		sub_blocks[0][0] = 0;

		Decomp.decompose_primal_dual(true, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks, true);

		std::vector<std::vector<utilities::vis_params>> uv_data;

		int temp_size = 20;

		//! This test case is setup for a single frequency
		CHECK(frequencies == 1);

		if(Decomp.global_comm().is_root()){
			uv_data = std::vector<std::vector<puripsi::utilities::vis_params>> (Decomp.global_number_of_frequencies());
			for(int f=0; f<Decomp.global_number_of_frequencies(); f++){
				uv_data[f] = std::vector<puripsi::utilities::vis_params>(Decomp.frequencies()[f].number_of_time_blocks);
			}
			for(int i=0;i<world.size();i++){
				uv_data[0][i].u = Vector<t_real>(temp_size);
				uv_data[0][i].u.fill(i);
				uv_data[0][i].v = Vector<t_real>(temp_size);
				uv_data[0][i].v.fill(i);
				uv_data[0][i].w = Vector<t_real>(temp_size);
				uv_data[0][i].w.fill(i);
				uv_data[0][i].vis = Vector<t_complex>(temp_size);
				uv_data[0][i].vis.fill(std::complex<t_real>(i,i));
				uv_data[0][i].weights = Vector<t_complex>(temp_size);
				uv_data[0][i].weights.fill(std::complex<t_real>(i,(i+1)*2));
				uv_data[0][i].time = Vector<t_real>(temp_size);
				uv_data[0][i].time.fill(i);
				uv_data[0][i].baseline = Vector<t_uint>(temp_size);
				uv_data[0][i].baseline.fill(i);
				uv_data[0][i].frequencies = Vector<t_real>(temp_size);
				uv_data[0][i].frequencies.fill(i);
				uv_data[0][i].units = utilities::vis_units::pixels;
				uv_data[0][i].ra = i;
				uv_data[0][i].dec = i;
				uv_data[0][i].average_frequency = i;
				uv_data[0][i].phase_centre_x = i;
				uv_data[0][i].phase_centre_y = i;
			}
		}

		std::vector<std::vector<puripsi::utilities::vis_params>> my_uv_data(Decomp.my_number_of_frequencies());
		for(int f=0; f<Decomp.my_number_of_frequencies(); f++){
			my_uv_data[f] = std::vector<puripsi::utilities::vis_params>(Decomp.my_frequencies()[f].number_of_time_blocks);
		}

		Decomp.distribute_uv_data(uv_data, my_uv_data);

		for(int i=0; i<my_uv_data.size(); i++){
			for(int j=0;j<temp_size;j++){
				CHECK(my_uv_data[0][i].u[j] == world.rank());
				CHECK(my_uv_data[0][i].v[j] == world.rank());
				CHECK(my_uv_data[0][i].w[j] == world.rank());
				CHECK(std::real(my_uv_data[0][i].vis[j]) == world.rank());
				CHECK(std::imag(my_uv_data[0][i].vis[j]) == world.rank());
				CHECK(std::real(my_uv_data[0][i].weights[j]) == world.rank());
				CHECK(std::imag(my_uv_data[0][i].weights[j]) != world.rank()+1);
				CHECK(std::imag(my_uv_data[0][i].weights[j]) == (world.rank()+1)*2);
				CHECK(my_uv_data[0][i].time[j] == world.rank());
				CHECK(my_uv_data[0][i].baseline[j] == world.rank());
				CHECK(my_uv_data[0][i].frequencies[j] == world.rank());
			}
			CHECK(my_uv_data[0][i].units == utilities::vis_units::pixels);
			CHECK(my_uv_data[0][i].ra == world.rank());
			CHECK(my_uv_data[0][i].dec == world.rank());
			CHECK(my_uv_data[0][i].average_frequency == world.rank());
			CHECK(my_uv_data[0][i].phase_centre_x == world.rank());
			CHECK(my_uv_data[0][i].phase_centre_y == world.rank());
		}

		std::vector<std::vector<utilities::vis_params>>new_global_uv_data;

		if(Decomp.global_comm().is_root()){
			new_global_uv_data = std::vector<std::vector<puripsi::utilities::vis_params>> (Decomp.global_number_of_frequencies());
			for(int f=0; f<Decomp.global_number_of_frequencies(); f++){
				new_global_uv_data[f] = std::vector<puripsi::utilities::vis_params>(Decomp.frequencies()[f].number_of_time_blocks);
			}
		}

		Decomp.gather_uv_data(new_global_uv_data, my_uv_data);

		if(Decomp.global_comm().is_root()){

			for(int i=0; i<new_global_uv_data.size(); i++){
				for(int j=0;j<new_global_uv_data[i].size();j++){
					CHECK(new_global_uv_data[0][i].u[j] == uv_data[0][i].u[j]);
					CHECK(new_global_uv_data[0][i].v[j] == uv_data[0][i].v[j]);
					CHECK(new_global_uv_data[0][i].w[j] == uv_data[0][i].w[j]);
					CHECK(std::real(new_global_uv_data[0][i].vis[j]) == std::real(uv_data[0][i].vis[j]));
					CHECK(std::imag(new_global_uv_data[0][i].vis[j]) == std::imag(uv_data[0][i].vis[j]));
					CHECK(std::real(new_global_uv_data[0][i].weights[j]) == std::real(uv_data[0][i].weights[j]));
					CHECK(std::imag(new_global_uv_data[0][i].weights[j]) == std::imag(uv_data[0][i].weights[j]));
					CHECK(std::imag(new_global_uv_data[0][i].weights[j]) == std::imag(uv_data[0][i].weights[j]));
					CHECK(new_global_uv_data[0][i].time[j] == uv_data[0][i].time[j]);
					CHECK(new_global_uv_data[0][i].baseline[j] == uv_data[0][i].baseline[j]);
					CHECK(new_global_uv_data[0][i].frequencies[j] == uv_data[0][i].frequencies[j]);
				}
				CHECK(new_global_uv_data[0][i].units == uv_data[0][i].units);
				CHECK(new_global_uv_data[0][i].ra == uv_data[0][i].ra);
				CHECK(new_global_uv_data[0][i].dec == uv_data[0][i].dec);
				CHECK(new_global_uv_data[0][i].average_frequency ==uv_data[0][i].average_frequency);
				CHECK(new_global_uv_data[0][i].phase_centre_x == uv_data[0][i].phase_centre_x);
				CHECK(new_global_uv_data[0][i].phase_centre_y == uv_data[0][i].phase_centre_y);
			}

		}

	}

	SECTION("Multi Frequency - single block"){

		bool parallel = true;

		int block_number = 1;

		auto NextDecomp = puripsi::AstroDecomposition(parallel, world);

		int frequencies = world.size();
		int nlevels = 2;

		std::vector<int> time_blocks = std::vector<int>(frequencies);
		for(int f=0; f<frequencies; f++){
			time_blocks[f] = block_number;
		}

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		for(int f=0; f<frequencies; f++){
			wavelet_levels[f] = nlevels;
		}

		std::vector<std::vector<int>> sub_blocks = std::vector<std::vector<int>>(frequencies);
		for(int f=0; f<frequencies; f++){
			sub_blocks[f] = std::vector<int>(block_number);
			sub_blocks[f][0] = 0;
		}

		//! This test case is setup for a single block per frequency
		CHECK(block_number == 1);

		NextDecomp.decompose_primal_dual(true, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks, true);

		std::vector<std::vector<utilities::vis_params>> uv_data;

		int temp_size = 20;

		if(NextDecomp.global_comm().is_root()){
			uv_data = std::vector<std::vector<puripsi::utilities::vis_params>>(NextDecomp.global_number_of_frequencies());
			for(int f=0; f<NextDecomp.global_number_of_frequencies(); f++){
				uv_data[f] = std::vector<puripsi::utilities::vis_params>(NextDecomp.frequencies()[f].number_of_time_blocks);
			}
			for(int f=0; f<NextDecomp.global_number_of_frequencies(); f++){
				uv_data[f][0].u = Vector<t_real>(temp_size);
				uv_data[f][0].u.fill(f);
				uv_data[f][0].v = Vector<t_real>(temp_size);
				uv_data[f][0].v.fill(f);
				uv_data[f][0].w = Vector<t_real>(temp_size);
				uv_data[f][0].w.fill(f);
				uv_data[f][0].vis = Vector<t_complex>(temp_size);
				uv_data[f][0].vis.fill(std::complex<t_real>(f,f));
				uv_data[f][0].weights = Vector<t_complex>(temp_size);
				uv_data[f][0].weights.fill(std::complex<t_real>(f,(f+1)*2));
				uv_data[f][0].time = Vector<t_real>(temp_size);
				uv_data[f][0].time.fill(f);
				uv_data[f][0].baseline = Vector<t_uint>(temp_size);
				uv_data[f][0].baseline.fill(f);
				uv_data[f][0].frequencies = Vector<t_real>(temp_size);
				uv_data[f][0].frequencies.fill(f);
				uv_data[f][0].units = utilities::vis_units::pixels;
				uv_data[f][0].ra = f;
				uv_data[f][0].dec = f;
				uv_data[f][0].average_frequency = f;
				uv_data[f][0].phase_centre_x = f;
				uv_data[f][0].phase_centre_y = f;
			}
		}

		std::vector<std::vector<puripsi::utilities::vis_params>> my_uv_data(NextDecomp.my_number_of_frequencies());
		for(int f=0; f<NextDecomp.my_number_of_frequencies(); f++){
			my_uv_data[f] = std::vector<puripsi::utilities::vis_params>(NextDecomp.my_frequencies()[f].number_of_time_blocks);
		}

		NextDecomp.distribute_uv_data(uv_data, my_uv_data);

		for(int i=0; i<my_uv_data.size(); i++){
			for(int j=0;j<temp_size;j++){
				CHECK(my_uv_data[i][0].u[j] == world.rank());
				CHECK(my_uv_data[i][0].v[j] == world.rank());
				CHECK(my_uv_data[i][0].w[j] == world.rank());
				CHECK(std::real(my_uv_data[i][0].vis[j]) == world.rank());
				CHECK(std::imag(my_uv_data[i][0].vis[j]) == world.rank());
				CHECK(std::real(my_uv_data[i][0].weights[j]) == world.rank());
				CHECK(std::imag(my_uv_data[i][0].weights[j]) != world.rank()+1);
				CHECK(std::imag(my_uv_data[i][0].weights[j]) == (world.rank()+1)*2);
				CHECK(my_uv_data[i][0].time[j] == world.rank());
				CHECK(my_uv_data[i][0].baseline[j] == world.rank());
				CHECK(my_uv_data[i][0].frequencies[j] == world.rank());
			}
			CHECK(my_uv_data[i][0].units == utilities::vis_units::pixels);
			CHECK(my_uv_data[i][0].ra == world.rank());
			CHECK(my_uv_data[i][0].dec == world.rank());
			CHECK(my_uv_data[i][0].average_frequency == world.rank());
			CHECK(my_uv_data[i][0].phase_centre_x == world.rank());
			CHECK(my_uv_data[i][0].phase_centre_y == world.rank());
		}

		std::vector<std::vector<utilities::vis_params>> new_global_uv_data;

		if(NextDecomp.global_comm().is_root()){
			new_global_uv_data = std::vector<std::vector<puripsi::utilities::vis_params>>(NextDecomp.global_number_of_frequencies());
			for(int f=0; f<NextDecomp.global_number_of_frequencies(); f++){
				new_global_uv_data[f] = std::vector<puripsi::utilities::vis_params>(NextDecomp.frequencies()[f].number_of_time_blocks);
			}
		}

		NextDecomp.gather_uv_data(new_global_uv_data, my_uv_data);

		if(NextDecomp.global_comm().is_root()){

			for(int i=0; i<new_global_uv_data.size(); i++){
				for(int j=0;j<temp_size;j++){
					CHECK(new_global_uv_data[i][0].u[j] == uv_data[i][0].u[j]);
					CHECK(new_global_uv_data[i][0].v[j] == uv_data[i][0].v[j]);
					CHECK(new_global_uv_data[i][0].w[j] == uv_data[i][0].w[j]);
					CHECK(std::real(new_global_uv_data[i][0].vis[j]) == std::real(uv_data[i][0].vis[j]));
					CHECK(std::imag(new_global_uv_data[i][0].vis[j]) == std::imag(uv_data[i][0].vis[j]));
					CHECK(std::real(new_global_uv_data[i][0].weights[j]) == std::real(uv_data[i][0].weights[j]));
					CHECK(std::imag(new_global_uv_data[i][0].weights[j]) == std::imag(uv_data[i][0].weights[j]));
					CHECK(std::imag(new_global_uv_data[i][0].weights[j]) == std::imag(uv_data[i][0].weights[j]));
					CHECK(new_global_uv_data[i][0].time[j] == uv_data[i][0].time[j]);
					CHECK(new_global_uv_data[i][0].baseline[j] == uv_data[i][0].baseline[j]);
					CHECK(new_global_uv_data[i][0].frequencies[j] == uv_data[i][0].frequencies[j]);
				}
				CHECK(new_global_uv_data[i][0].units == uv_data[i][0].units);
				CHECK(new_global_uv_data[i][0].ra == uv_data[i][0].ra);
				CHECK(new_global_uv_data[i][0].dec == uv_data[i][0].dec);
				CHECK(new_global_uv_data[i][0].average_frequency ==uv_data[i][0].average_frequency);
				CHECK(new_global_uv_data[i][0].phase_centre_x == uv_data[i][0].phase_centre_x);
				CHECK(new_global_uv_data[i][0].phase_centre_y == uv_data[i][0].phase_centre_y);
			}

		}

	}

	SECTION("Frequency local int collection"){

		bool parallel = true;

		int block_number = 6;

		auto NextDecomp = puripsi::AstroDecomposition(parallel, world);

		int frequencies = world.size();
		int nlevels = 2;

		std::vector<int> time_blocks = std::vector<int>(frequencies);
		for(int f=0; f<frequencies; f++){
			time_blocks[f] = block_number;
		}

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		for(int f=0; f<frequencies; f++){
			wavelet_levels[f] = nlevels;
		}

		std::vector<std::vector<int>> sub_blocks = std::vector<std::vector<int>>(frequencies);
		for(int f=0; f<frequencies; f++){
			sub_blocks[f] = std::vector<int>(block_number);
			sub_blocks[f][0] = 0;
		}

		NextDecomp.decompose_primal_dual(true, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks, true);

		Vector<t_int> global_data(NextDecomp.global_number_of_frequencies());
		Vector<t_int> local_data(NextDecomp.my_number_of_frequencies());

		for(int f=0; f<NextDecomp.my_number_of_frequencies(); f++){
			local_data[f] = NextDecomp.global_comm().rank();
		}

		int temp_size = 20;
		NextDecomp.gather_frequency_local_vector_int(global_data, local_data);

		if(NextDecomp.global_comm().is_root()){
			for(int f=0; f<NextDecomp.global_number_of_frequencies(); f++){
				CHECK(global_data[f] == NextDecomp.frequencies()[f].global_owner);
			}
		}

	}

	SECTION("Multi Frequency - multi block"){

		bool parallel = true;

		int block_number = 6;

		auto NextDecomp = puripsi::AstroDecomposition(parallel, world);

		int frequencies = world.size();
		int nlevels = 2;

		std::vector<int> time_blocks = std::vector<int>(frequencies);
		for(int f=0; f<frequencies; f++){
			time_blocks[f] = block_number;
		}

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		for(int f=0; f<frequencies; f++){
			wavelet_levels[f] = nlevels;
		}

		std::vector<std::vector<int>> sub_blocks = std::vector<std::vector<int>>(frequencies);
		for(int f=0; f<frequencies; f++){
			sub_blocks[f] = std::vector<int>(block_number);
			sub_blocks[f][0] = 0;
		}

		NextDecomp.decompose_primal_dual(true, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks, true);


		std::vector<std::vector<utilities::vis_params>> uv_data;

		int temp_size = 20;

		if(NextDecomp.global_comm().is_root()){
			uv_data = std::vector<std::vector<puripsi::utilities::vis_params>>(NextDecomp.global_number_of_frequencies());
			for(int f=0; f<NextDecomp.global_number_of_frequencies(); f++){
				uv_data[f] = std::vector<puripsi::utilities::vis_params>(NextDecomp.frequencies()[f].number_of_time_blocks);
			}
			for(int f=0; f<NextDecomp.global_number_of_frequencies(); f++){
				for(int t=0; t<NextDecomp.frequencies()[f].number_of_time_blocks; t++){
					uv_data[f][t].u = Vector<t_real>(temp_size);
					uv_data[f][t].u.fill(f);
					uv_data[f][t].v = Vector<t_real>(temp_size);
					uv_data[f][t].v.fill(f);
					uv_data[f][t].w = Vector<t_real>(temp_size);
					uv_data[f][t].w.fill(f);
					uv_data[f][t].vis = Vector<t_complex>(temp_size);
					uv_data[f][t].vis.fill(std::complex<t_real>(f,f));
					uv_data[f][t].weights = Vector<t_complex>(temp_size);
					uv_data[f][t].weights.fill(std::complex<t_real>(f,(f+1)*2));
					uv_data[f][t].time = Vector<t_real>(temp_size);
					uv_data[f][t].time.fill(f);
					uv_data[f][t].baseline = Vector<t_uint>(temp_size);
					uv_data[f][t].baseline.fill(f);
					uv_data[f][t].frequencies = Vector<t_real>(temp_size);
					uv_data[f][t].frequencies.fill(f);
					uv_data[f][t].units = utilities::vis_units::pixels;
					uv_data[f][t].ra = f;
					uv_data[f][t].dec = f;
					uv_data[f][t].average_frequency = f;
					uv_data[f][t].phase_centre_x = f;
					uv_data[f][t].phase_centre_y = f;
				}
			}
		}
		std::vector<std::vector<puripsi::utilities::vis_params>> my_uv_data(NextDecomp.my_number_of_frequencies());
		for(int f=0; f<NextDecomp.my_number_of_frequencies(); f++){
			my_uv_data[f] = std::vector<puripsi::utilities::vis_params>(NextDecomp.my_frequencies()[f].number_of_time_blocks);
		}

		NextDecomp.distribute_uv_data(uv_data, my_uv_data);

		for(int i=0; i<my_uv_data.size(); i++){
			for(int t=0; t<my_uv_data[i].size(); t++){
				for(int j=0;j<temp_size;j++){
					CHECK(my_uv_data[i][t].u[j] == world.rank());
					CHECK(my_uv_data[i][t].v[j] == world.rank());
					CHECK(my_uv_data[i][t].w[j] == world.rank());
					CHECK(std::real(my_uv_data[i][t].vis[j]) == world.rank());
					CHECK(std::imag(my_uv_data[i][t].vis[j]) == world.rank());
					CHECK(std::real(my_uv_data[i][t].weights[j]) == world.rank());
					CHECK(std::imag(my_uv_data[i][t].weights[j]) != world.rank()+1);
					CHECK(std::imag(my_uv_data[i][t].weights[j]) == (world.rank()+1)*2);
					CHECK(my_uv_data[i][t].time[j] == world.rank());
					CHECK(my_uv_data[i][t].baseline[j] == world.rank());
					CHECK(my_uv_data[i][t].frequencies[j] == world.rank());
				}
				CHECK(my_uv_data[i][t].units == utilities::vis_units::pixels);
				CHECK(my_uv_data[i][t].ra == world.rank());
				CHECK(my_uv_data[i][t].dec == world.rank());
				CHECK(my_uv_data[i][t].average_frequency == world.rank());
				CHECK(my_uv_data[i][t].phase_centre_x == world.rank());
				CHECK(my_uv_data[i][t].phase_centre_y == world.rank());
			}
		}

		std::vector<std::vector<utilities::vis_params>> new_global_uv_data;

		if(NextDecomp.global_comm().is_root()){
			new_global_uv_data = std::vector<std::vector<puripsi::utilities::vis_params>>(NextDecomp.global_number_of_frequencies());
			for(int f=0; f<NextDecomp.global_number_of_frequencies(); f++){
				new_global_uv_data[f] = std::vector<puripsi::utilities::vis_params>(NextDecomp.frequencies()[f].number_of_time_blocks);
			}
		}

		NextDecomp.gather_uv_data(new_global_uv_data, my_uv_data);

		if(NextDecomp.global_comm().is_root()){

			for(int i=0; i<new_global_uv_data.size(); i++){
				CHECK(new_global_uv_data[i].size() == uv_data[i].size());
				for(int t=0; t<new_global_uv_data[i].size(); t++){
					for(int j=0;j<temp_size;j++){
						CHECK(new_global_uv_data[i][t].u[j] == uv_data[i][t].u[j]);
						CHECK(new_global_uv_data[i][t].v[j] == uv_data[i][t].v[j]);
						CHECK(new_global_uv_data[i][t].w[j] == uv_data[i][t].w[j]);
						CHECK(std::real(new_global_uv_data[i][t].vis[j]) == std::real(uv_data[i][t].vis[j]));
						CHECK(std::imag(new_global_uv_data[i][t].vis[j]) == std::imag(uv_data[i][t].vis[j]));
						CHECK(std::real(new_global_uv_data[i][t].weights[j]) == std::real(uv_data[i][t].weights[j]));
						CHECK(std::imag(new_global_uv_data[i][t].weights[j]) == std::imag(uv_data[i][t].weights[j]));
						CHECK(std::imag(new_global_uv_data[i][t].weights[j]) == std::imag(uv_data[i][t].weights[j]));
						CHECK(new_global_uv_data[i][t].time[j] == uv_data[i][t].time[j]);
						CHECK(new_global_uv_data[i][t].baseline[j] == uv_data[i][t].baseline[j]);
						CHECK(new_global_uv_data[i][t].frequencies[j] == uv_data[i][t].frequencies[j]);
					}
					CHECK(new_global_uv_data[i][t].units == uv_data[i][t].units);
					CHECK(new_global_uv_data[i][t].ra == uv_data[i][t].ra);
					CHECK(new_global_uv_data[i][t].dec == uv_data[i][t].dec);
					CHECK(new_global_uv_data[i][t].average_frequency == uv_data[i][t].average_frequency);
					CHECK(new_global_uv_data[i][t].phase_centre_x == uv_data[i][t].phase_centre_x);
					CHECK(new_global_uv_data[i][t].phase_centre_y == uv_data[i][t].phase_centre_y);
				}
			}
		}

	}


	SECTION("Dirty image collection"){

		bool parallel = true;

		int block_number = 1;

		auto NextDecomp = puripsi::AstroDecomposition(parallel, world);

		int frequencies = world.size();
		int nlevels = 2;

		std::vector<int> time_blocks = std::vector<int>(frequencies);
		for(int f=0; f<frequencies; f++){
			time_blocks[f] = block_number;
		}

		std::vector<t_int> wavelet_levels = std::vector<t_int>(frequencies);
		for(int f=0; f<frequencies; f++){
			wavelet_levels[f] = nlevels;
		}

		std::vector<std::vector<int>> sub_blocks = std::vector<std::vector<int>>(frequencies);
		for(int f=0; f<frequencies; f++){
			sub_blocks[f] = std::vector<int>(block_number);
			sub_blocks[f][0] = 0;
		}

		NextDecomp.decompose_primal_dual(true, true, false, false, false, frequencies, wavelet_levels, time_blocks, sub_blocks, true);

		int size_of_image = 45;

		std::vector<Vector<t_complex>> dirty(NextDecomp.my_number_of_frequencies());
		for(int f=0; f<NextDecomp.my_number_of_frequencies(); f++){
			dirty[f] = Vector<t_complex>(size_of_image);
			dirty[f].fill(NextDecomp.global_comm().rank()+1);
		}
		std::vector<Vector<t_complex>> global_dirty(NextDecomp.global_number_of_frequencies());
		if(NextDecomp.global_comm().is_root()){
			for(int f=0; f<NextDecomp.global_number_of_frequencies(); f++){
				global_dirty[f] = Vector<t_complex>(size_of_image);
			}
		}

		NextDecomp.collect_dirty_image(dirty, global_dirty);

		if(NextDecomp.global_comm().is_root()){
			for(int f=0; f<NextDecomp.global_number_of_frequencies(); f++){
				for(int i=0; i<size_of_image; i++){
					CHECK(global_dirty[f][i].real() == f+1);
				}
			}
		}
	}

}
