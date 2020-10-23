#include "puripsi/astrodecomposition.h"

namespace puripsi {


void AstroDecomposition::distribute_uv_data(const std::vector<std::vector<utilities::vis_params>> &uv_data, std::vector<std::vector<utilities::vis_params>> &my_uv_data){

	if(decomp_.parallel_mpi){

		//! If I am the root process (the owner of all the data at the moment) then send data to the relevant process
		if(decomp_.global_comm.is_root()){
			int my_freq_index = 0;
			bool freq_used = false;
			for(int f=0; f<decomp_.number_of_frequencies; f++){
				int my_block_index = 0;
				for(int t=0; t<decomp_.frequencies[f].number_of_time_blocks; t++){
					//! If this is data for the root process then just copy it to the new data structure, do not send.
					if(decomp_.frequencies[f].time_blocks[t].global_owner == decomp_.global_comm.rank()){
						my_uv_data[my_freq_index][my_block_index].u = uv_data[f][t].u;
						my_uv_data[my_freq_index][my_block_index].v = uv_data[f][t].v;
						my_uv_data[my_freq_index][my_block_index].w = uv_data[f][t].w;
						my_uv_data[my_freq_index][my_block_index].vis = uv_data[f][t].vis;
						my_uv_data[my_freq_index][my_block_index].weights = uv_data[f][t].weights;
						my_uv_data[my_freq_index][my_block_index].time = uv_data[f][t].time;
						my_uv_data[my_freq_index][my_block_index].baseline = uv_data[f][t].baseline;
						my_uv_data[my_freq_index][my_block_index].frequencies = uv_data[f][t].frequencies;
						my_uv_data[my_freq_index][my_block_index].units = uv_data[f][t].units;
						my_uv_data[my_freq_index][my_block_index].ra = uv_data[f][t].ra;
						my_uv_data[my_freq_index][my_block_index].dec = uv_data[f][t].dec;
						my_uv_data[my_freq_index][my_block_index].average_frequency = uv_data[f][t].average_frequency;
						my_uv_data[my_freq_index][my_block_index].phase_centre_x = uv_data[f][t].phase_centre_x;
						my_uv_data[my_freq_index][my_block_index].phase_centre_y = uv_data[f][t].phase_centre_y;
						my_block_index++;
						freq_used = true;
						//! Otherwise, send to the owning process
					}else{
						t_uint tag = 0;
						int temp_size = uv_data[f][t].u.size();
						decomp_.global_comm.send_single(temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_eigen(uv_data[f][t].u, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						temp_size = uv_data[f][t].v.size();
						decomp_.global_comm.send_single(temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_eigen(uv_data[f][t].v, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						temp_size = uv_data[f][t].w.size();
						decomp_.global_comm.send_single(temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_eigen(uv_data[f][t].w, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						temp_size = uv_data[f][t].vis.size();
						decomp_.global_comm.send_single(temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_eigen(uv_data[f][t].vis, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						temp_size = uv_data[f][t].weights.size();
						decomp_.global_comm.send_single(temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_eigen(uv_data[f][t].weights, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						temp_size = uv_data[f][t].time.size();
						decomp_.global_comm.send_single(temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_eigen(uv_data[f][t].time, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						temp_size = uv_data[f][t].baseline.size();
						decomp_.global_comm.send_single(temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_eigen(uv_data[f][t].baseline, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						temp_size = uv_data[f][t].frequencies.size();
						decomp_.global_comm.send_single(temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_eigen(uv_data[f][t].frequencies, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						int temp_units = static_cast<int>(uv_data[f][t].units);
						decomp_.global_comm.send_single(temp_units, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_single(uv_data[f][t].ra, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_single(uv_data[f][t].dec, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_single(uv_data[f][t].average_frequency, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_single(uv_data[f][t].phase_centre_x, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_single(uv_data[f][t].phase_centre_y, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
					}
				}
				if(freq_used){
					freq_used = false;
					my_freq_index++;
				}
			}
			//! If I am not the root process then wait to receive the data I am expecting.
		}else{
			for(int f=0; f<my_decomp_.number_of_frequencies; f++){
				for(int t=0; t<my_decomp_.frequencies[f].number_of_time_blocks; t++){
					t_uint tag = 0;
					int temp_size;
					decomp_.global_comm.recv_single(&temp_size, decomp_.global_comm.root_id(), tag);
					my_uv_data[f][t].u = Vector<t_real>(temp_size);
					decomp_.global_comm.recv_eigen(my_uv_data[f][t].u, decomp_.global_comm.root_id(), tag);
					decomp_.global_comm.recv_single(&temp_size, decomp_.global_comm.root_id(), tag);
					my_uv_data[f][t].v = Vector<t_real>(temp_size);
					decomp_.global_comm.recv_eigen(my_uv_data[f][t].v, decomp_.global_comm.root_id(), tag);
					decomp_.global_comm.recv_single(&temp_size, decomp_.global_comm.root_id(), tag);
					my_uv_data[f][t].w = Vector<t_real>(temp_size);
					decomp_.global_comm.recv_eigen(my_uv_data[f][t].w, decomp_.global_comm.root_id(), tag);
					decomp_.global_comm.recv_single(&temp_size, decomp_.global_comm.root_id(), tag);
					my_uv_data[f][t].vis = Vector<t_complex>(temp_size);
					decomp_.global_comm.recv_eigen(my_uv_data[f][t].vis, decomp_.global_comm.root_id(), tag);
					decomp_.global_comm.recv_single(&temp_size, decomp_.global_comm.root_id(), tag);
					my_uv_data[f][t].weights = Vector<t_complex>(temp_size);
					decomp_.global_comm.recv_eigen(my_uv_data[f][t].weights, decomp_.global_comm.root_id(), tag);
					decomp_.global_comm.recv_single(&temp_size, decomp_.global_comm.root_id(), tag);
					my_uv_data[f][t].time = Vector<t_real>(temp_size);
					decomp_.global_comm.recv_eigen(my_uv_data[f][t].time, decomp_.global_comm.root_id(), tag);
					decomp_.global_comm.recv_single(&temp_size, decomp_.global_comm.root_id(), tag);
					my_uv_data[f][t].baseline = Vector<t_uint>(temp_size);
					decomp_.global_comm.recv_eigen(my_uv_data[f][t].baseline, decomp_.global_comm.root_id(), tag);
					decomp_.global_comm.recv_single(&temp_size, decomp_.global_comm.root_id(), tag);
					my_uv_data[f][t].frequencies = Vector<t_real>(temp_size);
					decomp_.global_comm.recv_eigen(my_uv_data[f][t].frequencies, decomp_.global_comm.root_id(), tag);
					int units;
					decomp_.global_comm.recv_single(&units, decomp_.global_comm.root_id(), tag);
					switch(static_cast<utilities::vis_units>(units)){
					case utilities::vis_units::lambda:
						break;
					case utilities::vis_units::radians:
						break;
					case utilities::vis_units::pixels:
						break;
					default:
						PSI_ERROR("Problem with distribute_uv_data, wrong units distributed");
					}
					my_uv_data[f][t].units = static_cast<utilities::vis_units>(units);
					decomp_.global_comm.recv_single(&my_uv_data[f][t].ra, decomp_.global_comm.root_id(), tag);
					decomp_.global_comm.recv_single(&my_uv_data[f][t].dec, decomp_.global_comm.root_id(), tag);
					decomp_.global_comm.recv_single(&my_uv_data[f][t].average_frequency, decomp_.global_comm.root_id(), tag);
					decomp_.global_comm.recv_single(&my_uv_data[f][t].phase_centre_x, decomp_.global_comm.root_id(), tag);
					decomp_.global_comm.recv_single(&my_uv_data[f][t].phase_centre_y, decomp_.global_comm.root_id(), tag);
				}
			}

		}
	}
	return;
}

void AstroDecomposition::gather_uv_data(std::vector<std::vector<utilities::vis_params>> &uv_data, const std::vector<std::vector<utilities::vis_params>> &my_uv_data){

	if(decomp_.parallel_mpi){

		//! If I am the root process (the owner of all the data at the moment) then receive data from owning processes
		if(decomp_.global_comm.is_root()){
			int my_freq_index = 0;
			bool freq_used = false;
			for(int f=0; f<decomp_.number_of_frequencies; f++){
				int my_block_index = 0;
				for(int t=0; t<decomp_.frequencies[f].number_of_time_blocks; t++){
					//! If this is data for the root process then just copy it to the new data structure, do not send.
					if(decomp_.frequencies[f].time_blocks[t].global_owner == decomp_.global_comm.rank()){
						uv_data[f][t].u = my_uv_data[my_freq_index][my_block_index].u;
						uv_data[f][t].v = my_uv_data[my_freq_index][my_block_index].v;
						uv_data[f][t].w = my_uv_data[my_freq_index][my_block_index].w;
						uv_data[f][t].vis = my_uv_data[my_freq_index][my_block_index].vis;
						uv_data[f][t].weights = my_uv_data[my_freq_index][my_block_index].weights;
						uv_data[f][t].time = my_uv_data[my_freq_index][my_block_index].time;
						uv_data[f][t].baseline = my_uv_data[my_freq_index][my_block_index].baseline;
						uv_data[f][t].frequencies = my_uv_data[my_freq_index][my_block_index].frequencies;
						uv_data[f][t].units = my_uv_data[my_freq_index][my_block_index].units;
						uv_data[f][t].ra = my_uv_data[my_freq_index][my_block_index].ra;
						uv_data[f][t].dec = my_uv_data[my_freq_index][my_block_index].dec;
						uv_data[f][t].average_frequency = my_uv_data[my_freq_index][my_block_index].average_frequency;
						uv_data[f][t].phase_centre_x = my_uv_data[my_freq_index][my_block_index].phase_centre_x;
						uv_data[f][t].phase_centre_y = my_uv_data[my_freq_index][my_block_index].phase_centre_y;
						my_block_index++;
						freq_used = true;
						//! Otherwise, received from the owning process
					}else{
						t_uint tag = 0;
						int temp_size;
						decomp_.global_comm.recv_single(&temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						uv_data[f][t].u = Vector<t_real>(temp_size);
						decomp_.global_comm.recv_eigen(uv_data[f][t].u, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.recv_single(&temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						uv_data[f][t].v = Vector<t_real>(temp_size);
						decomp_.global_comm.recv_eigen(uv_data[f][t].v, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.recv_single(&temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						uv_data[f][t].w = Vector<t_real>(temp_size);
						decomp_.global_comm.recv_eigen(uv_data[f][t].w, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.recv_single(&temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						uv_data[f][t].vis = Vector<t_complex>(temp_size);
						decomp_.global_comm.recv_eigen(uv_data[f][t].vis, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.recv_single(&temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						uv_data[f][t].weights = Vector<t_complex>(temp_size);
						decomp_.global_comm.recv_eigen(uv_data[f][t].weights, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.recv_single(&temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						uv_data[f][t].time = Vector<t_real>(temp_size);
						decomp_.global_comm.recv_eigen(uv_data[f][t].time, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.recv_single(&temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						uv_data[f][t].baseline = Vector<t_uint>(temp_size);
						decomp_.global_comm.recv_eigen(uv_data[f][t].baseline, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.recv_single(&temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						uv_data[f][t].frequencies = Vector<t_real>(temp_size);
						decomp_.global_comm.recv_eigen(uv_data[f][t].frequencies, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						int units;
						decomp_.global_comm.recv_single(&units, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						switch(static_cast<utilities::vis_units>(units)){
						case utilities::vis_units::lambda:
							break;
						case utilities::vis_units::radians:
							break;
						case utilities::vis_units::pixels:
							break;
						default:
							PSI_ERROR("Problem with distribute_uv_data, wrong units distributed");
						}
						uv_data[f][t].units = static_cast<utilities::vis_units>(units);
						decomp_.global_comm.recv_single(&uv_data[f][t].ra, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.recv_single(&uv_data[f][t].dec, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.recv_single(&uv_data[f][t].average_frequency, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.recv_single(&uv_data[f][t].phase_centre_x, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.recv_single(&uv_data[f][t].phase_centre_y, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
					}
				}
				if(freq_used){
					freq_used = false;
					my_freq_index++;
				}
			}
			//! If I am not the root process then send data to the root.
		}else{
			int my_freq_index = 0;
			bool freq_used = false;
			for(int f=0; f<decomp_.number_of_frequencies; f++){
				int my_block_index = 0;
				for(int t=0; t<decomp_.frequencies[f].number_of_time_blocks; t++){
					if(decomp_.frequencies[f].time_blocks[t].global_owner == decomp_.global_comm.rank()){
						t_uint tag = 0;
						int temp_size = my_uv_data[my_freq_index][my_block_index].u.size();
						decomp_.global_comm.send_single(temp_size, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_eigen(my_uv_data[my_freq_index][my_block_index].u, decomp_.global_comm.root_id(), tag);
						temp_size = my_uv_data[my_freq_index][my_block_index].v.size();
						decomp_.global_comm.send_single(temp_size, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_eigen(my_uv_data[my_freq_index][my_block_index].v, decomp_.global_comm.root_id(), tag);
						temp_size = my_uv_data[my_freq_index][my_block_index].w.size();
						decomp_.global_comm.send_single(temp_size, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_eigen(my_uv_data[my_freq_index][my_block_index].w, decomp_.global_comm.root_id(), tag);
						temp_size = my_uv_data[my_freq_index][my_block_index].vis.size();
						decomp_.global_comm.send_single(temp_size, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_eigen(my_uv_data[my_freq_index][my_block_index].vis, decomp_.global_comm.root_id(), tag);
						temp_size = my_uv_data[my_freq_index][my_block_index].weights.size();
						decomp_.global_comm.send_single(temp_size, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_eigen(my_uv_data[my_freq_index][my_block_index].weights, decomp_.global_comm.root_id(), tag);
						temp_size = my_uv_data[my_freq_index][my_block_index].time.size();
						decomp_.global_comm.send_single(temp_size, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_eigen(my_uv_data[my_freq_index][my_block_index].time, decomp_.global_comm.root_id(), tag);
						temp_size = my_uv_data[my_freq_index][my_block_index].baseline.size();
						decomp_.global_comm.send_single(temp_size, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_eigen(my_uv_data[my_freq_index][my_block_index].baseline, decomp_.global_comm.root_id(), tag);
						temp_size = my_uv_data[my_freq_index][my_block_index].frequencies.size();
						decomp_.global_comm.send_single(temp_size, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_eigen(my_uv_data[my_freq_index][my_block_index].frequencies, decomp_.global_comm.root_id(), tag);
						int temp_units = static_cast<int>(my_uv_data[my_freq_index][my_block_index].units);
						decomp_.global_comm.send_single(temp_units, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_single(my_uv_data[my_freq_index][my_block_index].ra, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_single(my_uv_data[my_freq_index][my_block_index].dec, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_single(my_uv_data[my_freq_index][my_block_index].average_frequency, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_single(my_uv_data[my_freq_index][my_block_index].phase_centre_x, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_single(my_uv_data[my_freq_index][my_block_index].phase_centre_y, decomp_.global_comm.root_id(), tag);
						my_block_index++;
						freq_used = true;
					}
				}
				if(freq_used){
					freq_used = false;
					my_freq_index++;
				}
			}

		}
	}
	return;
}




void AstroDecomposition::gather_frequency_local_vector_int(Vector<t_int> &global_data, const Vector<t_int> &local_data){

	if(decomp_.parallel_mpi){

		//! If I am the root process (the owner of all the data at the moment) then receive data from owning processes
		if(decomp_.global_comm.is_root()){
			int my_freq_index = 0;
			for(int f=0; f<decomp_.number_of_frequencies; f++){
				if(own_this_frequency(f)){
					global_data[f] = local_data[my_freq_index];
					my_freq_index++;
					//! Otherwise, received from the owning process
				}else{
					t_uint tag = 0;
					decomp_.global_comm.recv_single(&global_data[f], decomp_.frequencies[f].global_owner, tag);
				}
			}
			//! If I am not the root process then send data to the root.
		}else{
			int my_freq_index = 0;
			for(int f=0; f<decomp_.number_of_frequencies; f++){
				if(own_this_frequency(f)){
					t_uint tag = 0;
					decomp_.global_comm.send_single(local_data[my_freq_index], decomp_.global_comm.root_id(), tag);
					my_freq_index++;
				}
			}
		}
	}
	return;
}

void AstroDecomposition::distribute_parameters_int_real(int *channel_index, t_real *pixel_size){

	if(decomp_.parallel_mpi){
		*channel_index = decomp_.global_comm.broadcast(*channel_index, decomp_.global_comm.root_id());
		*pixel_size = decomp_.global_comm.broadcast(*pixel_size, decomp_.global_comm.root_id());
	}
	return;
}

void AstroDecomposition::distribute_parameters_int_int(t_int *n_blocks, t_int *n_measurements){

	if(decomp_.parallel_mpi){
		*n_blocks = decomp_.global_comm.broadcast(*n_blocks, decomp_.global_comm.root_id());
		*n_measurements = decomp_.global_comm.broadcast(*n_measurements, decomp_.global_comm.root_id());
	}
	return;
}

void AstroDecomposition::distribute_parameters_wideband(Vector<t_int> &blocks_per_channel){

	if(decomp_.parallel_mpi){
		blocks_per_channel = decomp_.global_comm.broadcast(blocks_per_channel, decomp_.global_comm.root_id());
	}

	return;
}

void AstroDecomposition::reduce_kappa(Vector<t_real> dirty, t_real *kappa, t_real nu2){

	if(decomp_.parallel_mpi){
		decomp_.global_comm.distributed_sum(dirty, decomp_.global_comm.root_id());
		*kappa = (dirty.maxCoeff()*1e-1)/nu2;
		*kappa = decomp_.global_comm.broadcast(*kappa, decomp_.global_comm.root_id());
	}
	return;
}

void AstroDecomposition::collect_dirty_image(std::vector<Vector<t_complex>> dirty, std::vector<Vector<t_complex>> &global_dirty){

	int my_freq = 0;

	for(int freq=0; freq<decomp_.number_of_frequencies; freq++){
		if((decomp_.frequencies[freq].global_owner == decomp_.global_comm.rank()) or decomp_.global_comm.is_root()){
			if((decomp_.frequencies[freq].global_owner == decomp_.global_comm.rank()) and decomp_.global_comm.is_root()){
				global_dirty[freq] = dirty[my_freq];
				my_freq++;
			}else{
				if(decomp_.global_comm.is_root()){
					t_uint tag = freq;
					decomp_.global_comm.recv_eigen(global_dirty[freq], decomp_.frequencies[freq].global_owner, tag);
				}else{
					t_uint tag = freq;
					decomp_.global_comm.send_eigen(dirty[my_freq], decomp_.global_comm.root_id(), tag);
					my_freq++;
				}
			}
		}
	}
	return;
}

}
