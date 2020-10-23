#ifndef PURIPSI_ASTRO_DECOMPOSITION_H
#define PURIPSI_ASTRO_DECOMPOSITION_H

#include "puripsi/config.h"
#include "puripsi/types.h"
#include "puripsi/utilities.h"
#include "puripsi/logging.h"
#include <psi/mpi/decomposition.h>


namespace puripsi {

class AstroDecomposition : public psi::mpi::Decomposition {

public:

	AstroDecomposition(const bool parallel_mpi) : psi::mpi::Decomposition(parallel_mpi) {};

	AstroDecomposition(const bool parallel_mpi, psi::mpi::Communicator comm) : psi::mpi::Decomposition(parallel_mpi, comm) { };


public:
	//! Distributes the uv data to processes based on the decomposition
	void distribute_uv_data(const std::vector<std::vector<utilities::vis_params>> &uv_data, std::vector<std::vector<utilities::vis_params>> &my_uv_data);
	void gather_uv_data(std::vector<std::vector<utilities::vis_params>> &uv_data, const std::vector<std::vector<utilities::vis_params>> &my_uv_data);
	template <class T1, class T2> void distribute_target_data(const T1 target_data, T1 &my_target_data) const;
	template <class T1, class T2> void gather_target_data(T1 &target_data, const T1 my_target_data) const;
	template <class T1, class T2> void distribute_uv_data(const T1 target_data, T1 &my_target_data) const;
	template <class T1, class T2> void gather_uv_data(T1 &target_data, const T1 my_target_data) const;
	void gather_frequency_local_vector_int(Vector<t_int> &global_data, const Vector<t_int> &local_data);
	void distribute_parameters_int_real(int *channel_index, t_real *pixel_size);
	void distribute_parameters_int_int(t_int *n_blocks, t_int *n_measurements);
	void distribute_parameters_wideband(Vector<t_int> &blocks_per_channel);

	void reduce_kappa(Vector<t_real> dirty, t_real *kappa, t_real nu2);
	void collect_dirty_image(std::vector<Vector<t_complex>> dirty, std::vector<Vector<t_complex>> &global_dirty);

};

template <class T1, class T2>
void AstroDecomposition::distribute_target_data(const T1 target_data, T1 &my_target_data) const{

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
						int temp_size = target_data[f][t].size();
						my_target_data[f][t] = T2(temp_size);
						for(int k=0; k<temp_size; k++){
							my_target_data[my_freq_index][my_block_index](k) = target_data[f][t](k);
						}
						my_block_index++;
						freq_used = true;
						//! Otherwise, send to the owning process
					}else{
						t_uint tag = 0;
						int temp_size = target_data[f][t].size();
						decomp_.global_comm.send_single(temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_eigen(target_data[f][t], decomp_.frequencies[f].time_blocks[t].global_owner, tag);
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
					my_target_data[f][t] = T2(temp_size);
					decomp_.global_comm.recv_eigen(my_target_data[f][t], decomp_.global_comm.root_id(), tag);
				}
			}
		}

	}
	return;
}

template <class T1, class T2>
void AstroDecomposition::gather_target_data(T1 &target_data, const T1 my_target_data) const{

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
						int temp_size = my_target_data[f][t].size();
						target_data[f][t] = T2(temp_size);
						for(int k=0; k<temp_size; k++){
							target_data[f][t](k) = my_target_data[my_freq_index][my_block_index](k);
						}
						my_block_index++;
						freq_used = true;
						//! Otherwise, received from the owning process
					}else{
						t_uint tag = 0;
						int temp_size;
						decomp_.global_comm.recv_single(&temp_size, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						target_data[f][t] = T2(temp_size);
						decomp_.global_comm.recv_eigen(target_data[f][t], decomp_.frequencies[f].time_blocks[t].global_owner, tag);
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
						int temp_size = my_target_data[my_freq_index][my_block_index].size();
						decomp_.global_comm.send_single(temp_size, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_eigen(my_target_data[my_freq_index][my_block_index], decomp_.global_comm.root_id(), tag);
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

template <class T1, class T2>
void AstroDecomposition::distribute_uv_data(const T1 target_data, T1 &my_target_data) const{

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
						int temp_rows = target_data[f][t].rows();
						int temp_cols = target_data[f][t].cols();
						my_target_data[f][t] = T2(temp_rows, temp_cols);
						for(int i=0; i<temp_rows; i++){
							for(int j=0; j<temp_cols; j++){
								my_target_data[my_freq_index][my_block_index](i,j) = target_data[f][t](i,j);
							}
						}
						my_block_index++;
						freq_used = true;
						//! Otherwise, send to the owning process
					}else{
						t_uint tag = 0;
						int temp_rows = target_data[f][t].rows();
						int temp_cols = target_data[f][t].cols();
						decomp_.global_comm.send_single(temp_rows, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_single(temp_cols, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.send_eigen(target_data[f][t], decomp_.frequencies[f].time_blocks[t].global_owner, tag);
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
					int temp_rows, temp_cols;
					decomp_.global_comm.recv_single(&temp_rows, decomp_.global_comm.root_id(), tag);
					decomp_.global_comm.recv_single(&temp_cols, decomp_.global_comm.root_id(), tag);
					my_target_data[f][t] = T2(temp_rows, temp_cols);
					decomp_.global_comm.recv_eigen(my_target_data[f][t], decomp_.global_comm.root_id(), tag);
				}
			}
		}

	}
	return;
}

template <class T1, class T2>
void AstroDecomposition::gather_uv_data(T1 &target_data, const T1 my_target_data) const{

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
						int temp_rows = my_target_data[f][t].rows();
						int temp_cols = my_target_data[f][t].cols();
						target_data[f][t] = T2(temp_rows, temp_cols);
						for(int i=0; i<temp_rows; i++){
							for(int j=0; j<temp_cols; j++){
								target_data[f][t](i,j) = my_target_data[my_freq_index][my_block_index](i,j);
							}
						}
						my_block_index++;
						freq_used = true;
						//! Otherwise, received from the owning process
					}else{
						t_uint tag = 0;
						int temp_rows, temp_cols;
						decomp_.global_comm.recv_single(&temp_rows, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						decomp_.global_comm.recv_single(&temp_cols, decomp_.frequencies[f].time_blocks[t].global_owner, tag);
						target_data[f][t] = T2(temp_rows, temp_cols);
						decomp_.global_comm.recv_eigen(target_data[f][t], decomp_.frequencies[f].time_blocks[t].global_owner, tag);
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
						int temp_rows = my_target_data[my_freq_index][my_block_index].rows();
						int temp_cols = my_target_data[my_freq_index][my_block_index].cols();
						decomp_.global_comm.send_single(temp_rows, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_single(temp_cols, decomp_.global_comm.root_id(), tag);
						decomp_.global_comm.send_eigen(my_target_data[my_freq_index][my_block_index], decomp_.global_comm.root_id(), tag);
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

}
#endif
