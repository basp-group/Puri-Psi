#ifndef PURIPSI_ASTRO_DECOMPOSITION_H
#define PURIPSI_ASTRO_DECOMPOSITION_H

#include "puripsi/config.h"
#include "puripsi/types.h"
#include "puripsi/utilities.h"
#include <psi/mpi/decomposition.h>


namespace puripsi {

class AstroDecomposition : public psi::mpi::Decomposition {

public:

	AstroDecomposition(const bool parallel_mpi) : psi::mpi::Decomposition(parallel_mpi) {};

	AstroDecomposition(const bool parallel_mpi, psi::mpi::Communicator comm) : psi::mpi::Decomposition(parallel_mpi, comm) {};


public:
	//! Distributes the uv data to processes based on the decomposition
	void distribute_uv_data(const std::vector<std::vector<utilities::vis_params>> &uv_data, std::vector<std::vector<utilities::vis_params>> &my_uv_data);
	void gather_uv_data(std::vector<std::vector<utilities::vis_params>> &uv_data, const std::vector<std::vector<utilities::vis_params>> &my_uv_data);
	void gather_frequency_local_vector_int(Vector<t_int> &global_data, const Vector<t_int> &local_data);
	void distribute_parameters_int_real(int *channel_index, t_real *pixel_size);
	void distribute_parameters_int_int(t_int *n_blocks, t_int *n_measurements);
	void distribute_parameters_wideband(Vector<t_int> &blocks_per_channel);

	void reduce_kappa(Vector<t_real> dirty, t_real *kappa, t_real nu2);
	void collect_dirty_image(std::vector<Vector<t_complex>> dirty, std::vector<Vector<t_complex>> &global_dirty);

};

}
#endif
