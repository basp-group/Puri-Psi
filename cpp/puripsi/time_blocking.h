#ifndef PURIPSI_TIME_BLOCKING_H
#define PURIPSI_TIME_BLOCKING_H
#include "puripsi/utilities.h"
#include "puripsi/casacore.h"

#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/casa/Arrays/Slicer.h>
#include <casacore/casa/Arrays/Cube.h>
#include <casacore/casa/Containers/Record.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/ms/MeasurementSets/MSMainColumns.h>
#include <casacore/ms/MeasurementSets/MSAntennaColumns.h>
#include <Eigen/Core>
#include <casacore/ms/MeasurementSets/MSFieldColumns.h>
#include <casacore/ms/MSOper/MSMetaData.h>   // for nRows
#include <casacore/tables/TaQL/ExprNode.h>   // for selection
#include <casacore/ms/MSSel/MSSelector.h>    // for MSSelector
#include <casacore/tables/TaQL/TableParse.h> // for TAQL
#include <casacore/ms/MSOper/MSKeys.h>

namespace puripsi
{
namespace details
{
// Auxiliary class useful to unique sort the scan IDs used in the MS file
class isGreater{
    public:
    std::vector<int>* GT;
    isGreater(std::vector<int> *g);
    void operator()(float i);
    private:
    int it = 0;
};

}

// Get the number of channels and spectral window information for a given MS table.
void extract_number_of_channels_and_spectal_windows(string &dataName, int fieldID, int &number_of_channels, std::set<::casacore::uInt> &spw_ids);

// Extract useful informations from the measurement set to do time blocking
int extract_blocks_information(string &dataName, Eigen::VectorXi &scan_id, Eigen::VectorXi &scan_size, std::vector<Eigen::VectorXi> &snapshots_size, int field_id);
int extract_blocks_information(string &dataName, Eigen::VectorXi &scan_id, Eigen::VectorXi &scan_size, std::vector<Eigen::VectorXi> &snapshots_size, int spw_index, int field_id);

// Combine scans (or snapshots within scans) to obtain blocks of an appropriate size
int time_blocking(int block_size, int nScans, Eigen::Ref<Eigen::VectorXi> scan_size,
                  Eigen::Ref<Eigen::VectorXi> scan_id, double tol_min,
                  double tol_max, std::vector<std::pair<int, int>> &blocks, std::vector<int> &blocks_size,
                  std::vector<Eigen::VectorXi> &snapshots_size, std::vector<int> &blocks_snapshots_nr);
// Decompose a scan into groups of snapshots (in case a scan is larger than the prescribed block size)
int snapshot_decomposition(std::vector<int> &blocks_size, int block_size, int min_block_size, int max_block_size, Eigen::VectorXi &snapshot_size);

void get_individual_file_block_data(std::string dataName, int channel_index, Vector<t_int> &n_blocks_per_dataset, int data_id,
		std::vector<puripsi::utilities::vis_params> &uv_data, 	psi::Vector<psi::t_real> reference_frequency,
		std::vector<std::vector<std::pair<int, int>>> blocks, std::vector<std::vector<int>> blocks_size, std::vector<std::vector<int>> blocks_snapshots_nr,
		int &id, int &n_blocks, int field_id);
void get_individual_file_block_data_spectal_window(std::string dataName, int spw_index, int channel_index, Vector<t_int> &n_blocks_per_dataset, int data_id,
		std::vector<puripsi::utilities::vis_params> &uv_data, 	psi::Vector<psi::t_real> reference_frequency,
		std::vector<std::vector<std::pair<int, int>>> blocks, std::vector<std::vector<int>> blocks_size, std::vector<std::vector<int>> blocks_snapshots_nr,
		int &id, int &n_blocks, int field_id);
std::vector<utilities::vis_params> get_time_blocks(std::string dataName, t_real *global_dl, t_real *global_pixel_size, t_int *global_n_blocks,
		t_int *global_n_measurements, int channel_index, double *tol_min, double *tol_max, int *block_size, int field_id=0);
std::vector<utilities::vis_params> get_time_blocks(std::string dataName, t_real *global_dl, t_real *global_pixel_size, t_int *global_n_blocks,
		t_int *global_n_measurements, int spw_index, int channel_index, double *tol_min, double *tol_max, int *block_size, int field_id=0);
std::vector<utilities::vis_params> get_time_blocks_multi_file(std::vector<std::string> &dataName, t_real *global_dl, t_real *global_pixel_size,
		t_int *global_n_blocks, t_int *global_n_measurements, int channel_index, Vector<t_int> &n_blocks_per_dataset, int field_id=0);
std::vector<utilities::vis_params> get_time_blocks_multi_file_spectral_window(std::vector<std::string> &dataName, t_real *global_dl, t_real *global_pixel_size,
		t_int *global_n_blocks, t_int *global_n_measurements, int spectral_window, int channel_index, Vector<t_int> &n_blocks_per_dataset, int field_id=0);

}
#endif
