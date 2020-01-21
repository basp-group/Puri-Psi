#include "puripsi/time_blocking.h"
namespace puripsi
{

//! TODO refactor this file so there is not as much duplicated content. Particularly remove copies of the same routine that only different by spectral
//! window functionality and create a version that can select spectral windows or not.

puripsi::details::isGreater::isGreater(std::vector<int> *g){
	GT = g;
}
void puripsi::details::isGreater::operator()(float i){
	if(i>0){
		GT->emplace_back(it);
		it++;
	}

	return;
}

// Get the number of channels and spectral window information for a given MS table.
void extract_number_of_channels_and_spectal_windows(string &dataName, int fieldID, int &number_of_channels, std::set<::casacore::uInt> &spw_ids){

	number_of_channels = 0;
	::casacore::MeasurementSet full_ms(dataName, ::casacore::TableLock::NoLocking);
	::casacore::MSMetaData full_meta_data(&full_ms, 1.);
	auto nChannels = full_meta_data.nChans();
	for(int i=0; i<nChannels.size();i++){
		number_of_channels += nChannels[i];
	}
	spw_ids = full_meta_data.getSpwsForField(fieldID); // get index of the spectral windows associated with the fieldID of interest
	return;

}

int extract_blocks_information(string &dataName, Eigen::VectorXi &scan_id, Eigen::VectorXi &scan_size, std::vector<Eigen::VectorXi> &snapshots_size, int field_id){
	// string dataName = "data/ms/CYG-C-6680-64.MS";
	// Create full MeasurementSet
	::casacore::MeasurementSet full_ms(dataName, ::casacore::TableLock::NoLocking);

	// Extract general informations (MSMetaData)
	::casacore::MSMetaData full_meta_data(&full_ms, 1.);
	// casacore::MSMetaData::MSMetaData(const MeasurementSet *const & ms, const Float maxCacheSizeMB)
	int nScans = full_meta_data.nScans();

	// Get the unique values of the scan numbers (in an Eigen vector)
	::casacore::ROScalarColumn<::casacore::Int> full_scan_nr (full_ms, "SCAN_NUMBER");
	Eigen::VectorXi full_scans_nr_eigen = Eigen::VectorXi::Map(full_scan_nr.getColumn().data(), full_scan_nr.nrow(), 1);
	std::set<int> full_ms_scan_set(full_scans_nr_eigen.data(), full_scans_nr_eigen.data()+full_scans_nr_eigen.size());
	std::vector<int> full_ms_scan_vector(full_ms_scan_set.begin(), full_ms_scan_set.end());
	scan_id = Eigen::VectorXi::Map(&full_ms_scan_vector[0], full_ms_scan_vector.size()); // error here!!

	// A. EXTRACT RELEVANT INFORMATION TO DEFINE THE DATA BLOCKS
	// Extract all the information of interest for the time blocking
	std::vector<Eigen::VectorXi> snapshots_position(nScans);
	snapshots_size.reserve(nScans);
	scan_size = Eigen::VectorXi::Zero(nScans);
	Eigen::VectorXi nSnapshots(nScans);
	int snapshot_number = 0;
	int scan_number = 0;
	for (int i=0; i<nScans; i++){
		// filter_local a std::to_string(); MeasurementSet with a given SCAN_NUMBER, get the associated TIME column, work on this for the blocking
		//::casacore::MeasurementSet ms_scan = full_ms(full_ms.col("SCAN_NUMBER") == scan_id[i]);
		::casacore::MeasurementSet ms_scan = full_ms(full_ms.col("SCAN_NUMBER") == scan_id[i] && full_ms.col("FIELD_ID") == field_id);
		// extract general informations (MSMetaData)
		::casacore::MSMetaData scan_meta_data(&ms_scan, 1.); // casacore::MSMetaData::MSMetaData(const MeasurementSet *const & ms, const Float maxCacheSizeMB)
		if(scan_meta_data.nRows() > 0){
			scan_size[scan_number] = scan_meta_data.nRows(); // number of visibilities within the scan considered
			// std::set<::casacore::Double> ::casacore::getTimesForScan(const ScanKey &scan);
			::casacore::ROScalarColumn<::casacore::Double> scan_time(ms_scan, "TIME");
			Eigen::VectorXd scan_time_eigen = Eigen::VectorXd::Map(scan_time.getColumn().data(), scan_time.nrow(), 1);
			// identify snapshots from scan_time
			Eigen::VectorXd dt = scan_time_eigen.segment(1, scan_time_eigen.size()-1) - scan_time_eigen.segment(0, scan_time_eigen.size()-1);
			// find position of the non-zero entries, gives position, size and number of snapshots
			std::vector<int> GT;
			std::for_each(dt.data(),dt.data()+dt.size(),puripsi::details::isGreater(&GT));
			if(GT.size() > 0){
				Eigen::VectorXi scan_snapshots_positions = Eigen::VectorXi::Map(&GT[0], GT.size(), 1);
				scan_snapshots_positions = scan_snapshots_positions.array() + 1;
				//std::cout << "Number of snapshots (scan " << i << "): " << scan_snapshots_positions.size() << "\n";
				nSnapshots[snapshot_number] = scan_snapshots_positions.size()+1;
				snapshots_size.emplace_back(Eigen::VectorXi::Zero(nSnapshots[snapshot_number], 1));
				snapshots_size[snapshot_number].segment(1, snapshots_size[snapshot_number].size()-2) = scan_snapshots_positions.segment(1, scan_snapshots_positions.size()-1).array() - scan_snapshots_positions.segment(0, scan_snapshots_positions.size()-1).array(); // -1+1
				snapshots_size[snapshot_number][0] = scan_snapshots_positions[0]; // -1+1
				snapshots_size[snapshot_number][nSnapshots[snapshot_number]-1] = scan_size[snapshot_number] - scan_snapshots_positions[scan_snapshots_positions.size()-1]; // -1+1
				snapshot_number++;
			}
			scan_number++;
		}
	}
	nScans =  snapshot_number;
	return nScans;
}

int extract_blocks_information(string &dataName, Eigen::VectorXi &scan_id, Eigen::VectorXi &scan_size, std::vector<Eigen::VectorXi> &snapshots_size, int spw_index, int field_id)
{
	// string dataName = "data/ms/CYG-C-6680-64.MS";
	// Create full MeasurementSet
	::casacore::MeasurementSet full_ms(dataName, ::casacore::TableLock::NoLocking);

	// Extract general informations (MSMetaData)
	::casacore::MSMetaData full_meta_data(&full_ms, 1.);
	// casacore::MSMetaData::MSMetaData(const MeasurementSet *const & ms, const Float maxCacheSizeMB)
	int nScans = full_meta_data.nScans();

	// Get the unique values of the scan numbers (in an Eigen vector)
	::casacore::ROScalarColumn<::casacore::Int> full_scan_nr (full_ms, "SCAN_NUMBER");
	Eigen::VectorXi full_scans_nr_eigen = Eigen::VectorXi::Map(full_scan_nr.getColumn().data(), full_scan_nr.nrow(), 1);
	std::set<int> full_ms_scan_set(full_scans_nr_eigen.data(), full_scans_nr_eigen.data()+full_scans_nr_eigen.size());
	std::vector<int> full_ms_scan_vector(full_ms_scan_set.begin(), full_ms_scan_set.end());
	scan_id = Eigen::VectorXi::Map(&full_ms_scan_vector[0], full_ms_scan_vector.size()); // error here!!

	// A. EXTRACT RELEVANT INFORMATION TO DEFINE THE DATA BLOCKS
	// Extract all the information of interest for the time blocking
	std::vector<Eigen::VectorXi> snapshots_position(nScans);
	snapshots_size.reserve(nScans);
	scan_size = Eigen::VectorXi::Zero(nScans);
	Eigen::VectorXi nSnapshots(nScans);
	int snapshot_number = 0;
	int scan_number = 0;
	for (int i=0; i<nScans; i++){
		// filter_local a std::to_string(); MeasurementSet with a given SCAN_NUMBER, get the associated TIME column, work on this for the blocking
		//::casacore::MeasurementSet ms_scan = full_ms(full_ms.col("SCAN_NUMBER") == scan_id[i]);
		::casacore::MeasurementSet ms_scan = full_ms(full_ms.col("SCAN_NUMBER") == scan_id[i] && full_ms.col("FIELD_ID") == field_id && full_ms.col("DATA_DESC_ID") == spw_index);
		// extract general informations (MSMetaData)
		::casacore::MSMetaData scan_meta_data(&ms_scan, 1.); // casacore::MSMetaData::MSMetaData(const MeasurementSet *const & ms, const Float maxCacheSizeMB)
		if(scan_meta_data.nRows() > 0){
			scan_size[scan_number] = scan_meta_data.nRows(); // number of visibilities within the scan considered
			// std::set<::casacore::Double> ::casacore::getTimesForScan(const ScanKey &scan);
			::casacore::ROScalarColumn<::casacore::Double> scan_time(ms_scan, "TIME");
			Eigen::VectorXd scan_time_eigen = Eigen::VectorXd::Map(scan_time.getColumn().data(), scan_time.nrow(), 1);
			// identify snapshots from scan_time
			Eigen::VectorXd dt = scan_time_eigen.segment(1, scan_time_eigen.size()-1) - scan_time_eigen.segment(0, scan_time_eigen.size()-1);
			// find position of the non-zero entries, gives position, size and number of snapshots
			std::vector<int> GT;
			std::for_each(dt.data(),dt.data()+dt.size(),puripsi::details::isGreater(&GT));
			if(GT.size() > 0){
				Eigen::VectorXi scan_snapshots_positions = Eigen::VectorXi::Map(&GT[0], GT.size(), 1);
				scan_snapshots_positions = scan_snapshots_positions.array() + 1;
				//std::cout << "Number of snapshots (scan " << i << "): " << scan_snapshots_positions.size() << "\n";
				nSnapshots[snapshot_number] = scan_snapshots_positions.size()+1;
				snapshots_size.emplace_back(Eigen::VectorXi::Zero(nSnapshots[snapshot_number], 1));
				snapshots_size[snapshot_number].segment(1, snapshots_size[snapshot_number].size()-2) = scan_snapshots_positions.segment(1, scan_snapshots_positions.size()-1).array() - scan_snapshots_positions.segment(0, scan_snapshots_positions.size()-1).array(); // -1+1
				snapshots_size[snapshot_number][0] = scan_snapshots_positions[0]; // -1+1
				snapshots_size[snapshot_number][nSnapshots[snapshot_number]-1] = scan_size[snapshot_number] - scan_snapshots_positions[scan_snapshots_positions.size()-1]; // -1+1
				snapshot_number++;
			}
			scan_number++;
		}
	}
	nScans =  snapshot_number;
	return nScans;
}

int snapshot_decomposition(std::vector<int> &blocks_size, int block_size, int min_block_size,
		int max_block_size, Eigen::VectorXi &snapshot_size){
	// initialization
	int size_current_block = snapshot_size[0];
	int nSnapshots = snapshot_size.size();
	int k = 0;
	int n_blocks = 0;
	int flag_min = false;
	std::pair<int, int> index_first_snapshot = {0, 0}; // current block, previous block
	// decomposition procedure
	while(k < nSnapshots){
		if (size_current_block < min_block_size){
			// 1. CURRENT BLOCK CONTAINS FEWER VISIBILITIES THAN THE PRESCRIBED TOLERANCE
			flag_min = true;
			if (k+1 < nSnapshots){
				// a) try to aggregate a new scan
				size_current_block += snapshot_size[k+1];
			}else{
				// b) last scan already, update block description
				int candidate_block_size = size_current_block + snapshot_size[k];
				int average_block_size = std::accumulate(blocks_size.begin(), blocks_size.end(), 0.)/n_blocks; // see if ok...

				if(candidate_block_size < 1.2*average_block_size){
					// i) group the last two blocks together
					blocks_size.back() += size_current_block;
					size_current_block = 0;
				}else{
					// ii) redefine the last two blocks (gather all the snapshots, try to find a uniform decomposition)
					int target_size = average_block_size/2;
					size_current_block = 0;
					int index = index_first_snapshot.second;
					while(size_current_block < target_size){
						index += 1;
						size_current_block += snapshot_size[index]; 
					}
					blocks_size.back() = size_current_block;
					blocks_size.emplace_back((snapshot_size.segment(index_first_snapshot.second, snapshot_size.size()-index+1)).sum());
					n_blocks += 1;
					size_current_block = 0;
				}
				// // b) last scan already, update block description
				// int current_distance = std::abs(size_current_block - block_size);
				// int candidate_distance = std::abs(block_size - (size_current_block + snapshot_size[k]));
				// if(current_distance < candidate_distance)
				// {
				// 	// i)
				// 	blocks_size.emplace_back(size_current_block);
				// 	n_blocks += 1;
				// 	size_current_block = 0;
				// }
				// else
				// {
				// 	// ii) not enough visibilities in the current group of scans,
				// 	// merge with the previous block
				// 	blocks_size.back() += size_current_block;
				// 	size_current_block = 0;
				// }
			}
			k = k+1;
		}else if (size_current_block > max_block_size){
			// 2. CURRENT BLOCK CONTAINS MORE VISIBILITIES THAN THE PRESCRIBED TOLERANCE
			if(flag_min){
				// a) too many visibilities in the current group of scans, try to remove the last scan
				int current_distance = size_current_block - block_size; // >= 0 in theory
				int previous_distance = block_size - (size_current_block - snapshot_size[k]); // >= 0 in theory
				if(previous_distance < current_distance){
					// undo last merge step, do not update k (already the appropriate value for the next iteration)
					size_current_block -= snapshot_size[k];
				}else{
					// leave block as is
					k += 1;
				}
				flag_min = false;
				n_blocks += 1;
			}else{
				// b) single large snapshot, but cannot be decomposed further
				n_blocks += 1;
				k++;
			}
			// update block description
			blocks_size.emplace_back(size_current_block);
			if(k < nSnapshots){
				size_current_block = snapshot_size[k];
				// k already updated before, so no k+1
			}
		}else{
			// 3. SIZE OF THE CURRENT BLOCK WITHIN THE PRESCRIBED TOLERANCE
			// update block description
			blocks_size.emplace_back(size_current_block);
			index_first_snapshot.second = index_first_snapshot.first;
			index_first_snapshot.first = k+1;
			n_blocks = n_blocks+1;
			if (k+1 < nSnapshots){
				size_current_block = snapshot_size[k+1];
			}
			k++;
		}
	}
	return n_blocks;
}

// Combine scans and snapshots to form a block whose size is close to a reference size
int time_blocking(int block_size, int nScans, Eigen::Ref<Eigen::VectorXi> scan_size,
		Eigen::Ref<Eigen::VectorXi> scan_id, double tol_min,
		double tol_max, std::vector<std::pair<int, int>> &blocks, std::vector<int> &blocks_size,
		std::vector<Eigen::VectorXi> &snapshots_size, std::vector<int> &blocks_snapshots_nr)
{

	int min_block_size = std::floor(block_size*tol_min);
	int max_block_size = std::floor(block_size*tol_max);

	int block_nScans = 1;
	int scan_index = 0; // index in scans (get the appropriate id, possibly non-contiguous integer)
	int starting_scan = scan_id[scan_index];
	int n_blocks = 0;
	int k = 0; // current scan index
	int size_current_block = scan_size[0];
	bool flag_min = false;
	Eigen::VectorXi n_blocks_snapshots = Eigen::VectorXi(nScans);

	blocks.reserve(nScans); // maximum number of blocks is nScans (not true if scans are furhter split... gives at least a first rough idea? to be possibly adjusted)
	blocks_size.reserve(nScans); // to be adjusted
	blocks_snapshots_nr.reserve(nScans); // number of blocks resulting from snapshots fusion for each scan

	// DEFINITION OF THE BLOCKS
	while (k < nScans){
		if (size_current_block <  min_block_size){
			// 1. CURRENT BLOCKS CONTAINS FEWER VISIBILITIES THAN THE PRESCRIBED TOLERANCE
			flag_min = true;
			if (k+1 < nScans){
				// a) try to aggregate a new scan
				size_current_block += scan_size[k+1];
				block_nScans++;
			}else{
				// b) last scan already, update block description
				int current_distance = std::abs(size_current_block - block_size);
				int candidate_distance = std::abs(block_size - (size_current_block + scan_size[k]));

				if(current_distance < candidate_distance){
					// i)
					blocks.emplace_back(std::make_pair(starting_scan, block_nScans));
					blocks_size.emplace_back(size_current_block);
					scan_index = scan_index+block_nScans;
					starting_scan = 0;
					block_nScans = 1;
					n_blocks++;
					size_current_block = 0;
				}else{
					// ii) not enough visibilities in the current group of scans, merge with the previous block
					blocks.back().second += block_nScans;
					blocks_size.back() += size_current_block;
					block_nScans = 1;
					size_current_block = 0;
				}
			}
			k++;
		}else if(size_current_block > max_block_size){
			// 2. CURRENT BLOCKS CONTAINS MORE VISIBILITIES THAN THE PRESCRIBED TOLERANCE
			if (flag_min){
				// a) too many visibilities in the current group of scans, try to remove the last scan
				int current_distance = size_current_block - block_size;
				int previous_distance = block_size - (size_current_block - scan_size[k]);
				if (previous_distance < current_distance){
					// undo last merge step, do not update k (already the appropriate value for the next iteration)
					block_nScans--;
					size_current_block -= scan_size[k];
					// do not update k
				}else{
					// leave block as is
					k++;
				}
				flag_min = false;
				n_blocks++;
				blocks_size.emplace_back(size_current_block);
			}else{
				// b) single large scan, further decomposed into groups of snapshots (procedure similar to this one)
				block_nScans = -1; // indicates that the scan has been split into groups of snapshots
				int n_blocks_snapshots = puripsi::snapshot_decomposition(blocks_size, block_size, min_block_size, max_block_size, snapshots_size[k]);
				blocks_snapshots_nr.emplace_back(n_blocks_snapshots);
				n_blocks += n_blocks_snapshots;
				// contained within the scan of interest
				k++;
			}
			// update block description
			blocks.emplace_back(std::make_pair(starting_scan, block_nScans));
			if(block_nScans>0){
				scan_index = scan_index+block_nScans;
			}else{
				scan_index++;
			}
			// starting_scan = starting_scan + block_nScans;
			block_nScans = 1;
			if (k < nScans){
				size_current_block = scan_size[k];
				starting_scan = scan_id[scan_index];
				// k already updated before, so no k+1
			}
		}else{
			// 3. SIZE OF THE CURRENT BLOCK WITHIN THE PRESCRIBED TOLERANCE
			// update block description
			blocks.emplace_back(std::make_pair(starting_scan, block_nScans));
			scan_index = scan_index+block_nScans;
			// starting_scan = starting_scan + block_nScans;
			if(scan_index < nScans){
				starting_scan = scan_id[scan_index];
			}
			block_nScans = 1;
			n_blocks++;
			blocks_size.emplace_back(size_current_block);
			if (k+1 < nScans){
				size_current_block = scan_size[k+1];
			}
			k++;
		}
	}
	return n_blocks;
}


std::vector<utilities::vis_params> get_time_blocks(std::string dataName, t_real *global_dl, t_real *global_pixel_size, t_int *global_n_blocks,
		t_int *global_n_measurements, int channel_index, double *tol_min, double *tol_max, int *block_size, int field_id){

	t_real dl = *global_dl;
	t_real pixel_size = *global_pixel_size;
	t_int n_blocks = *global_n_blocks;

	// Data parameters
	PURIPSI_HIGH_LOG("Using test file {}", dataName);

	//---- TIME BLOCKING ----//
	// 1. Extract general informations from the MS table
	Eigen::VectorXi full_ms_unique_scan_nr;
	Eigen::VectorXi scan_size;
	std::vector<Eigen::VectorXi> snapshots_size; // all theses vectors have the size nScans once the function has been applied
	int nScans = puripsi::extract_blocks_information(dataName, full_ms_unique_scan_nr, scan_size, snapshots_size, field_id);

	// 2. Define the data blocks
	// Combine scans and snapshots to form blocks whose size are close to a reference block_size
	// Note: the blocks are determined based in the size of the scans / snapshots BEFORE data flagging
	// (otherwise, need to repeat the same procedure n_channel times since the flags are channel dependent)
	std::vector<std::pair<int, int>> blocks;
	std::vector<int> blocks_size;
	std::vector<int> blocks_snapshots_nr;

	PURIPSI_LOW_LOG("Extracting time blocks");

	n_blocks = puripsi::time_blocking(*block_size, nScans, scan_size,
			full_ms_unique_scan_nr, *tol_min, *tol_max, blocks, blocks_size,
			snapshots_size, blocks_snapshots_nr);

	PURIPSI_LOW_LOG("Extracting data");

	// 3. Data extraction
	auto const ms_wrapper = puripsi::casa::MeasurementSet(dataName); // version where the data are not flagged
	int q = 0; // indexing in blocks
	int n = 0; // blocks indexing
	int m = 0; // indexing in blocks_snapshots_nr
	std::vector<Eigen::VectorXcd> data_blocks(n_blocks);

	// extract frequencies, and max frequency
	::casacore::MeasurementSet ms(dataName, ::casacore::TableLock::NoLocking);
	::casacore::ArrayColumn<::casacore::Double> freqCols(ms.spectralWindow(), ::casacore::MSSpectralWindow::columnName(::casacore::MSSpectralWindow::CHAN_FREQ));
	Eigen::VectorXd frequencies = Eigen::VectorXd::Map(freqCols(0).data(), freqCols(0).nelements(), 1);
	puripsi::t_real max_frequency = frequencies.maxCoeff();

	puripsi::t_real reference_frequency = frequencies(channel_index);

	std::vector<puripsi::utilities::vis_params> uv_data = std::vector<puripsi::utilities::vis_params>(n_blocks);
	t_real Bmax = 0.;

	// PA: the following lines need to be encapsulated inside a function, try to simplify the instructions (if possible)
	// At the moment, there are to many copies to my liking, implementation needs to be refined (possibly modify the internal definition
	// of the casacore wrapper defined in casacore.cc). To be possibly discussed with Adrian...
	while(n < n_blocks){
		if(blocks[q].second < 0){
			// 1. SNAPSHOT DECOMPOSITION OCCURRED: extract the data based on the info. previously extracted
			// a) filter the measurement set in terms of the scan
			// flag the data manually in this case
			//			std::string tmp_filter = "SCAN_NUMBER==" + std::to_string(blocks[q].first);
			std::string tmp_filter = "SCAN_NUMBER==" + std::to_string(blocks[q].first) + " AND FIELD_ID==" + std::to_string(field_id);
			puripsi::utilities::vis_params uv_data_temp = read_measurementset(ms_wrapper, channel_index, reference_frequency,
					puripsi::casa::MeasurementSet::ChannelWrapper::polarization::I, tmp_filter); // really long ! // LOOP over the spectral windows (add another parameter to read_measurement_set...)

			if(uv_data_temp.u.size() != 0){

				puripsi::casa::MeasurementSet::ChannelWrapper tmp_ms = ms_wrapper[std::make_pair(channel_index,tmp_filter)]; // try to avoid this if possible...
				Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags = tmp_ms.final_flag();

				int index_snapshot = 0;
				for(int k=0; k<blocks_snapshots_nr[m]; k++){
					uv_data[n].ra = tmp_ms.right_ascension();
					uv_data[n].dec = tmp_ms.declination();
					uv_data[n].units = puripsi::utilities::vis_units::lambda;
					uv_data[n].average_frequency = reference_frequency;
					// flags associated with the current group
					Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags2 = tmp_flags.segment(index_snapshot, blocks_size[n]);
					// remove flagged entries manually
					if(tmp_flags2.count() > 0){
						// some entries have been flagged
						uv_data[n].u = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[n]-tmp_flags2.count());
						uv_data[n].v = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[n]-tmp_flags2.count());
						uv_data[n].w = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[n]-tmp_flags2.count());
						uv_data[n].vis = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[n]-tmp_flags2.count());
						uv_data[n].weights = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[n]-tmp_flags2.count());


						// remove flagged entries manually
						int l = 0;
						for(int r=0; r<tmp_flags2.size(); r++){
							if(!tmp_flags2(r)){
								uv_data[n].u(l) = uv_data_temp.u(index_snapshot+r);
								uv_data[n].v(l) = uv_data_temp.v(index_snapshot+r);
								uv_data[n].w(l) = uv_data_temp.w(index_snapshot+r);
								uv_data[n].vis(l) = uv_data_temp.vis(index_snapshot+r);
								uv_data[n].weights(l) = uv_data_temp.weights(index_snapshot+r);
								l++;
							}
						}
					}else{
						// no flagged data
						uv_data[n].u = uv_data_temp.u.segment(index_snapshot, blocks_size[n]);
						uv_data[n].v = uv_data_temp.v.segment(index_snapshot, blocks_size[n]);
						uv_data[n].w = uv_data_temp.w.segment(index_snapshot, blocks_size[n]);
						uv_data[n].vis = uv_data_temp.vis.segment(index_snapshot, blocks_size[n]);
						uv_data[n].weights = uv_data_temp.weights.segment(index_snapshot, blocks_size[n]);
					}
					index_snapshot += blocks_size[n];
					// determine Bmax
					t_real b = std::sqrt( ((uv_data[n].u.array() * uv_data[n].u.array()) +
							(uv_data[n].v.array() * uv_data[n].v.array())).maxCoeff()); // already lambda_u...
					////*(puripsi::constant::c/max_frequency);
					Bmax = std::max(Bmax, b);

					n++;
				}
			}else{
				n_blocks = n_blocks - 1;
			}
			q++; // increase position in blocks
			m++; // increase position in blocks_snapshots_nr
		}else{
			// 2. ONLY GROUPS OF SCANS: directly extract the relevant data
			//			std::string tmp_filter = "SCAN_NUMBER>=" + std::to_string(blocks[q].first) + " AND SCAN_NUMBER<" + std::to_string(blocks[q].first + blocks[q].second);
			std::string tmp_filter = "SCAN_NUMBER>=" + std::to_string(blocks[q].first) + " AND SCAN_NUMBER<" + std::to_string(blocks[q].first + blocks[q].second) + " AND FIELD_ID==" + std::to_string(field_id);
			puripsi::casa::MeasurementSet::ChannelWrapper tmp_ms = ms_wrapper[std::make_pair(channel_index,tmp_filter)];
			puripsi::utilities::vis_params uv_data_temp = read_measurementset(ms_wrapper, channel_index, reference_frequency,
					puripsi::casa::MeasurementSet::ChannelWrapper::polarization::I, tmp_filter); // this is quite long at the moment!
			// std::cout << uv_data_temp.u.rows() << std::endl;
			if(uv_data_temp.u.size() != 0){
				Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags = tmp_ms.final_flag();
				// Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags = Eigen::Matrix<bool, Eigen::Dynamic, 1> ::Constant(uv_data_temp.u.size(),false);

				uv_data[n].ra = tmp_ms.right_ascension();
				uv_data[n].dec = tmp_ms.declination();
				uv_data[n].units = puripsi::utilities::vis_units::lambda;
				uv_data[n].average_frequency = reference_frequency;

				if(tmp_flags.count() > 0){
					// some entries have been flagged
					uv_data[n].u = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[n]-tmp_flags.count());
					uv_data[n].v = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[n]-tmp_flags.count());
					uv_data[n].w = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[n]-tmp_flags.count());
					uv_data[n].vis = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[n]-tmp_flags.count());
					uv_data[n].weights = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[n]-tmp_flags.count());

					// remove flagged entries manually
					int l = 0;
					for(int r=0; r<tmp_flags.size(); r++){
						if(!tmp_flags(r)){
							uv_data[n].u(l) = uv_data_temp.u(r);
							uv_data[n].v(l) = uv_data_temp.v(r);
							uv_data[n].w(l) = uv_data_temp.w(r);
							uv_data[n].vis(l) = uv_data_temp.vis(r);
							uv_data[n].weights(l) = uv_data_temp.weights(r);
							l++;
						}
					}
				}else{
					// no flagged data
					uv_data[n].u = uv_data_temp.u;
					uv_data[n].v = uv_data_temp.v;
					uv_data[n].w = uv_data_temp.w;
					uv_data[n].vis = uv_data_temp.vis;
					uv_data[n].weights = uv_data_temp.weights;
				}
				// determine Bmax
				t_real b = std::sqrt((uv_data[n].v.array() * uv_data[n].v.array() + uv_data[n].u.array() * uv_data[n].u.array()).maxCoeff()); // already lambda_u...
				//*(puripsi::constant::c/max_frequency);
				Bmax = std::max(Bmax, b);
				n++;

			}else{
				n_blocks = n_blocks - 1;
			}
			q++;
		}
	}

	// Check total number of measurements retained (for test.MS: 83168 after flagging)
	int n_measurements = 0;
	for(int n=0; n<n_blocks; ++n){
		// total number of measurements
		n_measurements += uv_data[n].vis.size();
	}

	// Compute pixel size (used in the definition of the measurement operators)
	t_real theta = 1/(2*Bmax);
	t_real theta_arcsec = theta * 180 * 3600 / EIGEN_PI;
	pixel_size = theta_arcsec / dl;
	//pixel_size = 0.64;
	//---- END TIME BLOCKING ----//

	*global_dl = dl;
	*global_pixel_size = pixel_size;
	*global_n_blocks = n_blocks;
	*global_n_measurements = n_measurements;
	return uv_data;

}


std::vector<utilities::vis_params> get_time_blocks(std::string dataName, t_real *global_dl, t_real *global_pixel_size, t_int *global_n_blocks,
		t_int *global_n_measurements, int spw_index, int channel_index, double *tol_min, double *tol_max, int *block_size, int field_id){

	t_real dl = *global_dl;
	t_real pixel_size = *global_pixel_size;
	t_int n_blocks = *global_n_blocks;

	// Data parameters
	PURIPSI_HIGH_LOG("Using test file {}", dataName);

	//---- TIME BLOCKING ----//
	// 1. Extract general informations from the MS table
	Eigen::VectorXi full_ms_unique_scan_nr;
	Eigen::VectorXi scan_size;
	std::vector<Eigen::VectorXi> snapshots_size; // all theses vectors have the size nScans once the function has been applied
	int nScans = puripsi::extract_blocks_information(dataName, full_ms_unique_scan_nr, scan_size, snapshots_size, spw_index, field_id);

	// 2. Define the data blocks
	// Combine scans and snapshots to form blocks whose size are close to a reference block_size
	// Note: the blocks are determined based in the size of the scans / snapshots BEFORE data flagging
	// (otherwise, need to repeat the same procedure n_channel times since the flags are channel dependent)
	std::vector<std::pair<int, int>> blocks;
	std::vector<int> blocks_size;
	std::vector<int> blocks_snapshots_nr;

	PURIPSI_LOW_LOG("Extracting time blocks");

	n_blocks = puripsi::time_blocking(*block_size, nScans, scan_size,
			full_ms_unique_scan_nr, *tol_min, *tol_max, blocks, blocks_size,
			snapshots_size, blocks_snapshots_nr);

	PURIPSI_LOW_LOG("Extracting data");

	// 3. Data extraction
	auto const ms_wrapper = puripsi::casa::MeasurementSet(dataName); // version where the data are not flagged
	int q = 0; // indexing in blocks
	int n = 0; // blocks indexing
	int m = 0; // indexing in blocks_snapshots_nr
	std::vector<Eigen::VectorXcd> data_blocks(n_blocks);

	// extract frequencies, and max frequency
	::casacore::MeasurementSet ms(dataName, ::casacore::TableLock::NoLocking);
	::casacore::ArrayColumn<::casacore::Double> freqCols(ms.spectralWindow(), ::casacore::MSSpectralWindow::columnName(::casacore::MSSpectralWindow::CHAN_FREQ));
	Eigen::VectorXd frequencies = Eigen::VectorXd::Map(freqCols(spw_index).data(), freqCols(spw_index).nelements(), 1);
	puripsi::t_real max_frequency = frequencies.maxCoeff();

	puripsi::t_real reference_frequency = frequencies(channel_index);

	std::vector<puripsi::utilities::vis_params> uv_data = std::vector<puripsi::utilities::vis_params>(n_blocks);
	t_real Bmax = 0.;

	PURIPSI_LOW_LOG("Total number of time blocks: {}", n_blocks);

	// PA: the following lines need to be encapsulated inside a function, try to simplify the instructions (if possible)
	// At the moment, there are to many copies to my liking, implementation needs to be refined (possibly modify the internal definition
	// of the casacore wrapper defined in casacore.cc). To be possibly discussed with Adrian...
	while(n < n_blocks){
		if(blocks[q].second < 0){
			// 1. SNAPSHOT DECOMPOSITION OCCURRED: extract the data based on the info. previously extracted
			// a) filter the measurement set in terms of the scan
			// flag the data manually in this case
			//			std::string tmp_filter = "SCAN_NUMBER==" + std::to_string(blocks[q].first);
			std::string tmp_filter = "SCAN_NUMBER==" + std::to_string(blocks[q].first) + " AND FIELD_ID==" + std::to_string(field_id) + " AND DATA_DESC_ID==" + std::to_string(spw_index);
			puripsi::utilities::vis_params uv_data_temp = read_measurementset(ms_wrapper, channel_index, reference_frequency,
					puripsi::casa::MeasurementSet::ChannelWrapper::polarization::I, tmp_filter); // really long ! // LOOP over the spectral windows (add another parameter to read_measurement_set...)

			if(uv_data_temp.u.size() != 0){

				puripsi::casa::MeasurementSet::ChannelWrapper tmp_ms = ms_wrapper[std::make_pair(channel_index,tmp_filter)]; // try to avoid this if possible...
				Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags = tmp_ms.final_flag();

				int index_snapshot = 0;
				for(int k=0; k<blocks_snapshots_nr[m]; k++){
					uv_data[n].ra = tmp_ms.right_ascension();
					uv_data[n].dec = tmp_ms.declination();
					uv_data[n].units = puripsi::utilities::vis_units::lambda;
					uv_data[n].average_frequency = reference_frequency;
					// flags associated with the current group
					Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags2 = tmp_flags.segment(index_snapshot, blocks_size[n]);
					// remove flagged entries manually
					if(tmp_flags2.count() > 0){
						// some entries have been flagged
						uv_data[n].u = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[n]-tmp_flags2.count());
						uv_data[n].v = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[n]-tmp_flags2.count());
						uv_data[n].w = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[n]-tmp_flags2.count());
						uv_data[n].vis = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[n]-tmp_flags2.count());
						uv_data[n].weights = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[n]-tmp_flags2.count());


						// remove flagged entries manually
						int l = 0;
						for(int r=0; r<tmp_flags2.size(); r++){
							if(!tmp_flags2(r)){
								uv_data[n].u(l) = uv_data_temp.u(index_snapshot+r);
								uv_data[n].v(l) = uv_data_temp.v(index_snapshot+r);
								uv_data[n].w(l) = uv_data_temp.w(index_snapshot+r);
								uv_data[n].vis(l) = uv_data_temp.vis(index_snapshot+r);
								uv_data[n].weights(l) = uv_data_temp.weights(index_snapshot+r);
								l++;
							}
						}
					}else{
						// no flagged data
						uv_data[n].u = uv_data_temp.u.segment(index_snapshot, blocks_size[n]);
						uv_data[n].v = uv_data_temp.v.segment(index_snapshot, blocks_size[n]);
						uv_data[n].w = uv_data_temp.w.segment(index_snapshot, blocks_size[n]);
						uv_data[n].vis = uv_data_temp.vis.segment(index_snapshot, blocks_size[n]);
						uv_data[n].weights = uv_data_temp.weights.segment(index_snapshot, blocks_size[n]);
					}
					index_snapshot += blocks_size[n];
					// determine Bmax
					t_real b = std::sqrt( ((uv_data[n].u.array() * uv_data[n].u.array()) +
							(uv_data[n].v.array() * uv_data[n].v.array())).maxCoeff()); // already lambda_u...
					////*(puripsi::constant::c/max_frequency);
					Bmax = std::max(Bmax, b);

					n++;
				}
			}else{
				n_blocks = n_blocks - 1;
			}
			q++; // increase position in blocks
			m++; // increase position in blocks_snapshots_nr
		}else{
			// 2. ONLY GROUPS OF SCANS: directly extract the relevant data
			//			std::string tmp_filter = "SCAN_NUMBER>=" + std::to_string(blocks[q].first) + " AND SCAN_NUMBER<" + std::to_string(blocks[q].first + blocks[q].second);
			std::string tmp_filter = "SCAN_NUMBER>=" + std::to_string(blocks[q].first) + " AND SCAN_NUMBER<" + std::to_string(blocks[q].first + blocks[q].second) + " AND FIELD_ID==" + std::to_string(field_id) + " AND DATA_DESC_ID==" + std::to_string(spw_index);
			puripsi::casa::MeasurementSet::ChannelWrapper tmp_ms = ms_wrapper[std::make_pair(channel_index,tmp_filter)];
			puripsi::utilities::vis_params uv_data_temp = read_measurementset(ms_wrapper, channel_index, reference_frequency,
					puripsi::casa::MeasurementSet::ChannelWrapper::polarization::I, tmp_filter); // this is quite long at the moment!
			// std::cout << uv_data_temp.u.rows() << std::endl;
			if(uv_data_temp.u.size() != 0){
				Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags = tmp_ms.final_flag();
				// Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags = Eigen::Matrix<bool, Eigen::Dynamic, 1> ::Constant(uv_data_temp.u.size(),false);

				uv_data[n].ra = tmp_ms.right_ascension();
				uv_data[n].dec = tmp_ms.declination();
				uv_data[n].units = puripsi::utilities::vis_units::lambda;
				uv_data[n].average_frequency = reference_frequency;

				if(tmp_flags.count() > 0){
					// some entries have been flagged
					uv_data[n].u = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[n]-tmp_flags.count());
					uv_data[n].v = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[n]-tmp_flags.count());
					uv_data[n].w = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[n]-tmp_flags.count());
					uv_data[n].vis = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[n]-tmp_flags.count());
					uv_data[n].weights = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[n]-tmp_flags.count());

					// remove flagged entries manually
					int l = 0;
					for(int r=0; r<tmp_flags.size(); r++){
						if(!tmp_flags(r)){
							uv_data[n].u(l) = uv_data_temp.u(r);
							uv_data[n].v(l) = uv_data_temp.v(r);
							uv_data[n].w(l) = uv_data_temp.w(r);
							uv_data[n].vis(l) = uv_data_temp.vis(r);
							uv_data[n].weights(l) = uv_data_temp.weights(r);
							l++;
						}
					}
				}else{
					// no flagged data
					uv_data[n].u = uv_data_temp.u;
					uv_data[n].v = uv_data_temp.v;
					uv_data[n].w = uv_data_temp.w;
					uv_data[n].vis = uv_data_temp.vis;
					uv_data[n].weights = uv_data_temp.weights;
				}
				// determine Bmax
				t_real b = std::sqrt((uv_data[n].v.array() * uv_data[n].v.array() + uv_data[n].u.array() * uv_data[n].u.array()).maxCoeff()); // already lambda_u...
				//*(puripsi::constant::c/max_frequency);
				Bmax = std::max(Bmax, b);
				n++;

			}else{
				n_blocks = n_blocks - 1;
			}
			q++;
		}
	}

	// Check total number of measurements retained (for test.MS: 83168 after flagging)
	int n_measurements = 0;
	for(int n=0; n<n_blocks; ++n){
		// total number of measurements
		n_measurements += uv_data[n].vis.size();
	}

	// Compute pixel size (used in the definition of the measurement operators)
	t_real theta = 1/(2*Bmax);
	t_real theta_arcsec = theta * 180 * 3600 / EIGEN_PI;
	pixel_size = theta_arcsec / dl;
	//pixel_size = 0.64;
	//---- END TIME BLOCKING ----//

	*global_dl = dl;
	*global_pixel_size = pixel_size;
	*global_n_blocks = n_blocks;
	*global_n_measurements = n_measurements;
	return uv_data;

}


void get_individual_file_block_data(std::string dataName, int channel_index, Vector<t_int> &n_blocks_per_dataset, int data_id,
		std::vector<puripsi::utilities::vis_params> &uv_data, 	psi::Vector<psi::t_real> reference_frequency,
		std::vector<std::vector<std::pair<int, int>>> blocks, std::vector<std::vector<int>> blocks_size,
		std::vector<std::vector<int>> blocks_snapshots_nr, int &id, int &n_blocks, int field_id){
	PURIPSI_LOW_LOG("Extracting data");
	auto const ms_wrapper = puripsi::casa::MeasurementSet(dataName); // version where the data are not flagged
	int q = 0; // indexing in blocks
	int n = 0; // blocks indexing
	int m = 0; // indexing in blocks_snapshots_nr
	std::vector<Eigen::VectorXcd> data_blocks(n_blocks_per_dataset(data_id));

	// PA: the following lines need to be encapsulated inside a function, try to simplify the instructions (if possible)
	// At the moment, there are to many copies to my liking, implementation needs to be refined (possibly modify the internal definition
	// of the casacore wrapper defined in casacore.cc). To be possibly discussed with Adrian...
	while(n < n_blocks_per_dataset(data_id)){
		if(blocks[data_id][q].second < 0){
			// 1. SNAPSHOT DECOMPOSITION OCCURRED: extract the data based on the info. previously extracted
			// a) filter the measurement set in terms of the scan
			// flag the data manually in this case
			std::string tmp_filter = "SCAN_NUMBER==" + std::to_string(blocks[data_id][q].first) + " AND FIELD_ID==" + std::to_string(field_id);
			puripsi::utilities::vis_params uv_data_temp = read_measurementset(ms_wrapper, channel_index, reference_frequency(data_id),
					puripsi::casa::MeasurementSet::ChannelWrapper::polarization::I, tmp_filter); // really long !
			if(uv_data_temp.u.size()  != 0){
				puripsi::casa::MeasurementSet::ChannelWrapper tmp_ms = ms_wrapper[std::make_pair(channel_index,tmp_filter)]; // try to avoid this if possible...
				Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags = tmp_ms.final_flag();

				int index_snapshot = 0;
				for(int k=0; k<blocks_snapshots_nr[data_id][m]; k++){
					uv_data.push_back(puripsi::utilities::vis_params());
					uv_data[id].ra = tmp_ms.right_ascension();
					uv_data[id].dec = tmp_ms.declination();
					uv_data[id].units = puripsi::utilities::vis_units::lambda;
					uv_data[id].average_frequency = reference_frequency(data_id);
					// flags associated with the current group
					Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags2 = tmp_flags.segment(index_snapshot, blocks_size[data_id][n]);
					// remove flagged entries manually
					if(tmp_flags2.count() > 0){
						// some entries have been flagged
						uv_data[id].u = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[data_id][n]-tmp_flags2.count());
						uv_data[id].v = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[data_id][n]-tmp_flags2.count());
						uv_data[id].w = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[data_id][n]-tmp_flags2.count());
						uv_data[id].vis = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[data_id][n]-tmp_flags2.count());
						uv_data[id].weights = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[data_id][n]-tmp_flags2.count());

						// remove flagged entries manually
						int l = 0;
						for(int r=0; r<tmp_flags2.size(); r++){
							if(!tmp_flags2(r)){
								uv_data[id].u(l) = uv_data_temp.u(index_snapshot+r);
								uv_data[id].v(l) = uv_data_temp.v(index_snapshot+r);
								uv_data[id].w(l) = uv_data_temp.w(index_snapshot+r);
								uv_data[id].vis(l) = uv_data_temp.vis(index_snapshot+r);
								uv_data[id].weights(l) = uv_data_temp.weights(index_snapshot+r);
								l++;
							}
						}
					}else{
						// no flagged data
						uv_data[id].u = uv_data_temp.u.segment(index_snapshot, blocks_size[data_id][n]);
						uv_data[id].v = uv_data_temp.v.segment(index_snapshot, blocks_size[data_id][n]);
						uv_data[id].w = uv_data_temp.w.segment(index_snapshot, blocks_size[data_id][n]);
						uv_data[id].vis = uv_data_temp.vis.segment(index_snapshot, blocks_size[data_id][n]);
						uv_data[id].weights = uv_data_temp.weights.segment(index_snapshot, blocks_size[data_id][n]);
					}
					index_snapshot += blocks_size[data_id][n];
					id++;
					n++;
				}
			}else{
				n_blocks_per_dataset(data_id) = n_blocks_per_dataset(data_id) - 1;
				n_blocks = n_blocks - 1;
			}
			q++; // increase position in blocks
			m++; // increase position in blocks_snapshots_nr
		}else{
			// 2. ONLY GROUPS OF SCANS: directly extract the relevant data
			std::string tmp_filter = "SCAN_NUMBER>=" + std::to_string(blocks[data_id][q].first) + " AND SCAN_NUMBER<" + std::to_string(blocks[data_id][q].first + blocks[data_id][q].second) + " AND FIELD_ID==" + std::to_string(field_id);
			puripsi::casa::MeasurementSet::ChannelWrapper tmp_ms = ms_wrapper[std::make_pair(channel_index,tmp_filter)];
			puripsi::utilities::vis_params uv_data_temp = read_measurementset(ms_wrapper, channel_index, reference_frequency(data_id),
					puripsi::casa::MeasurementSet::ChannelWrapper::polarization::I, tmp_filter); // this is quite long at the moment!
			if(uv_data_temp.u.size() != 0){
				uv_data.push_back(puripsi::utilities::vis_params());
				Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags = tmp_ms.final_flag();
				// Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags = Eigen::Matrix<bool, Eigen::Dynamic, 1> ::Constant(uv_data_temp.u.size(),false);

				uv_data[id].ra = tmp_ms.right_ascension();
				uv_data[id].dec = tmp_ms.declination();
				uv_data[id].units = puripsi::utilities::vis_units::lambda;
				uv_data[id].average_frequency = reference_frequency(data_id);

				if(tmp_flags.count() > 0){
					// some entries have been flagged
					uv_data[id].u = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[data_id][n]-tmp_flags.count());
					uv_data[id].v = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[data_id][n]-tmp_flags.count());
					uv_data[id].w = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[data_id][n]-tmp_flags.count());
					uv_data[id].vis = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[data_id][n]-tmp_flags.count());
					uv_data[id].weights = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[data_id][n]-tmp_flags.count());

					// remove flagged entries manually
					int l = 0;
					for(int r=0; r<tmp_flags.size(); r++){
						if(!tmp_flags(r)){
							uv_data[id].u(l) = uv_data_temp.u(r);
							uv_data[id].v(l) = uv_data_temp.v(r);
							uv_data[id].w(l) = uv_data_temp.w(r);
							uv_data[id].vis(l) = uv_data_temp.vis(r);
							uv_data[id].weights(l) = uv_data_temp.weights(r);
							l++;
						}
					}
				}else{
					// no flagged data
					uv_data[id].u = uv_data_temp.u;
					uv_data[id].v = uv_data_temp.v;
					uv_data[id].w = uv_data_temp.w;
					uv_data[id].vis = uv_data_temp.vis;
					uv_data[id].weights = uv_data_temp.weights;
				}
				n++;
				id++;
			}else{
				n_blocks_per_dataset(data_id) = n_blocks_per_dataset(data_id) - 1;
				n_blocks = n_blocks - 1;
			}
			q++;

		}
	}
	return;
}

void get_individual_file_block_data_spectral_window(std::string dataName, int spw_index, int channel_index, Vector<t_int> &n_blocks_per_dataset, int data_id,
		std::vector<puripsi::utilities::vis_params> &uv_data, 	psi::Vector<psi::t_real> reference_frequency,
		std::vector<std::vector<std::pair<int, int>>> blocks, std::vector<std::vector<int>> blocks_size,
		std::vector<std::vector<int>> blocks_snapshots_nr, int &id, int &n_blocks, int field_id){
	PURIPSI_LOW_LOG("Extracting data");
	auto const ms_wrapper = puripsi::casa::MeasurementSet(dataName); // version where the data are not flagged
	int q = 0; // indexing in blocks
	int n = 0; // blocks indexing
	int m = 0; // indexing in blocks_snapshots_nr
	std::vector<Eigen::VectorXcd> data_blocks(n_blocks_per_dataset(data_id));

	// PA: the following lines need to be encapsulated inside a function, try to simplify the instructions (if possible)
	// At the moment, there are to many copies to my liking, implementation needs to be refined (possibly modify the internal definition
	// of the casacore wrapper defined in casacore.cc). To be possibly discussed with Adrian...
	while(n < n_blocks_per_dataset(data_id)){
		if(blocks[data_id][q].second < 0){
			uv_data.push_back(puripsi::utilities::vis_params());
			// 1. SNAPSHOT DECOMPOSITION OCCURRED: extract the data based on the info. previously extracted
			// a) filter the measurement set in terms of the scan
			// flag the data manually in this case
			std::string tmp_filter = "SCAN_NUMBER==" + std::to_string(blocks[data_id][q].first) + " AND FIELD_ID==" + std::to_string(field_id) + " AND DATA_DESC_ID==" + std::to_string(spw_index);
			puripsi::utilities::vis_params uv_data_temp = read_measurementset(ms_wrapper, channel_index, reference_frequency(data_id),
					puripsi::casa::MeasurementSet::ChannelWrapper::polarization::I, tmp_filter); // really long !
			if(uv_data_temp.u.size()  != 0){
				puripsi::casa::MeasurementSet::ChannelWrapper tmp_ms = ms_wrapper[std::make_pair(channel_index,tmp_filter)]; // try to avoid this if possible...
				Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags = tmp_ms.final_flag();
				int index_snapshot = 0;
				for(int k=0; k<blocks_snapshots_nr[data_id][m]; k++){
					uv_data[id].ra = tmp_ms.right_ascension();
					uv_data[id].dec = tmp_ms.declination();
					uv_data[id].units = puripsi::utilities::vis_units::lambda;
					uv_data[id].average_frequency = reference_frequency(data_id);
					// flags associated with the current group
					Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags2 = tmp_flags.segment(index_snapshot, blocks_size[data_id][n]);
					// remove flagged entries manually
					if(tmp_flags2.count() > 0){
						// some entries have been flagged
						uv_data[id].u = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[data_id][n]-tmp_flags2.count());
						uv_data[id].v = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[data_id][n]-tmp_flags2.count());
						uv_data[id].w = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[data_id][n]-tmp_flags2.count());
						uv_data[id].vis = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[data_id][n]-tmp_flags2.count());
						uv_data[id].weights = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[data_id][n]-tmp_flags2.count());


						// remove flagged entries manually
						int l = 0;
						for(int r=0; r<tmp_flags2.size(); r++){
							if(!tmp_flags2(r)){
								uv_data[id].u(l) = uv_data_temp.u(index_snapshot+r);
								uv_data[id].v(l) = uv_data_temp.v(index_snapshot+r);
								uv_data[id].w(l) = uv_data_temp.w(index_snapshot+r);
								uv_data[id].vis(l) = uv_data_temp.vis(index_snapshot+r);
								uv_data[id].weights(l) = uv_data_temp.weights(index_snapshot+r);
								l++;
							}
						}
					}else{
						// no flagged data
						uv_data[id].u = uv_data_temp.u.segment(index_snapshot, blocks_size[data_id][n]);
						uv_data[id].v = uv_data_temp.v.segment(index_snapshot, blocks_size[data_id][n]);
						uv_data[id].w = uv_data_temp.w.segment(index_snapshot, blocks_size[data_id][n]);
						uv_data[id].vis = uv_data_temp.vis.segment(index_snapshot, blocks_size[data_id][n]);
						uv_data[id].weights = uv_data_temp.weights.segment(index_snapshot, blocks_size[data_id][n]);
					}
					index_snapshot += blocks_size[data_id][n];
					id++;
					n++;
				}
			}else{
				n_blocks_per_dataset(data_id) = n_blocks_per_dataset(data_id) - 1;
				n_blocks = n_blocks - 1;
			}
			q++; // increase position in blocks
			m++; // increase position in blocks_snapshots_nr
		}else{
			// 2. ONLY GROUPS OF SCANS: directly extract the relevant data
			std::string tmp_filter = "SCAN_NUMBER>=" + std::to_string(blocks[data_id][q].first) + " AND SCAN_NUMBER<" + std::to_string(blocks[data_id][q].first + blocks[data_id][q].second) + " AND FIELD_ID==" + std::to_string(field_id) + " AND DATA_DESC_ID==" + std::to_string(spw_index);
			puripsi::casa::MeasurementSet::ChannelWrapper tmp_ms = ms_wrapper[std::make_pair(channel_index,tmp_filter)];
			puripsi::utilities::vis_params uv_data_temp = read_measurementset(ms_wrapper, channel_index, reference_frequency(data_id),
					puripsi::casa::MeasurementSet::ChannelWrapper::polarization::I, tmp_filter); // this is quite long at the moment!
			if(uv_data_temp.u.size() != 0){
				uv_data.push_back(puripsi::utilities::vis_params());
				Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags = tmp_ms.final_flag();
				// Eigen::Matrix<bool, Eigen::Dynamic, 1> tmp_flags = Eigen::Matrix<bool, Eigen::Dynamic, 1> ::Constant(uv_data_temp.u.size(),false);

				uv_data[id].ra = tmp_ms.right_ascension();
				uv_data[id].dec = tmp_ms.declination();
				uv_data[id].units = puripsi::utilities::vis_units::lambda;
				uv_data[id].average_frequency = reference_frequency(data_id);

				if(tmp_flags.count() > 0){
					// some entries have been flagged
					uv_data[id].u = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[data_id][n]-tmp_flags.count());
					uv_data[id].v = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[data_id][n]-tmp_flags.count());
					uv_data[id].w = puripsi::Vector<puripsi::t_real>::Zero(blocks_size[data_id][n]-tmp_flags.count());
					uv_data[id].vis = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[data_id][n]-tmp_flags.count());
					uv_data[id].weights = puripsi::Vector<puripsi::t_complex>::Zero(blocks_size[data_id][n]-tmp_flags.count());

					// remove flagged entries manually
					int l = 0;
					for(int r=0; r<tmp_flags.size(); r++){
						if(!tmp_flags(r)){
							uv_data[id].u(l) = uv_data_temp.u(r);
							uv_data[id].v(l) = uv_data_temp.v(r);
							uv_data[id].w(l) = uv_data_temp.w(r);
							uv_data[id].vis(l) = uv_data_temp.vis(r);
							uv_data[id].weights(l) = uv_data_temp.weights(r);
							l++;
						}
					}
				}else{
					// no flagged data
					uv_data[id].u = uv_data_temp.u;
					uv_data[id].v = uv_data_temp.v;
					uv_data[id].w = uv_data_temp.w;
					uv_data[id].vis = uv_data_temp.vis;
					uv_data[id].weights = uv_data_temp.weights;
				}
				n++;
				id++;
			}else{
				n_blocks_per_dataset(data_id) = n_blocks_per_dataset(data_id) - 1;
				n_blocks = n_blocks - 1;
			}
			q++;

		}
	}
	return;
}

std::vector<utilities::vis_params> get_time_blocks_multi_file(std::vector<std::string> &dataName, t_real *global_dl, t_real *global_pixel_size,
		t_int *global_n_blocks, t_int *global_n_measurements, int channel_index, Vector<t_int> &n_blocks_per_dataset, int field_id){

	t_real dl = *global_dl;
	t_real pixel_size = *global_pixel_size;
	t_int n_blocks = *global_n_blocks;

	// Define blocking parameters
	double tol_min = .8;
	double tol_max = 1.2;
	int block_size = 90000;

	// Define auxiliary variables
	psi::Vector<psi::t_real> Bmax(dataName.size());
	psi::Vector<psi::t_real> reference_frequency(dataName.size());
	std::vector<std::vector<std::pair<int, int>>> blocks(dataName.size());
	std::vector<std::vector<int>> blocks_size(dataName.size());
	std::vector<std::vector<int>> blocks_snapshots_nr(dataName.size());

	for(int data_id=0; data_id < dataName.size(); data_id++){

		// Data parameters
		//---- TIME BLOCKING ----//
		// 1. Extract general informations from the MS table
		Eigen::VectorXi full_ms_unique_scan_nr;
		Eigen::VectorXi scan_size;
		std::vector<Eigen::VectorXi> snapshots_size; // all theses vectors have the size nScans once the function has been applied
		int nScans = puripsi::extract_blocks_information(dataName[data_id], full_ms_unique_scan_nr, scan_size, snapshots_size, field_id);

		// 2. Define the data blocks
		// Combine scans and snapshots to form blocks whose size are close to a reference block_size
		// Note: the blocks are determined based in the size of the scans / snapshots BEFORE data flagging
		// (otherwise, need to repeat the same procedure n_channel times since the flags are channel dependent)

		PURIPSI_LOW_LOG("Extracting time blocks");

		// 2. Define the data blocks
		n_blocks_per_dataset(data_id) = puripsi::time_blocking(block_size, nScans, scan_size,
				full_ms_unique_scan_nr, tol_min, tol_max, blocks[data_id], blocks_size[data_id],
				snapshots_size, blocks_snapshots_nr[data_id]);

		// Compute global_epsilon for the full MS table, compute epsilons for each block later on
		// load and flag data at once: /!\ MODIFICATION NEEDED HERE!
		::casacore::MeasurementSet ms(dataName[data_id], ::casacore::TableLock::NoLocking);
		::casacore::ArrayColumn<::casacore::Double> freqCols(ms.spectralWindow(), ::casacore::MSSpectralWindow::columnName(::casacore::MSSpectralWindow::CHAN_FREQ));
		Eigen::VectorXd frequencies = Eigen::VectorXd::Map(freqCols(0).data(), freqCols(0).nelements(), 1);
		reference_frequency(data_id) = frequencies(channel_index);

		const std::string filter_full = "NOT any(FLAG[" + std::to_string(channel_index) + ",]) AND NOT ANY(FLAG_ROW)";// + " AND FIELD_ID==" + std::to_string(field_id);
		auto const ms_wrapper = puripsi::casa::MeasurementSet(dataName[data_id]); // have an older version for this...
		puripsi::utilities::vis_params uv_data_temp = read_measurementset(ms_wrapper, channel_index, reference_frequency(data_id),
				puripsi::casa::MeasurementSet::ChannelWrapper::polarization::I, filter_full);
		if(uv_data_temp.u.size() != 0){
			Bmax(data_id) = std::sqrt( ((uv_data_temp.u.array() * uv_data_temp.u.array()) +
					(uv_data_temp.v.array() * uv_data_temp.v.array())).maxCoeff());
		}else{
			Bmax(data_id) = 0;
		}
	}

	// 3. Data extraction
	n_blocks = n_blocks_per_dataset.sum();
	std::vector<puripsi::utilities::vis_params> uv_data;
	uv_data.reserve(n_blocks);
	int id = 0; // position in uv_data std::vector

	for(int data_id=0; data_id < dataName.size(); data_id++){
		PURIPSI_HIGH_LOG("Processing file {} with {} blocks",dataName[data_id],n_blocks_per_dataset[data_id]);
		PURIPSI_HIGH_LOG("Channel index {}", channel_index);
		get_individual_file_block_data(dataName[data_id], channel_index, n_blocks_per_dataset, data_id,
				uv_data, reference_frequency, blocks, blocks_size, blocks_snapshots_nr, id, n_blocks, field_id);
	}

	PURIPSI_HIGH_LOG("Total number of time blocks: {}", n_blocks);

	// Check total number of measurements retained (for test.MS: 83168 after flagging)
	int n_measurements = 0;
	for(int n=0; n<n_blocks; ++n){
		// total number of measurements
		n_measurements += uv_data[n].vis.size();
	}

	// Compute pixel size (used in the definition of the measurement operators)
	t_real theta = 1/(2*Bmax.maxCoeff());
	t_real theta_arcsec = theta * 180 * 3600 / EIGEN_PI;
	pixel_size = theta_arcsec / dl;
	//---- END TIME BLOCKING ----//


	*global_dl = dl;
	*global_pixel_size = pixel_size;
	*global_n_blocks = n_blocks;
	*global_n_measurements = n_measurements;
	return uv_data;
}

std::vector<utilities::vis_params> get_time_blocks_multi_file_spectral_window(std::vector<std::string> &dataName, t_real *global_dl, t_real *global_pixel_size,
		t_int *global_n_blocks, t_int *global_n_measurements, int spectral_window, int channel_index, Vector<t_int> &n_blocks_per_dataset, int field_id){

	t_real dl = *global_dl;
	t_real pixel_size = *global_pixel_size;
	t_int n_blocks = *global_n_blocks;

	// Define blocking parameters
	double tol_min = .8;
	double tol_max = 1.2;
	int block_size = 90000;

	// Define auxiliary variables
	psi::Vector<psi::t_real> Bmax(dataName.size());
	psi::Vector<psi::t_real> reference_frequency(dataName.size());
	std::vector<std::vector<std::pair<int, int>>> blocks(dataName.size());
	std::vector<std::vector<int>> blocks_size(dataName.size());
	std::vector<std::vector<int>> blocks_snapshots_nr(dataName.size());

	for(int data_id=0; data_id < dataName.size(); data_id++){

		// Data parameters
		//---- TIME BLOCKING ----//
		// 1. Extract general informations from the MS table
		Eigen::VectorXi full_ms_unique_scan_nr;
		Eigen::VectorXi scan_size;
		std::vector<Eigen::VectorXi> snapshots_size; // all theses vectors have the size nScans once the function has been applied
		int nScans = puripsi::extract_blocks_information(dataName[data_id], full_ms_unique_scan_nr, scan_size, snapshots_size, spectral_window, field_id);

		// 2. Define the data blocks
		// Combine scans and snapshots to form blocks whose size are close to a reference block_size
		// Note: the blocks are determined based in the size of the scans / snapshots BEFORE data flagging
		// (otherwise, need to repeat the same procedure n_channel times since the flags are channel dependent)

		PURIPSI_LOW_LOG("Extracting time blocks");

		// 2. Define the data blocks
		n_blocks_per_dataset[data_id] = puripsi::time_blocking(block_size, nScans, scan_size,
				full_ms_unique_scan_nr, tol_min, tol_max, blocks[data_id], blocks_size[data_id],
				snapshots_size, blocks_snapshots_nr[data_id]);

		// Compute global_epsilon for the full MS table, compute epsilons for each block later on
		// load and flag data at once: /!\ MODIFICATION NEEDED HERE!
		::casacore::MeasurementSet ms(dataName[data_id], ::casacore::TableLock::NoLocking);
		::casacore::ArrayColumn<::casacore::Double> freqCols(ms.spectralWindow(), ::casacore::MSSpectralWindow::columnName(::casacore::MSSpectralWindow::CHAN_FREQ));
		Eigen::VectorXd frequencies = Eigen::VectorXd::Map(freqCols(spectral_window).data(), freqCols(spectral_window).nelements(), 1);
		reference_frequency(data_id) = frequencies(channel_index);

		const std::string filter_full = "NOT any(FLAG[" + std::to_string(channel_index) + ",]) AND NOT ANY(FLAG_ROW)" + " AND FIELD_ID==" + std::to_string(field_id) + " AND DATA_DESC_ID==" + std::to_string(spectral_window);
		auto const ms_wrapper = puripsi::casa::MeasurementSet(dataName[data_id]); // have an older version for this...
		puripsi::utilities::vis_params uv_data_temp = read_measurementset(ms_wrapper, channel_index, reference_frequency(data_id),
				puripsi::casa::MeasurementSet::ChannelWrapper::polarization::I, filter_full);
		if(uv_data_temp.u.size() != 0){
			Bmax(data_id) = std::sqrt( ((uv_data_temp.u.array() * uv_data_temp.u.array()) +
					(uv_data_temp.v.array() * uv_data_temp.v.array())).maxCoeff());
		}else{
			Bmax(data_id) = 0;
		}
	}

	// 3. Data extraction
	n_blocks = n_blocks_per_dataset.sum();
	std::vector<puripsi::utilities::vis_params> uv_data;
	uv_data.reserve(n_blocks);
	int id = 0; // position in uv_data std::vector


	for(int data_id=0; data_id < dataName.size(); data_id++){
		get_individual_file_block_data_spectral_window(dataName[data_id], spectral_window, channel_index, n_blocks_per_dataset, data_id,
				uv_data, reference_frequency, blocks, blocks_size, blocks_snapshots_nr, id, n_blocks, field_id);
	}

	// Check total number of measurements retained (for test.MS: 83168 after flagging)
	int n_measurements = 0;
	for(int n=0; n<n_blocks; ++n){
		// total number of measurements
		n_measurements += uv_data[n].vis.size();
	}

	// Compute pixel size (used in the definition of the measurement operators)
	t_real theta = 1/(2*Bmax.maxCoeff());
	t_real theta_arcsec = theta * 180 * 3600 / EIGEN_PI;
	pixel_size = theta_arcsec / dl;
	//---- END TIME BLOCKING ----//


	*global_dl = dl;
	*global_pixel_size = pixel_size;
	*global_n_blocks = n_blocks;
	*global_n_measurements = n_measurements;
	return uv_data;
}


} // end namespace puripsi
