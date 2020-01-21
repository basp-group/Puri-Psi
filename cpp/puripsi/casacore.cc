#include "puripsi/casacore.h"
#include "puripsi/config.h"
#include <sstream>
#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/tables/TaQL/ExprNode.h>
#include "puripsi/logging.h"
#include "puripsi/types.h"

namespace puripsi {
namespace casa {
std::string const MeasurementSet::default_filter = "WHERE NOT ANY(FLAG)";
MeasurementSet &MeasurementSet::filename(std::string const &filename) {
	clear();
	filename_ = filename;
	return *this;
}

::casacore::Table const &MeasurementSet::table(std::string const &name) const {
	auto const tabname = name == "" ? filename() : filename() + "/" + name;
	auto i_result = tables_->find(tabname);
	if(i_result == tables_->end())
		//i_result = tables_->emplace(tabname, ::casacore::Table(tabname)).first;
		i_result = tables_->emplace(tabname, ::casacore::Table(tabname,::casacore::TableLock::NoLocking)).first;

	return i_result->second;
}

std::size_t MeasurementSet::size() const {
	if(table().nrow() == 0)
		return 0;
	auto const column = array_column<::casacore::Double>("CHAN_FREQ", "SPECTRAL_WINDOW");
	auto const orig = column.shape(0);
	for(t_uint i(1); i < column.nrow(); ++i)
		if(orig != column.shape(i))
			throw std::runtime_error("Can only do rectangular measurement set for now");
	return orig(0);
}

// /** Stokes component */
// Vector<t_complex> MeasurementSet::ChannelWrapper::I(std::string const &col) const {
//   Matrix<t_complex> data = table_column<t_complex>(ms_.table(), col, filter());
//   Matrix<t_real> weights = table_column<t_real>(ms_.table(), "WEIGHT", filter());
//   Vector<t_complex> I = (weights.col(0).array().cast<t_complex>()*data.col(0).array() +
//                       weights.col(3).array().cast<t_complex>()*data.col(3).array())/(weights.col(0).array() + weights.col(3).array()).cast<t_complex>();
//   return I;
// }
// /** Standard deviation for the Stokes component [TO BE IMPROVED LATER ON!]*/
// Vector<t_real> MeasurementSet::ChannelWrapper::wI(Sigma const &col) const {
//   Matrix<t_real> weights = table_column<t_real>(ms_.table(), "WEIGHT", filter());
//   Vector<t_real> w = (weights.col(3) + weights.col(0)).array().sqrt().cwiseInverse()/2;
//   /* the last /2 comes to counter-act a weird 2 factor in the .cc...   */
//   return w;
// }

MeasurementSet::const_iterator MeasurementSet::begin(std::string const &filter) const {
	return const_iterator(0, *this, filter);
}
MeasurementSet::const_iterator MeasurementSet::end(std::string const &filter) const {
	return const_iterator(size(), *this, filter);
}
MeasurementSet::ChannelWrapper MeasurementSet::operator[](t_uint i) const {
	return ChannelWrapper(i, *this, "");
}

MeasurementSet::ChannelWrapper MeasurementSet::
operator[](std::tuple<t_uint, std::string> const &i) const {
	if(std::get<0>(i) >= size())
		throw std::out_of_range("Not that many channels");
	return ChannelWrapper(std::get<0>(i), *this, std::get<1>(i));
}

std::string MeasurementSet::ChannelWrapper::filter() const {
	std::ostringstream sstr;
	// sstr << "WHERE NOT any(FLAG[" << channel_ << ",])";
	if(not filter_.empty())
		// sstr << "AND " << filter_;
		sstr << " WHERE " << filter_;
	return sstr.str();
}
std::string MeasurementSet::ChannelWrapper::index(std::string const &variable) const {
	std::ostringstream sstr;
	sstr << variable << "[" << channel_ << ",]";
	return sstr.str();
}

Vector<t_real>
MeasurementSet::ChannelWrapper::joined_spectral_window(std::string const &column) const {
	auto const raw = raw_spectral_window(column);
	auto const ids = ms_.column<::casacore::Int>("DATA_DESC_ID", filter());
	auto const spids
	= table_column<::casacore::Int>(ms_.table("DATA_DESCRIPTION"), "SPECTRAL_WINDOW_ID");
	Vector<t_real> result(ids.size());
	for(Eigen::DenseIndex i(0); i < ids.size(); ++i) {
		assert(ids(i) < spids.size());
		assert(spids(ids(i)) < raw.size());
		result(i) = raw(spids(ids(i)));
	}
	return result;
}

bool MeasurementSet::ChannelWrapper::is_valid() const {
	std::ostringstream sstr;
	// sstr << "USING STYLE PYTHON SELECT FLAG[" << channel_ << ",] as R FROM $1 WHERE NOT any(FLAG["
	//      << channel_ << ",])";
	sstr << "USING STYLE PYTHON SELECT FLAG[" << channel_ << ",] as R FROM $1";
	if(not filter_.empty())
		// sstr << "AND " << filter_;
		sstr << " WHERE " << filter_;
	auto const taql_table = ::casacore::tableCommand(sstr.str(), ms_.table());
	return taql_table.table().nrow() > 0;
}

Vector<bool> MeasurementSet::ChannelWrapper::flag() const {
	std::ostringstream sstr;
	sstr << "USING STYLE PYTHON SELECT FLAG[" << channel_ << ",] as R FROM $1";
	if(not filter_.empty())
		sstr << " WHERE " << filter_;
	auto const taql_table = ::casacore::tableCommand(sstr.str(), ms_.table());
	Vector<bool> result = details::get_taql_array<bool>(taql_table).rowwise().any();
	return result;
}

Vector<bool> MeasurementSet::ChannelWrapper::flag_row() const {
	return table_column<bool>(ms_.table(), "FLAG_ROW", filter());
}

Vector<bool> MeasurementSet::ChannelWrapper::final_flag() const {
	std::ostringstream sstr;
	sstr << "USING STYLE PYTHON SELECT FLAG[" << channel_ << ",] as R FROM $1";
	if(not filter_.empty())
		sstr << " WHERE " << filter_;
	auto const taql_table = ::casacore::tableCommand(sstr.str(), ms_.table());
	// Vector<bool> flag = details::get_taql_array<bool>(taql_table).rowwise().any();
  Matrix<bool> flag_temp = details::get_taql_array<bool>(taql_table);
  Vector<bool> flag = flag_temp.col(0).array().binaryExpr(flag_temp.col(3).array(), [](bool x, bool y){return x||y;});  
	Vector<bool> flag_row = table_column<bool>(ms_.table(), "FLAG_ROW", filter());
	return flag_row.array().binaryExpr(flag.array(), [](bool x, bool y){return x||y;});
}

std::string
MeasurementSet::ChannelWrapper::stokes(std::string const &pol, std::string const &column) const {
	std::ostringstream sstr;
	sstr << "mscal.stokes(" << column << ", '" << pol << "')";
	return sstr.str();
}

Vector<t_real> MeasurementSet::ChannelWrapper::raw_spectral_window(std::string const &stuff) const {
	std::ostringstream sstr;
	sstr << stuff << "[" << channel_ << "]";
	return table_column<t_real>(ms_.table("SPECTRAL_WINDOW"), sstr.str());
}

MeasurementSet::Direction
MeasurementSet::direction(t_real tolerance, std::string const &filter) const {
	auto const field_ids_raw = column<::casacore::Int>("FIELD_ID", filter);
	auto const source_ids_raw = table_column<::casacore::Int>(table("FIELD"), "SOURCE_ID");
	std::set<::casacore::Int> source_ids;
	for(Eigen::DenseIndex i(0); i < field_ids_raw.size(); ++i) {
		assert(field_ids_raw(i) < source_ids_raw.size());
		source_ids.insert(source_ids_raw(field_ids_raw(i)));
	}
	if(source_ids.size() == 0 and source_ids_raw.size() > 0) {
		PURIPSI_DEBUG("Could not find sources. Try different filter, no matching data in channel. "
				"Currently using filter: "
				+ filter);
		Vector<t_real> original(2);
		original(0) = 0.;
		original(1) = 0.;
		return original;
	} else if(source_ids_raw.size() == 0)
		throw std::runtime_error("Could not find sources. Cannot determine direction");
	auto const directions = table_column<::casacore::Double>(table("SOURCE"), "DIRECTION");
	auto const original = directions.row(*source_ids.begin());
	for(auto const other : source_ids)
		if(not directions.row(other).isApprox(original, tolerance))
			throw std::runtime_error("Found more than one direction");
	return original;
}

MeasurementSet::const_iterator &MeasurementSet::const_iterator::operator++() {
	++channel_;
	wrapper_ = std::make_shared<value_type>(channel_, ms_, filter_);
	return *this;
}

MeasurementSet::const_iterator MeasurementSet::const_iterator::operator++(int) {
	operator++();
	return const_iterator(channel_ - 1, ms_, filter_);
}

bool MeasurementSet::const_iterator::operator==(const_iterator const &c) const {
	if(not same_measurement_set(c))
		throw std::runtime_error("Iterators are not over the same measurement set");
	return channel_ == c.channel_;
}

utilities::vis_params
read_measurementset(std::string const &filename,
		const MeasurementSet::ChannelWrapper::polarization polarization,
		const std::vector<t_int> &channels_input, std::string const &filter) {
	auto const ms_file = puripsi::casa::MeasurementSet(filename);
	return read_measurementset(ms_file, polarization, channels_input, filter);
};

utilities::vis_params
read_measurementset(MeasurementSet const &ms_file,
		const MeasurementSet::ChannelWrapper::polarization polarization,
		const std::vector<t_int> &channels_input, std::string const &filter) {

	utilities::vis_params uv_data;
	t_uint rows = 0;
	std::vector<t_int> channels = channels_input;
	if(channels.empty()) {
		PURIPSI_LOW_LOG("All Channels = {}", ms_file.size());
		Vector<t_int> temp_vector = Vector<t_int>::LinSpaced(ms_file.size(), 0, ms_file.size());
		if(temp_vector.size() == 1) // fixing unwanted behavior of LinSpaced when ms_file.size() = 1
			temp_vector(0) = 0;
		channels = std::vector<t_int>(temp_vector.data(), temp_vector.data() + temp_vector.size());
	}

	// counting number of rows
	for(auto channel_number : channels) {
		rows += ms_file[std::make_pair(channel_number, filter)].size();
	}

	PURIPSI_LOW_LOG("Visibilities = {}", rows);
	if(rows != 0){
		uv_data.u = Vector<t_real>::Zero(rows);
		uv_data.v = Vector<t_real>::Zero(rows);
		uv_data.w = Vector<t_real>::Zero(rows);
		uv_data.vis = Vector<t_complex>::Zero(rows);
		uv_data.weights = Vector<t_complex>::Zero(rows);
		uv_data.ra = ms_file[channels[0]].right_ascension(); // convert directions from radians to degrees
		uv_data.dec = ms_file[channels[0]].declination();
		// calculate average frequency
		uv_data.average_frequency = average_frequency(ms_file, filter, channels);

		// add data to channel
		t_uint row = 0;

		for(auto channel_number : channels) {
			PURIPSI_DEBUG("Adding channel {} to plane...", channel_number);
			if(channel_number < ms_file.size()) {
				auto const channel = ms_file[std::make_pair(channel_number, filter)];
				if(channel.size() > 0) {
					if(uv_data.ra != channel.right_ascension() or uv_data.dec != channel.declination())
						throw std::runtime_error("Channels contain multiple pointings.");
					Vector<t_real> const frequencies = channel.frequencies();
					uv_data.u.segment(row, channel.size()) = channel.lambda_u();
					uv_data.v.segment(row, channel.size()) = -channel.lambda_v();
					uv_data.w.segment(row, channel.size()) = channel.lambda_w();
					t_real const the_casa_factor = 2;
					switch(polarization) {
					case MeasurementSet::ChannelWrapper::polarization::I:
					{
						uv_data.vis.segment(row, channel.size()) = channel.I("DATA"); // *0.5
						uv_data.weights.segment(row, channel.size()).real()
            										  = channel.wI(MeasurementSet::ChannelWrapper::Sigma::OVERALL); // * the_casa_factor
						// go for sigma rather than sigma_spectrum sqrt(weights)*2
						break;
					}
					case MeasurementSet::ChannelWrapper::polarization::Q:
						uv_data.vis.segment(row, channel.size()) = channel.Q("DATA") * 0.5;
						uv_data.weights.segment(row, channel.size()).real()
            										  = channel.wQ(MeasurementSet::ChannelWrapper::Sigma::OVERALL)
													  * the_casa_factor; // go for sigma rather than sigma_spectrum
						break;
					case MeasurementSet::ChannelWrapper::polarization::U:
						uv_data.vis.segment(row, channel.size()) = channel.U("DATA") * 0.5;
						uv_data.weights.segment(row, channel.size()).real()
            										  = channel.wU(MeasurementSet::ChannelWrapper::Sigma::OVERALL)
													  * the_casa_factor; // go for sigma rather than sigma_spectrum
						break;
					case MeasurementSet::ChannelWrapper::polarization::V:
						uv_data.vis.segment(row, channel.size()) = channel.V("DATA") * 0.5;
						uv_data.weights.segment(row, channel.size()).real()
            										  = channel.wV(MeasurementSet::ChannelWrapper::Sigma::OVERALL)
													  * the_casa_factor; // go for sigma rather than sigma_spectrum
						break;
					case MeasurementSet::ChannelWrapper::polarization::LL:
						uv_data.vis.segment(row, channel.size()) = channel.LL("DATA");
						uv_data.weights.segment(row, channel.size()).real() = channel.wLL(
								MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
						// sigma_spectrum
						break;
					case MeasurementSet::ChannelWrapper::polarization::LR:
						uv_data.vis.segment(row, channel.size()) = channel.LR("DATA");
						uv_data.weights.segment(row, channel.size()).real() = channel.wLR(
								MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
						// sigma_spectrum
						break;
					case MeasurementSet::ChannelWrapper::polarization::RL:
						uv_data.vis.segment(row, channel.size()) = channel.RL("DATA");
						uv_data.weights.segment(row, channel.size()).real() = channel.wRL(
								MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
						// sigma_spectrum
						break;
					case MeasurementSet::ChannelWrapper::polarization::RR:
						uv_data.vis.segment(row, channel.size()) = channel.RR("DATA");
						uv_data.weights.segment(row, channel.size()).real() = channel.wRR(
								MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
						// sigma_spectrum
						break;
					case MeasurementSet::ChannelWrapper::polarization::XX:
						uv_data.vis.segment(row, channel.size()) = channel.XX("DATA");
						uv_data.weights.segment(row, channel.size()).real() = channel.wXX(
								MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
						// sigma_spectrum
						break;
					case MeasurementSet::ChannelWrapper::polarization::XY:
						uv_data.vis.segment(row, channel.size()) = channel.XY("DATA");
						uv_data.weights.segment(row, channel.size()).real() = channel.wXY(
								MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
						// sigma_spectrum
						break;
					case MeasurementSet::ChannelWrapper::polarization::YX:
						uv_data.vis.segment(row, channel.size()) = channel.YX("DATA");
						uv_data.weights.segment(row, channel.size()).real() = channel.wYX(
								MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
						// sigma_spectrum
						break;
					case MeasurementSet::ChannelWrapper::polarization::YY:
						uv_data.vis.segment(row, channel.size()) = channel.YY("DATA");
						uv_data.weights.segment(row, channel.size()).real() = channel.wYY(
								MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
						// sigma_spectrum
						break;
					case MeasurementSet::ChannelWrapper::polarization::P:
						t_complex I(0., 1.);
						uv_data.vis.segment(row, channel.size()) = channel.Q("DATA") + I * channel.U("DATA");
						uv_data.weights.segment(row, channel.size()).real()
            										  = (channel.wQ(MeasurementSet::ChannelWrapper::Sigma::OVERALL).array()
            												  * channel.wQ(MeasurementSet::ChannelWrapper::Sigma::OVERALL).array()
															  + channel.wU(MeasurementSet::ChannelWrapper::Sigma::OVERALL).array()
															  * channel.wU(MeasurementSet::ChannelWrapper::Sigma::OVERALL).array())
															  .sqrt(); // go for sigma rather than
						// sigma_spectrum
						break;
					}
					row += channel.size();
				}
			}
		}
		// make consistent with vis file format exported from casa
		uv_data.weights = 1. / uv_data.weights.array();
		uv_data.units = utilities::vis_units::lambda;
		return uv_data;
	}else{
		uv_data.u = Vector<t_real>::Zero(0);
		return uv_data;
	}
}

utilities::vis_params
read_measurementset(MeasurementSet const &ms_file,
		const t_int channel_index, const t_real average_frequency,
		const MeasurementSet::ChannelWrapper::polarization polarization,
		std::string const &filter) {
	// CHANGE IMPLEMENTATION TO REDUCE NUMBER OF CALLS TO ms_file[...]
	utilities::vis_params uv_data;
	auto const channel = ms_file[std::make_pair(channel_index, filter)]; // this instructions takes a lot of time!
	t_uint rows = channel.size();
	uv_data.ra = channel.right_ascension(); // convert directions from radians to degrees
	uv_data.dec = channel.declination();
	PURIPSI_DEBUG("Adding channel {} to plane...", channel_index);
	PURIPSI_LOW_LOG("Visibilities = {}", rows);
	Vector<t_complex> tmp_vis = Vector<t_complex>::Zero(rows);
	Vector<t_complex> tmp_weights = Vector<t_complex>::Zero(rows);
	uv_data.u = Vector<t_real>::Zero(0);
	if(channel_index < ms_file.size()) {
		// auto const channel = ms_file[std::make_pair(channel_index, filter)];
		if(rows > 0) {
			if(uv_data.ra != channel.right_ascension() or uv_data.dec != channel.declination())
				throw std::runtime_error("Channels contain multiple pointings.");
			uv_data.u = channel.lambda_u();
			uv_data.v = -channel.lambda_v();
			uv_data.w = channel.lambda_w();

			// calculate average frequency
			uv_data.average_frequency = average_frequency;

			// // compute the final flags
			// Vector<bool> flags = (channel.flag().array()).binaryExpr(channel.flag_row().array(), [](bool x, bool y){return x||y;});
			// Vector<t_real> const frequencies = channel.frequencies();

			t_real const the_casa_factor = 2;
			switch(polarization) {
			case MeasurementSet::ChannelWrapper::polarization::I:
				tmp_vis = channel.I("DATA"); // * 0.5;
				tmp_weights.real()
            										= channel.wI(MeasurementSet::ChannelWrapper::Sigma::OVERALL);
				//* the_casa_factor; // go for sigma rather than sigma_spectrum sqrt(weights)*2
				break;
			case MeasurementSet::ChannelWrapper::polarization::Q:
				tmp_vis = channel.Q("DATA") * 0.5;
				tmp_weights.real()
            										= channel.wQ(MeasurementSet::ChannelWrapper::Sigma::OVERALL)
													* the_casa_factor; // go for sigma rather than sigma_spectrum
				break;
			case MeasurementSet::ChannelWrapper::polarization::U:
				tmp_vis = channel.U("DATA") * 0.5;
				tmp_weights.real()
            										= channel.wU(MeasurementSet::ChannelWrapper::Sigma::OVERALL)
													* the_casa_factor; // go for sigma rather than sigma_spectrum
				break;
			case MeasurementSet::ChannelWrapper::polarization::V:
				tmp_vis = channel.V("DATA") * 0.5;
				tmp_weights.real()
            										= channel.wV(MeasurementSet::ChannelWrapper::Sigma::OVERALL)
													* the_casa_factor; // go for sigma rather than sigma_spectrum
				break;
			case MeasurementSet::ChannelWrapper::polarization::LL:
				tmp_vis = channel.LL("DATA");
				tmp_weights.real() = channel.wLL(
						MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
				// sigma_spectrum
				break;
			case MeasurementSet::ChannelWrapper::polarization::LR:
				tmp_vis = channel.LR("DATA");
				tmp_weights.real() = channel.wLR(
						MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
				// sigma_spectrum
				break;
			case MeasurementSet::ChannelWrapper::polarization::RL:
				tmp_vis = channel.RL("DATA");
				tmp_weights.real() = channel.wRL(
						MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
				// sigma_spectrum
				break;
			case MeasurementSet::ChannelWrapper::polarization::RR:
				tmp_vis = channel.RR("DATA");
				tmp_weights.real() = channel.wRR(
						MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
				// sigma_spectrum
				break;
			case MeasurementSet::ChannelWrapper::polarization::XX:
				tmp_vis = channel.XX("DATA");
				tmp_weights.real() = channel.wXX(
						MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
				// sigma_spectrum
				break;
			case MeasurementSet::ChannelWrapper::polarization::XY:
				tmp_vis = channel.XY("DATA");
				tmp_weights.real() = channel.wXY(
						MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
				// sigma_spectrum
				break;
			case MeasurementSet::ChannelWrapper::polarization::YX:
				tmp_vis = channel.YX("DATA");
				tmp_weights.real() = channel.wYX(
						MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
				// sigma_spectrum
				break;
			case MeasurementSet::ChannelWrapper::polarization::YY:
				tmp_vis = channel.YY("DATA");
				tmp_weights.real() = channel.wYY(
						MeasurementSet::ChannelWrapper::Sigma::OVERALL); // go for sigma rather than
				// sigma_spectrum
				break;
			case MeasurementSet::ChannelWrapper::polarization::P:
				t_complex I(0., 1.);
				tmp_vis = channel.Q("DATA") + I * channel.U("DATA");
				tmp_weights.real()
            										= (channel.wQ(MeasurementSet::ChannelWrapper::Sigma::OVERALL).array()
            												* channel.wQ(MeasurementSet::ChannelWrapper::Sigma::OVERALL).array()
															+ channel.wU(MeasurementSet::ChannelWrapper::Sigma::OVERALL).array()
															* channel.wU(MeasurementSet::ChannelWrapper::Sigma::OVERALL).array())
															.sqrt(); // go for sigma rather than
				// sigma_spectrum
				break;
			}
		}
	}

	uv_data.vis = tmp_vis;
	uv_data.weights = tmp_weights;

	// make consistent with vis file format exported from casa
	uv_data.weights = 1. / uv_data.weights.array();
	uv_data.units = utilities::vis_units::lambda;
	return uv_data;

}

std::vector<utilities::vis_params>
read_measurementset_channels(std::string const &filename,
		const MeasurementSet::ChannelWrapper::polarization pol,
		const t_int &channel_width, std::string const &filter) {
	// Read and average the channels into a vector of vis_params
	std::vector<utilities::vis_params> channels_vis;
	auto const ms_file = puripsi::casa::MeasurementSet(filename);
	t_int const total_channels = ms_file.size();
	t_int const planes = (channel_width == 0) ? 1 : std::floor(total_channels / channel_width);
	PURIPSI_DEBUG("Number of planes {} ...", planes);
	for(int i = 0; i < planes; i++) {
		PURIPSI_DEBUG("Reading plane {} ...", i);
		t_int const end = std::min((i + 1) * channel_width, total_channels);
		Vector<t_int> temp_block = Vector<t_int>::LinSpaced(channel_width, i * channel_width, end);
		if(channel_width == 1 or total_channels == i)
			temp_block(0) = i;
		auto const block = std::vector<t_int>(temp_block.data(), temp_block.data() + temp_block.size());
		auto const measurement = read_measurementset(ms_file, pol, block, filter);
		if(measurement.u.size() != 0){
			channels_vis.emplace_back(measurement);
		}
	}
	return channels_vis;
};

t_real average_frequency(const puripsi::casa::MeasurementSet &ms_file, std::string const &filter,
		const std::vector<t_int> &channels) {

	// calculate average frequency
	t_real frequency_sum = 0;
	t_real width_sum = 0.;
	for(auto channel_number : channels) {
		auto const channel = ms_file[channel_number];
		auto const frequencies = channel.frequencies();
		auto const width = channel.width();
		frequency_sum += (frequencies.array() * width.array()).sum();
		width_sum += width.sum();
	}
	return frequency_sum / width_sum / 1e6;
}

t_uint MeasurementSet::ChannelWrapper::size() const {
	if(ms_.table().nrow() == 0)
		return 0;
	std::ostringstream sstr;
	// sstr << "USING STYLE PYTHON SELECT FLAG[" << channel_ << ",] as R FROM $1 WHERE NOT any(FLAG["
	//      << channel_ << ",])";
	sstr << "USING STYLE PYTHON SELECT FLAG[" << channel_ << ",] as R FROM $1";
	if(not filter_.empty())
		sstr << " WHERE " << filter_; // "AND "
	auto const taql_table = ::casacore::tableCommand(sstr.str(), ms_.table());
	auto const vtable = taql_table.table();
	return vtable.nrow();
}

MeasurementSet::const_iterator &MeasurementSet::const_iterator::operator+=(t_int n) {
	channel_ += n;
	return *this;
}

bool MeasurementSet::const_iterator::operator>(const_iterator const &c) const {
	if(not same_measurement_set(c))
		throw std::runtime_error("Iterators are not over the same measurement set");
	return channel_ > c.channel_;
}

bool MeasurementSet::const_iterator::operator>=(const_iterator const &c) const {
	if(not same_measurement_set(c))
		throw std::runtime_error("Iterators are not over the same measurement set");
	return channel_ >= c.channel_;
}
} // namespace casa
} // namespace puripsi
