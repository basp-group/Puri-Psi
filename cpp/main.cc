#include <array>
#include <ctime>
#include <random>
#include <cstddef>
#include <psi/primal_dual.h>
#include <psi/positive_quadrant.h>
#include <psi/relative_variation.h>
#include <psi/reweighted.h>
#include <psi/utilities.h>
#include <psi/wavelets.h>
#include <psi/wavelets/sara.h>
#include <psi/power_method.h>
#include "cmdl.h"
#include "puripsi/operators.h"
#include "puripsi/casacore.h"
#include "puripsi/logging.h"
#include "puripsi/pfitsio.h"
#include "puripsi/types.h"

using namespace puripsi;
namespace {

// to be modified (include fmax, see Matlab codes to get the nominal frequency and
// the proper scaling of the u-v coverage)
void bandwidth_scaling(puripsi::utilities::vis_params const &uv_data, puripsi::Params &params) {
	t_real const max_u = std::sqrt((uv_data.u.array() * uv_data.u.array()).maxCoeff());
	t_real const max_v = std::sqrt((uv_data.v.array() * uv_data.v.array()).maxCoeff());
	if(params.cellsizex == 0 and params.cellsizey == 0) {
		t_real const max = std::sqrt(
				(uv_data.u.array() * uv_data.u.array() + uv_data.v.array() * uv_data.v.array()).maxCoeff());
		params.cellsizex = (180 * 3600) / max / constant::pi / 2;
		params.cellsizey = (180 * 3600) / max / constant::pi / 2;
	}
	if(params.cellsizex == 0)
		params.cellsizex = (180 * 3600) / max_u / constant::pi / 2;
	if(params.cellsizey == 0)
		params.cellsizey = (180 * 3600) / max_v / constant::pi / 2;
}

pfitsio::header_params
create_new_header(puripsi::utilities::vis_params const &uv_data, puripsi::Params const &params) {
	// header information
	pfitsio::header_params header;
	header.mean_frequency = uv_data.average_frequency;
	header.ra = uv_data.ra;
	header.dec = uv_data.dec;
	header.cell_x = params.cellsizex;
	header.cell_y = params.cellsizey;
	header.residual_convergence = params.residual_convergence;
	header.relative_variation = params.relative_variation;
	return header;
}

// unused...
t_real estimate_noise(puripsi::Params const &params) {

	// Read in visibilities for noise estimate
	t_real sigma_real = 1 / std::sqrt(2);
	t_real sigma_imag = 1 / std::sqrt(2);

	if(params.noisefile != "") {
		auto const noise_uv_data = puripsi::casa::read_measurementset(
				params.noisefile, puripsi::casa::MeasurementSet::ChannelWrapper::polarization::V);
		Vector<t_complex> const noise_vis = noise_uv_data.weights.array() * noise_uv_data.vis.array();
		sigma_real = utilities::median(noise_vis.real().cwiseAbs()) / 0.6745;
		sigma_imag = utilities::median(noise_vis.imag().cwiseAbs()) / 0.6745;
	}

	PURIPSI_MEDIUM_LOG("RMS noise of {}Jy + i{}Jy", sigma_real, sigma_real);
	return std::sqrt(sigma_real * sigma_real + sigma_imag * sigma_imag); //calculation is for combined real and imaginary sigma, factor of 1/sqrt(2) in epsilon calculation
}

puripsi::casa::MeasurementSet::ChannelWrapper::polarization choose_pol(std::string const & stokes){
	/*
   Chooses the polarisation to read from a measurement set.
	 */
	auto stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::I;
	//stokes
	if (stokes == "I" or stokes == "i")
		stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::I;
	if (stokes == "Q" or stokes == "q")
		stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::Q;
	if (stokes == "U" or stokes == "u")
		stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::U;
	if (stokes == "V" or stokes == "v")
		stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::V;
	//linear
	if (stokes == "XX" or stokes == "xx")
		stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::XX;
	if (stokes == "YY" or stokes == "yy")
		stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::YY;
	if (stokes == "XY" or stokes == "xy")
		stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::XY;
	if (stokes == "YX" or stokes == "yx")
		stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::YX;
	//circular
	if (stokes == "LL" or stokes == "ll")
		stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::LL;
	if (stokes == "RR" or stokes == "rr")
		stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::RR;
	if (stokes == "LR" or stokes == "lr")
		stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::LR;
	if (stokes == "RL" or stokes == "rl")
		stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::RL;
	if (stokes == "P" or stokes == "p")
		stokes_val = puripsi::casa::MeasurementSet::ChannelWrapper::polarization::P;
	return stokes_val;
}

t_real save_psf_and_dirty_image(
		std::shared_ptr<psi::LinearTransform<psi::Vector<psi::t_complex>> const> const &measurements,
		puripsi::utilities::vis_params const &uv_data, puripsi::Params const &params) {
	// returns psf normalisation
	puripsi::pfitsio::header_params header = create_new_header(uv_data, params);
	std::string const dirty_image_fits = params.name + "_dirty_" + params.weighting;
	std::string const psf_fits = params.name + "_psf_" + params.weighting;
	Vector<t_complex> const psf_image = measurements->adjoint() * (uv_data.weights.array());
	Image<t_real> psf = Image<t_complex>::Map(psf_image.data(), params.height, params.width).real();
	t_real max_val = psf.array().abs().maxCoeff();
	PURIPSI_LOW_LOG("PSF peak is {}", max_val);
	psf = psf;//not normalised, so it is easy to compare scales
	header.fits_name = psf_fits + ".fits";
	PURIPSI_HIGH_LOG("Saving {}", header.fits_name);
	pfitsio::write2d_header(psf, header);
	Vector<t_complex> const dirty_image = measurements->adjoint() * uv_data.vis;
	Image<t_complex> dimage
	= Image<t_complex>::Map(dirty_image.data(), params.height, params.width);
	header.fits_name = dirty_image_fits + ".fits";
	PURIPSI_HIGH_LOG("Saving {}", header.fits_name);
	pfitsio::write2d_header(dimage.real(), header);
	if(params.stokes_val == puripsi::casa::MeasurementSet::ChannelWrapper::polarization::P){
		header.fits_name = dirty_image_fits + "_imag.fits";
		PURIPSI_HIGH_LOG("Saving {}", header.fits_name);
		pfitsio::write2d_header(dimage.imag(), header);
	}
	return max_val;
}

void save_final_image(std::string const &outfile_fits, std::string const &residual_fits,
		Vector<t_complex> const &x, utilities::vis_params const &uv_data,
		Params const &params,
		std::shared_ptr<psi::LinearTransform<psi::Vector<psi::t_complex>> const> const &measurements) {

	//! Save final output image
	puripsi::pfitsio::header_params header = create_new_header(uv_data, params);
	Image<t_complex> const image = Image<t_complex>::Map(x.data(),params.height, params.width);
	// header information
	header.pix_units = "JY/PIXEL";
	header.niters = params.iter;
	header.epsilon = params.epsilon;
	header.fits_name = outfile_fits + ".fits";
	pfitsio::write2d_header(image.real(), header);
	if(params.stokes_val == puripsi::casa::MeasurementSet::ChannelWrapper::polarization::P){
		header.fits_name = outfile_fits + "_imag.fits";
		pfitsio::write2d_header(image.real(), header);
	}
	Vector<t_complex> const res_vector = measurements->adjoint() * (uv_data.vis - *measurements * x);
	const Image<t_complex> residual
	= Image<t_complex>::Map(res_vector.data(), params.height, params.width);
	header.pix_units = "JY/PIXEL";
	header.fits_name = residual_fits + ".fits";
	pfitsio::write2d_header(residual.real(), header);
	if(params.stokes_val == puripsi::casa::MeasurementSet::ChannelWrapper::polarization::P){
		header.fits_name = residual_fits + "_imag.fits";
		pfitsio::write2d_header(residual.real(), header);
	}
};

std::tuple<Vector<t_complex>, Vector<t_complex>> read_estimates(
		std::shared_ptr<psi::LinearTransform<psi::Vector<psi::t_complex>> const> const &measurements,
		puripsi::utilities::vis_params const &uv_data, puripsi::Params const &params) {
	Vector<t_complex> initial_estimate = measurements->adjoint() * uv_data.vis;
	Vector<t_complex> initial_residuals = Vector<t_complex>::Zero(uv_data.vis.size());
	// loading data from check point.
	if(utilities::file_exists(params.name + "_diagnostic") and params.warmstart == true) {
		PURIPSI_HIGH_LOG("Loading checkpoint for {}", params.name.c_str());
		std::string const outfile_fits = params.name + "_solution_" + params.weighting + "_update.fits";
		if(utilities::file_exists(outfile_fits)) {
			auto const image = pfitsio::read2d(outfile_fits);
			if(params.height != image.rows() or params.width != image.cols()) {
				std::runtime_error("Initial model estimate is the wrong size.");
			}
			initial_estimate = Matrix<t_complex>::Map(image.data(), image.size(), 1);
			Vector<t_complex> const model = *measurements * initial_estimate;
			initial_residuals = uv_data.vis - model;
		}
	}
	std::tuple<Vector<t_complex>, Vector<t_complex>> const estimates(initial_estimate,
			initial_residuals);
	return estimates;
}

}

int main(int argc, char **argv) {
	psi::logging::initialize();
	puripsi::logging::initialize();

	Params params = parse_cmdl(argc, argv);
	psi::logging::set_level(params.psi_logging_level);
	puripsi::logging::set_level(params.psi_logging_level);
	params.stokes_val = choose_pol(params.stokes);
	PURIPSI_HIGH_LOG("Stokes input {}", params.stokes);
	//checking if reading measurement set or .vis file
	std::size_t found = params.visfile.find_last_of(".");
	std::string format =  "." + params.visfile.substr(found+1);
	std::transform(format.begin(), format.end(), format.begin(), ::tolower);
	auto uv_data = (format == ".ms") ? puripsi::casa::read_measurementset(params.visfile, params.stokes_val) : utilities::read_visibility(params.visfile, params.use_w_term);
	bandwidth_scaling(uv_data, params);

	string algorithm = "PrimalDual";

	// calculate weights outside of measurement operator
	uv_data.weights = utilities::init_weights(
			uv_data.u, uv_data.v, uv_data.weights, params.over_sample, params.weighting, 0,
			params.over_sample * params.width, params.over_sample * params.height);
	// uv_data.weights = uv_data.weights / uv_data.weights.sum() * uv_data.weights.size();
	uv_data.vis = uv_data.vis.array() * uv_data.weights.array(); // ok
	auto const noise_rms = estimate_noise(params); // to be removed (estimate epsilon)
	std::shared_ptr<psi::LinearTransform<Vector<t_complex>>> measurements_transform;
	measurements_transform = std::make_shared<psi::LinearTransform<Vector<t_complex>>>(puripsi::operators::MeasurementOperator<Vector<t_complex>, t_complex>(
			uv_data, params.height, params.width, params.cellsizey, params.cellsizex,
			params.over_sample, params.power_method_iterations, 1e-4, params.kernel, params.J, params.J,
			params.use_w_term));

	psi::wavelets::SARA const sara{
		std::make_tuple("Dirac", 3u), std::make_tuple("DB1", 3u), std::make_tuple("DB2", 3u),
				std::make_tuple("DB3", 3u),   std::make_tuple("DB4", 3u), std::make_tuple("DB5", 3u),
				std::make_tuple("DB6", 3u),   std::make_tuple("DB7", 3u), std::make_tuple("DB8", 3u)};

	auto Psi = psi::linear_transform<t_complex>(sara, params.height, params.width);

	PURIPSI_LOW_LOG("Saving dirty map");
	params.psf_norm = save_psf_and_dirty_image(measurements_transform, uv_data, params);

	auto const estimates = read_estimates(measurements_transform, uv_data, params); // to be potentially removed

	// to be replaced by nnls
	t_real const epsilon = params.n_mu * std::sqrt(2 * uv_data.vis.size()) * noise_rms / std::sqrt(2) * params.psf_norm; // Calculation of l_2 bound following SARA paper

	params.epsilon = epsilon;
	params.residual_convergence
	= (params.residual_convergence < 0) ? 0. : params.residual_convergence * epsilon;

	std::ofstream out_diagnostic;
	out_diagnostic.precision(13);
	out_diagnostic.open(params.name + "_diagnostic", std::ios_base::app);

	Vector<t_complex> final_model = Vector<t_complex>::Zero(params.width * params.height);
	std::string outfile_fits = "";
	std::string residual_fits = "";

	if(algorithm == "PrimalDual"){

		auto const nlevels = sara.size();

		auto const tau = 0.49;
		PURIPSI_MEDIUM_LOG("tau is {} ", tau);

		psi::Vector<t_complex> rand = psi::Vector<t_complex>::Random(params.height * params.width * nlevels);
		PURIPSI_HIGH_LOG("Setting up power method to calculate sigma values for primal dual");
		auto const pm = psi::algorithm::PowerMethod<psi::t_complex>().tolerance(1e-6);

		PURIPSI_HIGH_LOG("Calculating sigma1");
		auto const nu1data = pm.AtA(Psi, rand);
		auto const nu1 = nu1data.magnitude.real();
		auto sigma1 = 1e0 / nu1;
		PURIPSI_MEDIUM_LOG("sigma1 is {} ", sigma1);

		rand = psi::Vector<t_complex>::Random(params.height * params.width * (params.over_sample/2));

		PURIPSI_HIGH_LOG("Calculating sigma2");
		auto const nu2data = pm.AtA(*measurements_transform, rand);
		auto const nu2 = nu2data.magnitude.real();
		auto sigma2 = 1e0 / nu2;
		PURIPSI_MEDIUM_LOG("sigma2 is {} ", sigma2);

		PURIPSI_HIGH_LOG("Calculating kappa");
		auto const kappa = ((measurements_transform->adjoint() * uv_data.vis).real().maxCoeff() * 1e-3) / nu2;
		PURIPSI_MEDIUM_LOG("kappa is {} ", kappa);

		auto convergence_function = [](const Vector<t_complex> &x) { return true; };

		t_uint iters = 300;
		if(params.niters != 0)
			iters = params.niters;


		PURIPSI_HIGH_LOG("Creating primal-dual Functor");
		auto pd = psi::algorithm::PrimalDual<t_complex>(uv_data.vis)
    		.itermax(iters)
			.tau(tau)
			.kappa(kappa)
			.sigma1(sigma1)
			.sigma2(sigma2)
			.levels(nlevels)
			.l2ball_epsilon(epsilon)
			.nu(nu2)
			.relative_variation(1e-5)
			.positivity_constraint(true)
			.residual_convergence(epsilon * 1.001)
			.Psi(Psi)
			.Phi(*measurements_transform)
			.is_converged(convergence_function);


		Vector<t_complex> final_model = Vector<t_complex>::Zero(params.width * params.height);
		std::string outfile_fits = "";
		std::string residual_fits = "";
		if(params.no_reweighted) {
			auto diagnostic = pd(estimates);
			outfile_fits = params.name + "_solution_" + params.weighting + "_final";
			residual_fits = params.name + "_residual_" + params.weighting + "_final";
			final_model = diagnostic.x;
		} else {
			auto const posq = psi::algorithm::positive_quadrant(pd);
			auto const min_delta = noise_rms * std::sqrt(uv_data.vis.size())
			/ std::sqrt(9 * params.height * params.width);
			// Sets weight after each pd iteration.
			// In practice, this means replacing the proximal of the l1 objective function.
			auto reweighted
			= psi::algorithm::reweighted(pd).itermax(10).min_delta(min_delta).is_converged(
					psi::RelativeVariation<std::complex<t_real>>(1e-3));
			auto diagnostic = reweighted();
			outfile_fits = params.name + "_solution_" + params.weighting + "_final_reweighted";
			residual_fits = params.name + "_residual_" + params.weighting + "_final_reweighted";
			final_model = diagnostic.algo.x;

		}
	}else{
		PURIPSI_ERROR("Incorrect algorithm chosen: {}",algorithm);
	}
	save_final_image(outfile_fits, residual_fits, final_model, uv_data, params, measurements_transform);
	out_diagnostic.close();

	return 0;
}
