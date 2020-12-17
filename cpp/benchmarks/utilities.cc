#include <sstream>
#include <fstream>
#include <psi/linear_transform.h>
#include "puripsi/directories.h"
#include "puripsi/pfitsio.h"
#include "puripsi/operators.h"
#include <benchmarks/utilities.h>

using namespace puripsi;
using namespace puripsi::notinstalled;

namespace b_utilities {

  void Arguments(benchmark::internal::Benchmark* b) {
    int im_size_max = 4096; // 4096
    int uv_size_max = 10000000; // 1M, 10M, 100M
    int kernel_max = 16; // 16
    for (int i=128; i<=im_size_max; i*=2)
      for (int j=1000000; j<=uv_size_max; j*=10)
        for (int k=2; k<=kernel_max; k*=2)
          if (k*k<i)
            b->Args({i,j,k});
  }


  double duration(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end){
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    return elapsed_seconds.count();
  }


  bool updateImage(t_uint newSize, Image<t_complex>& image, t_uint& sizex, t_uint& sizey) {
    if (sizex==newSize) {
      return false;
    }
    const std::string &name = "M31_"+std::to_string(newSize);
    std::string const fitsfile = image_filename(name + ".fits");
    std::cout << "Reading image: " << fitsfile << "\n";
    image = pfitsio::read2d(fitsfile);
    sizex = image.cols();
    sizey = image.rows();
    t_real const max = image.array().abs().maxCoeff();
    image = image * 1. / max;
    return true;
  }

  bool updateEmptyImage(t_uint newSize, Vector<t_complex>& image, t_uint& sizex, t_uint& sizey) {
    if (sizex==newSize) {
      return false;
    }
    image.resize(newSize*newSize);
    sizex = newSize;
    sizey = newSize;
    return true;
  }

  bool updateMeasurements(t_uint newSize, utilities::vis_params& data) {
    if (data.vis.size()==newSize) {
      return false;
    }
    data = b_utilities::random_measurements(newSize);
    return true;
  }

  bool updateMeasurements(t_uint newSize, utilities::vis_params& data, t_real& epsilon, bool newImage, Image<t_complex>& image) {
    if (data.vis.size()==newSize && !newImage) {
      return false;
    }
    const t_real FoV = 1;      // deg
    const t_real cellsize = FoV / image.size() * 60. * 60.;
    std::tuple<utilities::vis_params, t_real> temp =
      b_utilities::dirty_measurements(image, newSize, 30., cellsize);
    data = std::get<0>(temp);
    epsilon = utilities::calculate_l2_radius(data.vis,  std::get<1>(temp));
   
    return true;
  }

 
  std::tuple<utilities::vis_params, t_real>
  dirty_measurements(Image<t_complex> const &ground_truth_image, t_uint number_of_vis, t_real snr,
		     const t_real &cellsize) {
    auto uv_data = random_measurements(number_of_vis);
    // creating operator to generate measurements
    auto measurement_op = std::make_shared<psi::LinearTransform<psi::Vector<psi::t_complex>>>(puripsi::operators::MeasurementOperator<Vector<t_complex>, t_complex>(
	  uv_data, ground_truth_image.rows(), ground_truth_image.cols(), cellsize, cellsize,
          2, 0, 1e-4, kernels::kernel::kb, 8, 8, false));
    // Generates measurements from image
    uv_data.vis = (*measurement_op)
      * Image<t_complex>::Map(ground_truth_image.data(), ground_truth_image.size(), 1);

    // working out value of signal given SNR
    auto const sigma = utilities::SNR_to_standard_deviation(uv_data.vis, snr);
    // adding noise to visibilities
    uv_data.vis = utilities::add_noise(uv_data.vis, 0., sigma);
    return std::make_tuple(uv_data, sigma);
}


  utilities::vis_params random_measurements(t_int size) {
    std::stringstream filename;
    filename << "random_" << size << ".vis";
    std::string const vis_file = visibility_filename(filename.str());
    std::ifstream vis_file_str(vis_file);

    utilities::vis_params uv_data;
    if (vis_file_str.good()) {
      uv_data = utilities::read_visibility(vis_file, false);
      uv_data.units = utilities::vis_units::radians;
    }
    else {
      t_real const sigma_m = constant::pi / 3;
      const t_real max_w = 100.; // lambda
      uv_data = utilities::random_sample_density(size, 0, sigma_m, max_w);
      uv_data.units = utilities::vis_units::radians;
      utilities::write_visibility(uv_data, vis_file);
    }
    return uv_data;
  }

} 
