#ifndef BENCHMARK_UTILITIES_H
#define BENCHMARK_UTILITIES_H

#include <chrono>
#include "puripsi/utilities.h"
#include <benchmark/benchmark.h>

using namespace puripsi;

namespace b_utilities {

  void Arguments(benchmark::internal::Benchmark* b);

  double duration(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end);

  bool updateImage(t_uint newSize, Image<t_complex>& image, t_uint& sizex, t_uint& sizey);
  bool updateEmptyImage(t_uint newSize, Vector<t_complex>& image, t_uint& sizex, t_uint& sizey);
  
  bool updateMeasurements(t_uint newSize, utilities::vis_params& data);
  bool updateMeasurements(t_uint newSize, utilities::vis_params& data, t_real& epsilon, bool newImage, Image<t_complex>& image);
  
  std::tuple<utilities::vis_params, t_real>
    dirty_measurements(Image<t_complex> const &ground_truth_image, t_uint number_of_vis, t_real snr,
		       const t_real &cellsize);

  utilities::vis_params random_measurements(t_int size);

}

#endif
