#include <array>
#include <benchmark/benchmark.h>
#include <psi/primal_dual.h>
#include <psi/relative_variation.h>
#include <psi/utilities.h>
#include <psi/wavelets.h>
#include <psi/wavelets/sara.h>
#include <psi/power_method.h>
#include "puripsi/operators.h"
#include "puripsi/types.h"
#include "puripsi/utilities.h"
#include "benchmarks/utilities.h"

using namespace puripsi;

class PrimaldualFixture : public ::benchmark::Fixture
{
public:
  void SetUp(const ::benchmark::State& state) {
    // Reading image from file and update related quantities
    bool newImage = b_utilities::updateImage(state.range(0), m_image, m_imsizex, m_imsizey);
    
    // Generating random uv(w) coverage and update related quantities
    bool newMeasurements = b_utilities::updateMeasurements(state.range(1), m_uv_data, m_epsilon, newImage, m_image);
    
    bool newKernel = m_kernel!=state.range(2);
    if (newImage || newMeasurements || newKernel) {
      m_kernel = state.range(2);
      // creating the measurement operator
      const t_real FoV = 1;      // deg
      const t_real cellsize = FoV / m_imsizex * 60. * 60.;
      const bool w_term = false;
      auto const over_sample = 2;
      m_measurements_transform =  std::make_shared<psi::LinearTransform<psi::Vector<psi::t_complex>>>(puripsi::operators::MeasurementOperator<Vector<t_complex>, t_complex>(
												 m_uv_data, m_imsizey, m_imsizex, cellsize, cellsize, over_sample, 0, 0.0001, kernels::kernel::kb,  m_kernel, m_kernel, w_term));
      auto const gamma = (m_measurements_transform->adjoint() * m_uv_data.vis).real().maxCoeff() * 1e-3;

      m_nlevels = m_sara.size();

      psi::LinearTransform<Vector<t_complex>> Psi = psi::linear_transform<t_complex>(m_sara, m_imsizey, m_imsizex);

      psi::Vector<t_complex> rand = psi::Vector<t_complex>::Random(m_imsizey * m_imsizex * m_nlevels);

      auto const pm = psi::algorithm::PowerMethod<psi::t_complex>().tolerance(1e-6);

      auto const nu1data = pm.AtA(Psi, rand);
      auto const nu1 = nu1data.magnitude.real();
      m_sigma1 = 1e0 / nu1;
      rand = psi::Vector<t_complex>::Random(m_imsizey * m_imsizex * (over_sample/2));
      
      auto const nu2data = pm.AtA(*m_measurements_transform, rand);
      auto const nu2 = nu2data.magnitude.real();
      m_sigma2 = 1e0 / nu2;      
      
      m_kappa = ((m_measurements_transform->adjoint() * m_uv_data.vis).real().maxCoeff() * 1e-3) / nu2;  

      // create the pd algorithm
      m_pd = std::make_shared<psi::algorithm::PrimalDual<t_complex>>(m_uv_data.vis);
      m_pd->itermax(2)
	.tau(m_tau)
	.kappa(m_kappa)
	.sigma1(m_sigma1)
	.sigma2(m_sigma2)
	.levels(m_nlevels)
	.l2ball_epsilon(m_epsilon)
	.nu(nu2)
	.relative_variation(1e-3)
	.positivity_constraint(true)
	.residual_convergence(m_epsilon * 1.001)
	.Psi(Psi)
	.Phi(*m_measurements_transform);
    }
  }
  
  void TearDown(const ::benchmark::State& state) {
  }

  t_uint m_counter;  
  const  psi::wavelets::SARA m_sara{std::make_tuple("Dirac", 3u), std::make_tuple("DB1", 3u), std::make_tuple("DB2", 3u),
      std::make_tuple("DB3", 3u),   std::make_tuple("DB4", 3u), std::make_tuple("DB5", 3u),
      std::make_tuple("DB6", 3u),   std::make_tuple("DB7", 3u), std::make_tuple("DB8", 3u)};

  Image<t_complex> m_image;
  t_uint m_imsizex;
  t_uint m_imsizey;
 
  utilities::vis_params m_uv_data;
  t_real m_epsilon;
  t_real m_tau;
  t_real m_kappa;
  t_real m_sigma1;
  t_real m_sigma2;

  t_uint m_kernel;
  std::shared_ptr<psi::LinearTransform<Vector<t_complex>>> m_measurements_transform;
  t_real m_nlevels;
  std::shared_ptr<psi::algorithm::PrimalDual<t_complex>> m_pd;
};


BENCHMARK_DEFINE_F(PrimaldualFixture, Apply)(benchmark::State &state) {
  // Benchmark the application of the algorithm
  while(state.KeepRunning()) {
    auto start = std::chrono::high_resolution_clock::now();
    (*m_pd)();
    auto end   = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(b_utilities::duration(start,end));
    }
  
  }

BENCHMARK_REGISTER_F(PrimaldualFixture, Apply)
//->Apply(b_utilities::Arguments)
->Args({1024,1000000,4})->Args({1024,10000000,4})
->UseManualTime()
->Repetitions(5)//->ReportAggregatesOnly(true)
->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
