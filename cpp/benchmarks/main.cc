#include <benchmark/benchmark.h>


// This reporter does nothing.
// We can use it to disable output from all but the root process
class NullReporter : public ::benchmark::BenchmarkReporter {
public:
  NullReporter() {}
  virtual bool ReportContext(const Context &) {return true;}
  virtual void ReportRuns(const std::vector<Run> &) {}
  virtual void Finalize() {}
};

// reporter according to the process rank
int main(int argc, char const **argv) {

  ::benchmark::Initialize(&argc, const_cast<char**>(argv));

  if(world.is_root())
    // root process will use a reporter from the usual set provided by
    // ::benchmark
    ::benchmark::RunSpecifiedBenchmarks();
  else {
    // reporting from other processes is disabled by passing a custom reporter
    NullReporter null;
    ::benchmark::RunSpecifiedBenchmarks(&null);
  }

  return 0;
}
