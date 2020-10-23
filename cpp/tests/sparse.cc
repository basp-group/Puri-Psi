#include "puripsi/config.h"
#include <catch2/catch.hpp>
#include "puripsi/directories.h"
#include "puripsi/logging.h"
#include "puripsi/types.h"
using namespace puripsi;
using namespace puripsi::notinstalled;

TEST_CASE("sparse [iterator]", "[iterator]") {
  t_int NZnum = 40;
  std::vector<t_tripletList> entries;
  entries.reserve(NZnum);
  Vector<t_int> rows = Vector<t_int>::LinSpaced(NZnum, 0, 2 * NZnum);
  Vector<t_int> cols = Vector<t_int>::LinSpaced(NZnum, 0, 3 * NZnum);

  for(t_int i = 0; i < NZnum; ++i) {
    entries.emplace_back(rows(i), cols(i), i);
  }
  const t_int total_rows = 5 * NZnum;
  const t_int total_cols = 4 * NZnum;
  Sparse<t_complex> G(total_rows, total_cols);
  G.setFromTriplets(entries.begin(), entries.end());
  for(t_int i = 0; i < total_cols; ++i) {
    for(Sparse<t_complex>::InnerIterator pix(G, i); pix; ++pix) {
      PURIPSI_HIGH_LOG("{} {} {}", pix.row(), pix.col(), std::real(pix.value()));
    }
  }
  // CHECK(expected.isApprox(output, 1e-13));
}
