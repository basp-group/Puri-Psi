#include "puripsi/cimg.h"
#include "puripsi/directories.h"
#include "puripsi/pfitsio.h"
#include "puripsi/types.h"
using namespace puripsi;
using namespace puripsi::notinstalled;

int main(int nargs, char const **args) {
  // up samples M31 example
  std::string const fitsfile = image_filename("M31.fits");
  std::string const fitsfile2 = image_filename("M31_512.fits");
  Image<t_real> const M31 = pfitsio::read2d(fitsfile).real();
  Image<t_real> const M31_large = pfitsio::read2d(fitsfile2).real();
#ifdef PURIPSI_CImg
  CDisplay display = cimg::make_display<Image<t_real>>(M31, "M31 small");
  while(!display.is_closed()) {
    if(display.is_key())
      display = cimg::make_display<Image<t_real>>(M31_large, "Image of M31 large");
  }
#endif
}
