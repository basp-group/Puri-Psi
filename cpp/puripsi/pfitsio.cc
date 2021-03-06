#include "puripsi/pfitsio.h"
#include "puripsi/config.h"
#include "puripsi/logging.h"
namespace puripsi {
namespace pfitsio {
//! Write image to fits file
void write2d_header(const Image<t_real> &eigen_image, const pfitsio::header_params &header,
                    const bool &overwrite) {
  /*
    Writes an image to a fits file.

    image:: image data, a 2d Image.
        header:: structure containing header information
        overwrite:: if true, overwrites old fits file with same name

  */

  if(overwrite == true) {
    remove(header.fits_name.c_str());
  };
  long naxes[4]
      = {static_cast<long>(eigen_image.rows()), static_cast<long>(eigen_image.cols()), 1, 1};
  std::unique_ptr<CCfits::FITS> pFits(new CCfits::FITS(header.fits_name, FLOAT_IMG, 4, naxes));
  long fpixel(1);
  std::vector<long> extAx = {eigen_image.rows(), eigen_image.cols()};

  std::valarray<double> image(naxes[0] * naxes[1]);
  std::copy(eigen_image.data(), eigen_image.data() + eigen_image.size(), &image[0]);
  // Writing to fits header
  pFits->pHDU().addKey("AUTHOR", "PuriPsi", "");
  pFits->pHDU().addKey("BUNIT", header.pix_units, ""); // units
  pFits->pHDU().addKey("NAXIS", 4, "");                // number of axes
  pFits->pHDU().addKey("NAXIS1", static_cast<t_int>(eigen_image.cols()), "");
  pFits->pHDU().addKey("NAXIS2", static_cast<t_int>(eigen_image.rows()), "");
  pFits->pHDU().addKey("NAXIS3", 1, "");
  pFits->pHDU().addKey("NAXIS4", 1, "");
  pFits->pHDU().addKey("CRPIX1", static_cast<t_int>(std::floor(eigen_image.cols() / 2)) + 1, "");
  pFits->pHDU().addKey("CRPIX2", static_cast<t_int>(std::floor(eigen_image.rows() / 2)) + 1, "");
  pFits->pHDU().addKey("CRPIX3", 1, "");
  pFits->pHDU().addKey("CRPIX4", 1, "");
  pFits->pHDU().addKey("CRVAL1", header.ra * 180. / puripsi::constant::pi, "");
  pFits->pHDU().addKey("CRVAL2", header.dec * 180. / puripsi::constant::pi, "");
  pFits->pHDU().addKey("CRVAL3", header.mean_frequency * std::pow(10, 6) * 1., "");
  pFits->pHDU().addKey("CRVAL4", 1, "");
  pFits->pHDU().addKey("CTYPE1", "RA---SIN", "");
  pFits->pHDU().addKey("CTYPE2", "DEC---SIN", "");
  pFits->pHDU().addKey("CTYPE3", "FREQ-OBS", "");
  pFits->pHDU().addKey("CTYPE4", "STOKES", "");
  pFits->pHDU().addKey("CDELT1", -header.cell_x / 3600., "");
  pFits->pHDU().addKey("CDELT2", header.cell_y / 3600., "");
  pFits->pHDU().addKey("CDELT3", header.channel_width * std::pow(10, 6) * 1., "");
  pFits->pHDU().addKey("CDELT4", 1, "");
  pFits->pHDU().addKey("BTYPE", "intensity", "");
  pFits->pHDU().addKey("EQUINOX", 2000, "");
  pFits->pHDU().addKey("PURIPSI-NITERS", header.niters, "");
  if(header.hasconverged) {
    pFits->pHDU().addKey("PURIPSI-CONVERGED", "T", "");
  } else {
    pFits->pHDU().addKey("PURIPSI-CONVERGED", "F", "");
  }
  pFits->pHDU().addKey("PURIPSI-RELATIVEVARIATION", header.relative_variation, "");
  pFits->pHDU().addKey("PURIPSI-RESIDUALCONVERGENCE", header.residual_convergence, "");
  pFits->pHDU().addKey("PURIPSI-EPSILON", header.epsilon, "");

  // Writing image to fits file
  pFits->pHDU().write(fpixel, naxes[0] * naxes[1], image);
}

void write2d(const Image<t_real> &eigen_image, const std::string &fits_name,
             const std::string &pix_units, const bool &overwrite) {
  /*
    Writes an image to a fits file.

    image:: image data, a 2d Image.
    fits_name:: string containing the file name of the fits file.
    pix_units:: units of flux
    ra:: centre pixel coordinate in ra
    dec:: centre pixel coordinate in dec

  */
  PURIPSI_LOW_LOG("Writing fits file {}", fits_name);
  if(overwrite == true) {
    remove(fits_name.c_str());
  };
  long naxes[2] = {static_cast<long>(eigen_image.rows()), static_cast<long>(eigen_image.cols())};
  std::unique_ptr<CCfits::FITS> pFits(new CCfits::FITS(fits_name, FLOAT_IMG, 2, naxes));
  long fpixel(1);
  std::vector<long> extAx = {eigen_image.rows(), eigen_image.cols()};

  std::valarray<double> image(naxes[0] * naxes[1]);
  std::copy(eigen_image.data(), eigen_image.data() + eigen_image.size(), &image[0]);
  pFits->pHDU().addKey("AUTHOR", "PuriPsi", "");
  pFits->pHDU().addKey("BUNIT", pix_units, "");
  pFits->pHDU().write(fpixel, naxes[0] * naxes[1], image);
}

Image<t_complex> read2d(const std::string &fits_name) {
  /*
    Reads in an image from a fits file and returns the image.

    fits_name:: name of fits file
  */

  try{
    std::unique_ptr<CCfits::FITS> pInfile(new CCfits::FITS(fits_name, CCfits::Read, true));
    std::valarray<t_real> contents;
    CCfits::PHDU &image = pInfile->pHDU();
    image.read(contents);
    t_int ax1(image.axis(0));
    t_int ax2(image.axis(1));
    Image<t_complex> eigen_image(ax1, ax2);
    std::copy(&contents[0], &contents[0] + eigen_image.size(), eigen_image.data());  
    return eigen_image;
  }catch(CCfits::FITS::CantOpen){
    PURIPSI_CRITICAL("Failed to open fits file named: {}",fits_name);
    exit(1);
  }
    
}

Image<t_complex> read2dpartial(const std::string &fits_name, t_int ax2, t_int offset) {
  /*
    Reads in a portion of an image from a fits file and returns it.

    fits_name:: name of fits file
  */

  try{
    std::unique_ptr<CCfits::FITS> pInfile(new CCfits::FITS(fits_name, CCfits::Read, true));
    std::valarray<t_real> contents;
    CCfits::PHDU &image = pInfile->pHDU();
    image.read(contents);
    t_int ax1(image.axis(0));
    // t_int ax2(image.axis(1));
    Image<t_complex> eigen_image(ax1, ax2);
    std::copy(&contents[0] + offset, &contents[0] + offset + eigen_image.size(), eigen_image.data());
    return eigen_image;
  }catch(CCfits::FITS::CantOpen){
    PURIPSI_CRITICAL("Failed to open fits file named: {}",fits_name);
    exit(1);
  }
    
}

void write3d_header(const std::vector<Image<t_real>> &eigen_images,
                    const pfitsio::header_params &header, const bool &overwrite) {
  /*
     Writes an image to a fits file.
     image:: image data, a 2d Image.
     header:: structure containing header information
     overwrite:: if true, overwrites old fits file with same name
*/
  if(overwrite == true) {
    remove(header.fits_name.c_str());
  };
  long naxes[4]
      = {static_cast<long>(eigen_images[0].rows()), static_cast<long>(eigen_images[0].cols()),
         static_cast<long>(eigen_images.size()), 1};
  std::unique_ptr<CCfits::FITS> pFits(new CCfits::FITS(header.fits_name, FLOAT_IMG, 4, naxes));
  long fpixel(1);
  std::vector<long> extAx = {eigen_images[0].rows(), eigen_images[0].cols()};

  std::valarray<double> image(naxes[0] * naxes[1] * eigen_images.size());
  for(int i = 0; i < eigen_images.size(); i++)
    std::copy(eigen_images[i].data(), eigen_images[i].data() + eigen_images[i].size(),
              &image[naxes[0] * naxes[1] * i]);
  // Writing to fits header
  pFits->pHDU().addKey("AUTHOR", "PuriPsi", "");
  pFits->pHDU().addKey("BUNIT", header.pix_units, ""); // units
  pFits->pHDU().addKey("NAXIS", 4, "");                // number of axes
  pFits->pHDU().addKey("NAXIS1", static_cast<t_int>(eigen_images[0].cols()), "");
  pFits->pHDU().addKey("NAXIS2", static_cast<t_int>(eigen_images[0].rows()), "");
  pFits->pHDU().addKey("NAXIS3", eigen_images.size(), "");
  pFits->pHDU().addKey("NAXIS4", 1, "");
  pFits->pHDU().addKey("CRPIX1", static_cast<t_int>(std::floor(eigen_images[0].cols() / 2)) + 1,
                       "");
  pFits->pHDU().addKey("CRPIX2", static_cast<t_int>(std::floor(eigen_images[0].rows() / 2)) + 1,
                       "");
  pFits->pHDU().addKey("CRPIX3", 1, "");
  pFits->pHDU().addKey("CRPIX4", header.polarisation, "");
  pFits->pHDU().addKey("CRVAL1", header.ra * 180. / puripsi::constant::pi, "");
  pFits->pHDU().addKey("CRVAL2", header.dec * 180. / puripsi::constant::pi, "");
  pFits->pHDU().addKey("CRVAL3", header.mean_frequency * std::pow(10, 6) * 1., "");
  pFits->pHDU().addKey("CRVAL4", 1, "");
  pFits->pHDU().addKey("CTYPE1", "RA---SIN", "");
  pFits->pHDU().addKey("CTYPE2", "DEC---SIN", "");
  pFits->pHDU().addKey("CTYPE3", "FREQ-OBS", "");
  pFits->pHDU().addKey("CTYPE4", "STOKES", "");
  pFits->pHDU().addKey("CDELT1", -header.cell_x / 3600., "");
  pFits->pHDU().addKey("CDELT2", header.cell_y / 3600., "");
  pFits->pHDU().addKey("CDELT3", header.channel_width * std::pow(10, 6) * 1., "");
  pFits->pHDU().addKey("CDELT4", 1, "");
  pFits->pHDU().addKey("BTYPE", "intensity", "");
  pFits->pHDU().addKey("EQUINOX", 2000, "");
  pFits->pHDU().addKey("PURIPSI-NITERS", header.niters, "");
  if(header.hasconverged) {
    pFits->pHDU().addKey("PURIPSI-CONVERGED", "T", "");
  } else {
    pFits->pHDU().addKey("PURIPSI-CONVERGED", "F", "");
  }
  pFits->pHDU().addKey("PURIPSI-RELATIVEVARIATION", header.relative_variation, "");
  pFits->pHDU().addKey("PURIPSI-RESIDUALCONVERGENCE", header.residual_convergence, "");
  pFits->pHDU().addKey("PURIPSI-EPSILON", header.epsilon, "");

  // Writing cube to fits file
  pFits->pHDU().write(fpixel, naxes[0] * naxes[1] * eigen_images.size(), image);
}

void write3d(const std::vector<Image<t_real>> &eigen_images, const std::string &fits_name,
             const std::string &pix_units, const bool &overwrite) {
  /*
     Writes a vector of images to a fits file.
     image:: image data, a 3d Image.
     fits_name:: string containing the file name of the fits file.
     pix_units:: units of flux
     ra:: centre pixel coordinate in ra
     dec:: centre pixel coordinate in dec
*/
  if(overwrite == true) {
    remove(fits_name.c_str());
  };
  long naxes[3]
      = {static_cast<long>(eigen_images[0].rows()), static_cast<long>(eigen_images[0].cols()),
         static_cast<long>(eigen_images.size())};
  std::unique_ptr<CCfits::FITS> pFits(new CCfits::FITS(fits_name, FLOAT_IMG, 3, naxes));
  long fpixel(1);
  std::vector<long> extAx = {eigen_images[0].rows(), eigen_images[0].cols()};

  std::valarray<double> image(naxes[0] * naxes[1] * eigen_images.size());
  for(int i = 0; i < eigen_images.size(); i++)
    std::copy(eigen_images[i].data(), eigen_images[i].data() + eigen_images[i].size(),
              &image[naxes[0] * naxes[1] * i]);
  pFits->pHDU().addKey("AUTHOR", "PuriPsi", "");
  pFits->pHDU().addKey("BUNIT", pix_units, "");
  pFits->pHDU().addKey("NAXIS", 3, "");
  pFits->pHDU().addKey("NAXIS1", naxes[0], "");
  pFits->pHDU().addKey("NAXIS2", naxes[1], "");
  pFits->pHDU().addKey("NAXIS3", eigen_images.size(), "");
  pFits->pHDU().write(fpixel, naxes[0] * naxes[1] * eigen_images.size(), image);
}
std::vector<Image<t_complex>> read3d(const std::string &fits_name) {
  /*
     Reads in a cube from a fits file and returns the vector of images.
     fits_name:: name of fits file
     */

  std::unique_ptr<CCfits::FITS> pInfile(new CCfits::FITS(fits_name, CCfits::Read, true));
  std::valarray<t_real> contents;
  CCfits::PHDU &image = pInfile->pHDU();
  image.read(contents);
  t_int ax1(image.axis(0));
  t_int ax2(image.axis(1));
  t_int channels_total;
  image.readKey("NAXIS3", channels_total);
  std::vector<Image<t_complex>> eigen_images;
  for(int i = 0; i < channels_total; i++) {
    Image<t_complex> eigen_image(ax1, ax2);
    std::copy(&contents[ax1 * ax2 * i], &contents[ax1 * ax2 * i] + eigen_image.size(),
              eigen_image.data());
    eigen_images.push_back(eigen_image);
  }
  return eigen_images;
}
} // namespace pfitsio
} // namespace puripsi
