#ifndef PURIPSI_ASTRO_IO_H
#define PURIPSI_ASTRO_IO_H

#include "puripsi/config.h"
#include "puripsi/types.h"
#include "puripsi/utilities.h"
#include <psi/io.h>


namespace puripsi {
namespace astroio{
class AstroIO : public psi::io::IO<t_real> {

public:

	AstroIO() : psi::io::IO<t_real>() {};


public:
	psi::io::IOStatus save_uv_data(const std::vector<std::vector<utilities::vis_params>> uv_data, t_real dl, t_real pixel_size, std::string checkpoint_filename);
	psi::io::IOStatus load_uv_data(std::vector<std::vector<utilities::vis_params>> &uv_data, std::string checkpoint_filename);
	psi::io::IOStatus load_uv_data_header(int &frequencies, Vector<t_int> &blocks, t_real &dl, t_real &pixel_size, std::string checkpoint_filename);
};

}
}
#endif
