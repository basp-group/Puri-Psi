#include "puripsi/astroio.h"
#include "puripsi/logging.h"

namespace puripsi {

namespace astroio {

psi::io::IOStatus AstroIO::save_uv_data(const std::vector<std::vector<utilities::vis_params>> uv_data, t_real dl, t_real pixel_size, std::string checkpoint_filename){

	psi::io::IOStatus error = psi::io::IOStatus::Success;
	FILE* pFile;
	pFile = fopen(checkpoint_filename.c_str(), "wb");
	int size;
	if(pFile != NULL){
		size = uv_data.size();
		if((error == psi::io::IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
			error = psi::io::IOStatus::FileWriteError;
		}
		for(int f=0; f<uv_data.size(); f++){
			size = uv_data[f].size();
			if((error == psi::io::IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
				error = psi::io::IOStatus::FileWriteError;
			}
		}
		if((error == psi::io::IOStatus::Success) and (fwrite(&dl, sizeof(dl), 1, pFile) != 1)){
			error = psi::io::IOStatus::FileWriteError;
		}
		if((error == psi::io::IOStatus::Success) and (fwrite(&pixel_size, sizeof(pixel_size), 1, pFile) != 1)){
			error = psi::io::IOStatus::FileWriteError;
		}
		for(int f=0; f<uv_data.size(); f++){
			for(int t=0; t<uv_data[f].size(); t++){
				size = uv_data[f][t].u.size();
				if((error == psi::io::IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
				if((error == psi::io::IOStatus::Success) and (fwrite(uv_data[f][t].u.data(), sizeof(uv_data[f][t].u[0]), size, pFile) != size)){
					error = psi::io::IOStatus::FileWriteError;
				}
				size = uv_data[f][t].v.size();
				if((error == psi::io::IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
				if((error == psi::io::IOStatus::Success) and (fwrite(uv_data[f][t].v.data(), sizeof(uv_data[f][t].v[0]), size, pFile) != size)){
					error = psi::io::IOStatus::FileWriteError;
				}
				size = uv_data[f][t].w.size();
				if((error == psi::io::IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
				if((error == psi::io::IOStatus::Success) and (fwrite(uv_data[f][t].w.data(), sizeof(uv_data[f][t].w[0]), size, pFile) != size)){
					error = psi::io::IOStatus::FileWriteError;
				}
				size = uv_data[f][t].vis.size();
				if((error == psi::io::IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
				if((error == psi::io::IOStatus::Success) and (fwrite(uv_data[f][t].vis.data(), sizeof(uv_data[f][t].vis[0]), size, pFile) != size)){
					error = psi::io::IOStatus::FileWriteError;
				}
				size = uv_data[f][t].weights.size();
				if((error == psi::io::IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
				if((error == psi::io::IOStatus::Success) and (fwrite(uv_data[f][t].weights.data(), sizeof(uv_data[f][t].weights[0]), size, pFile) != size)){
					error = psi::io::IOStatus::FileWriteError;
				}
				size = uv_data[f][t].time.size();
				if((error == psi::io::IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
				if((error == psi::io::IOStatus::Success) and (fwrite(uv_data[f][t].time.data(), sizeof(uv_data[f][t].time[0]), size, pFile) != size)){
					error = psi::io::IOStatus::FileWriteError;
				}
				size = uv_data[f][t].baseline.size();
				if((error == psi::io::IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
				if((error == psi::io::IOStatus::Success) and (fwrite(uv_data[f][t].baseline.data(), sizeof(uv_data[f][t].baseline[0]), size, pFile) != size)){
					error = psi::io::IOStatus::FileWriteError;
				}
				size = uv_data[f][t].frequencies.size();
				if((error == psi::io::IOStatus::Success) and (fwrite(&size, sizeof(size), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
				if((error == psi::io::IOStatus::Success) and (fwrite(uv_data[f][t].frequencies.data(), sizeof(uv_data[f][t].frequencies[0]), size, pFile) != size)){
					error = psi::io::IOStatus::FileWriteError;
				}
				int temp_units = static_cast<int>(uv_data[f][t].units);
				if((error == psi::io::IOStatus::Success) and (fwrite(&temp_units, sizeof(temp_units), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
				if((error == psi::io::IOStatus::Success) and (fwrite(&uv_data[f][t].ra, sizeof(uv_data[f][t].ra), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
				if((error == psi::io::IOStatus::Success) and (fwrite(&uv_data[f][t].dec, sizeof(uv_data[f][t].dec), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
				if((error == psi::io::IOStatus::Success) and (fwrite(&uv_data[f][t].average_frequency, sizeof(uv_data[f][t].average_frequency), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
				if((error == psi::io::IOStatus::Success) and (fwrite(&uv_data[f][t].phase_centre_x, sizeof(uv_data[f][t].phase_centre_x), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
				if((error == psi::io::IOStatus::Success) and (fwrite(&uv_data[f][t].phase_centre_y, sizeof(uv_data[f][t].phase_centre_y), 1, pFile) != 1)){
					error = psi::io::IOStatus::FileWriteError;
				}
			}
		}
		fclose(pFile);
	}else{
		error = psi::io::IOStatus::OpenFailure;
	}
	return error;
}

psi::io::IOStatus  AstroIO::load_uv_data(std::vector<std::vector<utilities::vis_params>> &uv_data, std::string checkpoint_filename){

	psi::io::IOStatus error = psi::io::IOStatus::Success;
	FILE* pFile;
	pFile = fopen(checkpoint_filename.c_str(), "r");
	if(pFile != NULL){
		int frequencies;
		//! Read the header information  (number of frequencies and size of blocks)
		//! Currently we don't use that data here, but we read it to sensure we are in the correct
		//! place in the file.
		if(error == psi::io::IOStatus::Success and fread(&frequencies, sizeof(frequencies), 1, pFile) != 1){
			error = psi::io::IOStatus::FileReadError;
		}
		std::vector<t_int> block_sizes(frequencies);
		int temp;
		for(int f=0; f<frequencies; f++){
			if(error == psi::io::IOStatus::Success and fread(&temp, sizeof(temp), 1, pFile) != 1){
				error = psi::io::IOStatus::FileReadError;
			}
			block_sizes[f] = temp;
		}
		t_real temp_data;
		//! Read but discard the dl variable data (this will have been loaded with the header read function
		if(error == psi::io::IOStatus::Success and fread(&temp_data, sizeof(temp_data), 1, pFile) != 1){
			error = psi::io::IOStatus::FileReadError;
		}
		//! Read but discard the pixel_size variable data (this will have been loaded with the header read function
		if(error == psi::io::IOStatus::Success and fread(&temp_data, sizeof(temp_data), 1, pFile) != 1){
			error = psi::io::IOStatus::FileReadError;
		}
		if(frequencies > 0 and frequencies == uv_data.size()){
			for(int f=0; f<uv_data.size(); f++){
				if(uv_data[f].size() == block_sizes[f]){
					for(int t=0; t<uv_data[f].size(); t++){
						int size;
						//! Read the size of the u data vector from the file
						if(error == psi::io::IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}else{
							if(size != uv_data[f][t].u.size()){
								uv_data[f][t].u = Vector<t_real>(size);
							}
						}
						//! Read the u data vector from the file
						if(error == psi::io::IOStatus::Success and fread(uv_data[f][t].u.data(), sizeof(uv_data[f][t].u.data()[0]), size, pFile) != size){
							error = psi::io::IOStatus::FileReadError;
						}
						//! Read the size of the v data vector from the file
						if(error == psi::io::IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}else{
							if(size != uv_data[f][t].v.size()){
								uv_data[f][t].v = Vector<t_real>(size);
							}
						}
						//! Read the v data vector from the file
						if(error == psi::io::IOStatus::Success and fread(uv_data[f][t].v.data(), sizeof(uv_data[f][t].v.data()[0]), size, pFile) != size){
							error = psi::io::IOStatus::FileReadError;
						}
						//! Read the size of the w data vector from the file
						if(error == psi::io::IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}else{
							if(size != uv_data[f][t].w.size()){
								uv_data[f][t].w = Vector<t_real>(size);
							}
						}
						//! Read the w data vector from the file
						if(error == psi::io::IOStatus::Success and fread(uv_data[f][t].w.data(), sizeof(uv_data[f][t].w.data()[0]), size, pFile) != size){
							error = psi::io::IOStatus::FileReadError;
						}
						//! Read the size of the vis data vector from the file
						if(error == psi::io::IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}else{
							if(size != uv_data[f][t].vis.size()){
								uv_data[f][t].vis = Vector<t_complex>(size);
							}
						}
						//! Read the vis data vector from the file
						if(error == psi::io::IOStatus::Success and fread(uv_data[f][t].vis.data(), sizeof(uv_data[f][t].vis.data()[0]), size, pFile) != size){
							error = psi::io::IOStatus::FileReadError;
						}
						//! Read the size of the weights data vector from the file
						if(error == psi::io::IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}else{
							if(size != uv_data[f][t].weights.size()){
								uv_data[f][t].weights = Vector<t_complex>(size);
							}
						}
						//! Read the weights data vector from the file
						if(error == psi::io::IOStatus::Success and fread(uv_data[f][t].weights.data(), sizeof(uv_data[f][t].weights.data()[0]), size, pFile) != size){
							error = psi::io::IOStatus::FileReadError;
						}
						//! Read the size of the time data vector from the file
						if(error == psi::io::IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}else{
							if(size != uv_data[f][t].time.size()){
								uv_data[f][t].time = Vector<t_real>(size);
							}
						}
						//! Read the time data vector from the file
						if(error == psi::io::IOStatus::Success and fread(uv_data[f][t].time.data(), sizeof(uv_data[f][t].time.data()[0]), size, pFile) != size){
							error = psi::io::IOStatus::FileReadError;
						}
						//! Read the size of the baseline data vector from the file
						if(error == psi::io::IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}else{
							if(size != uv_data[f][t].baseline.size()){
								uv_data[f][t].baseline = Vector<t_uint>(size);
							}
						}
						//! Read the baseline data vector from the file
						if(error == psi::io::IOStatus::Success and fread(uv_data[f][t].baseline.data(), sizeof(uv_data[f][t].baseline.data()[0]), size, pFile) != size){
							error = psi::io::IOStatus::FileReadError;
						}
						//! Read the size of the frequencies data vector from the file
						if(error == psi::io::IOStatus::Success and fread(&size, sizeof(size), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}else{
							if(size != uv_data[f][t].frequencies.size()){
								uv_data[f][t].frequencies = Vector<t_real>(size);
							}
						}
						//! Read the frequencies data vector from the file
						if(error == psi::io::IOStatus::Success and fread(uv_data[f][t].frequencies.data(), sizeof(uv_data[f][t].frequencies.data()[0]), size, pFile) != size){
							error = psi::io::IOStatus::FileReadError;
						}
						int units;
						//! Read the units field from the file
						if(error == psi::io::IOStatus::Success and fread(&units, sizeof(units), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}
						switch(static_cast<utilities::vis_units>(units)){
						case utilities::vis_units::lambda:
							break;
						case utilities::vis_units::radians:
							break;
						case utilities::vis_units::pixels:
							break;
						default:
							PSI_ERROR("Problem with distribute_uv_data, wrong units distributed");
						}
						uv_data[f][t].units = static_cast<utilities::vis_units>(units);
						t_real real_data;
						//! Read the ra field from the file
						if(error == psi::io::IOStatus::Success and fread(&real_data, sizeof(real_data), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}
						uv_data[f][t].ra = real_data;
						//! Read the dec field from the file
						if(error == psi::io::IOStatus::Success and fread(&real_data, sizeof(real_data), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}
						uv_data[f][t].dec = real_data;
						//! Read the average_frequency field from the file
						if(error == psi::io::IOStatus::Success and fread(&real_data, sizeof(real_data), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}
						uv_data[f][t].average_frequency = real_data;
						t_int int_data;
						//! Read the phase_centre_x field from the file
						if(error == psi::io::IOStatus::Success and fread(&real_data, sizeof(real_data), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}
						uv_data[f][t].phase_centre_x = real_data;
						//! Read the phase_centre_y field from the file
						if(error == psi::io::IOStatus::Success and fread(&real_data, sizeof(real_data), 1, pFile) != 1){
							error = psi::io::IOStatus::FileReadError;
						}
						uv_data[f][t].phase_centre_y = real_data;
					}
				}else{
					error = psi::io::IOStatus::WrongNumberOfTimeBlocks;
				}
			}
		}else{
			error = psi::io::IOStatus::WrongNumberOfFrequencies;
		}

		fclose(pFile);
	}else{
		error = psi::io::IOStatus::OpenFailure;
	}

	return error;

}

psi::io::IOStatus AstroIO::load_uv_data_header(int &frequencies, Vector<t_int> &blocks, t_real &dl, t_real &pixel_size, std::string checkpoint_filename){

	psi::io::IOStatus error = psi::io::IOStatus::Success;
	FILE* pFile;
	pFile = fopen(checkpoint_filename.c_str(), "r");
	if(pFile != NULL){
		if(error == psi::io::IOStatus::Success and fread(&frequencies, sizeof(frequencies), 1, pFile) != 1){
			error = psi::io::IOStatus::FileReadError;
		}
		if(error == psi::io::IOStatus::Success and frequencies > 0){
			blocks.conservativeResize(frequencies);
			int temp_block;
			for(int f=0; f<frequencies; f++){
				if(error == psi::io::IOStatus::Success and fread(&temp_block, sizeof(blocks[0]), 1, pFile) != 1){
					error = psi::io::IOStatus::FileReadError;
				}
				blocks[f] = temp_block;
			}
		}else{
			error = psi::io::IOStatus::WrongNumberOfFrequencies;
		}
		if(error == psi::io::IOStatus::Success and fread(&dl, sizeof(dl), 1, pFile) != 1){
			error = psi::io::IOStatus::FileReadError;
		}
		if(error == psi::io::IOStatus::Success and fread(&pixel_size, sizeof(pixel_size), 1, pFile) != 1){
			error = psi::io::IOStatus::FileReadError;
		}
		fclose(pFile);
	}else{
		error = psi::io::IOStatus::OpenFailure;
	}

	return error;

}

}

}
