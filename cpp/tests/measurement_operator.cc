#include <catch2/catch.hpp>
#include <complex>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "H5Cpp.h"

#include <psi/types.h>
#include <psi/utilities.h>
#include "puripsi/types.h"
#include "puripsi/utilities.h"
#include "puripsi/operators.h"
#include "puripsi/logging.h"

using namespace H5;
using namespace puripsi;
using namespace operators;

typedef struct complex_type{
	double real;
	double imag;
} complex_type;

TEST_CASE("Test consistency of MeasurementOperator with its Matlab counterpart, [consistency]"){

	std::string filename = "../../../cpp/tests/test_measurement_operator_N=256_p=1.h5";
	H5File file(filename, H5F_ACC_RDONLY);


	// Load uv coverage
	DataSet dataset = file.openDataSet("uvw");
	DataSpace dataspace = dataset.getSpace();
	int rank = dataspace.getSimpleExtentNdims();
	hsize_t dims_uvw[rank];
	int ndims = dataspace.getSimpleExtentDims(dims_uvw, NULL);
	psi::Image<psi::t_real> uvw = psi::Image<psi::t_real>::Zero(dims_uvw[1], dims_uvw[0]); //! needed because reading is column major with eigen, storage is row major in HDF5
	dataset.read(uvw.data(), PredType::NATIVE_DOUBLE);
	uvw.transposeInPlace(); //! needed because reading is column major with eigen, storage is row major in HDF5
	INFO ( "Size uvw: " << uvw(0,0) << " x " << uvw.cols());
	INFO ( "u(0) = " << uvw(0,0) << " v(0) = " << uvw(0,1) << " | w(0) = " << uvw(0,2));
	INFO ( "u(1) = " << uvw(1,0) << " v(1) = " << uvw(1,1) << " | w(1) = " << uvw(1,2));
	dataspace.close();
	dataset.close();

	// verify all entries of w are equal to 1 for this example
	// error in the structure of the file otherwise
	CHECK(uvw.col(2).matrix().isApprox(psi::Vector<psi::t_real>::Constant(uvw.rows(), 1), 1e-12));

	// Load test image
	dataset = file.openDataSet("image");
	dataspace = dataset.getSpace();
	rank = dataspace.getSimpleExtentNdims();
	hsize_t dims_image[rank];
	ndims = dataspace.getSimpleExtentDims(dims_image, NULL);
	psi::Image<psi::t_real> x = psi::Image<psi::t_real>::Zero(dims_image[1], dims_image[0]);  //! need to invert dimensions since reading is column major with eigen, storage is row major in HDF5

	dataset.read(x.data(), PredType::NATIVE_DOUBLE);
	x.transposeInPlace(); //! transpose needed to have the same format as the reference Matlab arrays
	dataspace.close();
	dataset.close();

	// vectorizing the image x for the following comparisons
	psi::Vector<psi::t_real> x_ = Vector<psi::t_real>::Map(x.data(), x.size());
	psi::Vector<psi::t_complex> x0 = Vector<psi::t_complex>::Zero(x_.size());
	x0.real() = x_;
	psi::Image<psi::t_complex> x_c = psi::Image<psi::t_complex>::Zero(x.rows(), x.cols());
	x_c.real() = x;

	INFO("Size x: " << x.rows() << " x " << x.cols());
	INFO("x(1,0) = " << x(1,0) << " | x(0,1) = " << x(0,1));


	// Load test visibilities
	CompType complex_data_type(sizeof(complex_type));
	complex_data_type.insertMember( "real", 0, PredType::NATIVE_DOUBLE);
	complex_data_type.insertMember( "imag", sizeof(double), PredType::NATIVE_DOUBLE);
	dataset = file.openDataSet("visibilities");
	dataspace = dataset.getSpace();
	rank = dataspace.getSimpleExtentNdims();
	hsize_t dims_vis[rank];
	ndims = dataspace.getSimpleExtentDims(dims_vis, NULL);
	psi::Vector<psi::t_complex> y = psi::Vector<psi::t_complex>::Zero(dims_vis[0]);

	dataset.read(y.data(), complex_data_type);
	dataspace.close();
	dataset.close();
	INFO("Size y: " << dims_vis[0]);
	CHECK(dims_vis[0] > 1);

	// Load parameters for the measurement operator
	dataset = file.openDataSet("J");
	dataspace = dataset.getSpace();
	int J;
	dataset.read(&J, PredType::NATIVE_INT);
	dataspace.close();
	dataset.close();
	INFO("J = : " << J);

	file.close();

	// Parameters measurement operator
	psi::t_int const Ju = J;
	psi::t_int const Jv = J;
	int imsizex = x.cols();
	int imsizey = x.rows();
	t_real const over_sample = 2.;
	const std::string kernel = "kb";
	const t_uint ftsizev = std::floor(imsizey * over_sample);
	const t_uint ftsizeu = std::floor(imsizex * over_sample);
	psi::t_real nshiftx = static_cast<psi::t_real>(imsizex)/2.;
	psi::t_real nshifty = static_cast<psi::t_real>(imsizey)/2.;
	t_real pixel_size = 1.0; //! pixel size is not active when giving uv-coverage in radians

	utilities::vis_params uv_vis;
	uv_vis.u = uvw.col(0);
	uv_vis.v = uvw.col(1);
	uv_vis.w = uvw.col(2);
	uv_vis.weights = psi::Vector<psi::t_real>::Constant(uvw.col(0).size(), 1);
	uv_vis.vis = psi::Vector<psi::t_real>::Constant(uvw.col(0).size(), 1);
	uv_vis.units = puripsi::utilities::vis_units::radians;

	// Full measurement operator
	MeasurementOperator<Vector<t_complex>, t_complex> Phi =
			MeasurementOperator<Vector<t_complex>, t_complex>(uv_vis, imsizey, imsizex, pixel_size, pixel_size, over_sample, 1000,
					0.0001, kernels::kernel_from_string.at(kernel), nshifty, nshiftx, Ju, Jv);

	// Full measurement operator using a shared ptr
	std::shared_ptr<MeasurementOperator<Vector<t_complex>, t_complex>> shared_Phi =
			std::make_shared<MeasurementOperator<Vector<t_complex>, t_complex>>(uv_vis, imsizey, imsizex, pixel_size, pixel_size, over_sample, 1000,
					0.0001, kernels::kernel_from_string.at(kernel), nshifty, nshiftx, Ju, Jv);

	// Vector of full measurement operators using shared ptrs
	std::vector<std::shared_ptr<MeasurementOperator<Vector<t_complex>, t_complex>>> vector_Phi;
	vector_Phi.reserve(2);
	vector_Phi.emplace_back(std::make_shared<MeasurementOperator<Vector<t_complex>, t_complex>>(uv_vis, imsizey, imsizex, pixel_size, pixel_size, over_sample, 1000,
			0.0001, kernels::kernel_from_string.at(kernel), nshifty, nshiftx, Ju, Jv));
	vector_Phi.emplace_back(std::make_shared<MeasurementOperator<Vector<t_complex>, t_complex>>(uv_vis, imsizey, imsizex, pixel_size, pixel_size, over_sample, 1000,
			0.0001, kernels::kernel_from_string.at(kernel), nshifty, nshiftx, Ju, Jv));

	Vector<t_real> preconditioner = Vector<t_real>::Ones(imsizex*imsizey);

	// Preconditioning measurement operator
	MeasurementOperator<Vector<t_complex>, t_complex> pre_Phi =
			MeasurementOperator<Vector<t_complex>, t_complex>(uv_vis, preconditioner, imsizey, imsizex, pixel_size, pixel_size, over_sample, 1000,
					0.0001, kernels::kernel_from_string.at(kernel), nshifty, nshiftx, Ju, Jv);

	Vector<t_real> random_preconditioner = Vector<t_real>::Random(imsizex*imsizey);

	// Preconditioning measurement operator
	MeasurementOperator<Vector<t_complex>, t_complex> random_pre_Phi =
			MeasurementOperator<Vector<t_complex>, t_complex>(uv_vis, random_preconditioner, imsizey, imsizex, pixel_size, pixel_size, over_sample, 1000,
					0.0001, kernels::kernel_from_string.at(kernel), nshifty, nshiftx, Ju, Jv);

	// Initial tests and data generation
	Vector<t_complex> Phix2 = Phi.degrid(Matrix<t_complex>::Map(x0.data(), imsizey, imsizex));
	Vector<t_complex> Phix = Phi * Vector<t_complex>::Map(x0.data(), imsizey*imsizex);
	Vector<t_complex> Phity2(imsizex*imsizey);
	Image<t_complex> image = Image<t_complex>::Map(Phity2.data(), imsizey, imsizex);
	Vector<t_complex> Phity = Phi.adjoint() * y;
	image = Phi.grid(y);

	// Testing above using vector of measurement operators
	psi::t_uint nvis_temp = Phi.M()[2];
	psi::Vector<psi::t_complex> y1_temp = psi::Vector<psi::t_complex>::Random(nvis_temp);
	psi::Vector<psi::t_complex> temp_vec = psi::Vector<psi::t_complex>::Map(x0.data(), imsizey*imsizex);;
	Phix  = *(vector_Phi[0]) * temp_vec;
	psi::Vector<psi::t_complex> temp_Phity1_temp = (*vector_Phi[0]).adjoint()*y1_temp;




	SECTION("Test consistency grid correction (S)"){
		// Load scale coefficients (gridding correction)
		H5File file(filename, H5F_ACC_RDONLY);
		DataSet dataset = file.openDataSet("scale");
		DataSpace dataspace = dataset.getSpace();
		int rank = dataspace.getSimpleExtentNdims();
		hsize_t dims_scale[rank];
		int ndims = dataspace.getSimpleExtentDims(dims_scale, NULL);
		psi::Image<psi::t_real> S = psi::Image<psi::t_real>::Zero(dims_scale[1], dims_scale[0]);
		dataset.read(S.data(), PredType::NATIVE_DOUBLE);
		dataspace.close();
		dataset.close();
		file.close();
		S.transposeInPlace();
		INFO("Size S: " << S.rows() << " x " << S.cols());

		psi::Image<psi::t_real> Scc = Phi.S();
		INFO("S(0,0) = " << S(0,0) << " | S(1,0) = " << S(1,0) << " | S(0,1) = " << S(0,1));
		INFO("Scc(0,0) = " << Scc(0,0) << " | Scc(1,0) = " << Scc(1,0) << " | Scc(0,1) = " << Scc(0,1));
		t_real relative_error = (Scc - S).matrix().norm()/S.matrix().norm();

		CHECK(relative_error < 1e-12);
	}

	SECTION("Test consistency FFT after 0-padding and grid correction (Ax)"){

		// Load Ax
		H5File file(filename, H5F_ACC_RDONLY);
		DataSet dataset = file.openDataSet("Ax");
		DataSpace dataspace = dataset.getSpace();
		int rank = dataspace.getSimpleExtentNdims();
		hsize_t dims_Ax[rank];
		int ndims = dataspace.getSimpleExtentDims(dims_Ax, NULL);
		psi::Matrix<psi::t_complex> Ax = psi::Matrix<psi::t_complex>::Zero(dims_Ax[1], dims_Ax[0]);
		dataset.read(Ax.data(), complex_data_type);
		dataspace.close();
		dataset.close();
		Ax.transposeInPlace();
		INFO("Size AX: " << Ax.rows() << " x " << Ax.cols());
		INFO("Ax(0,0) = " << Ax(0,0) << " | Ax(1,0) = " << Ax(1,0) << " | Ax(0,1) = " << Ax(0,1));
		file.close();

		psi::Matrix<psi::t_complex> Ax_cc = Phi.FFT(x_c);
        Ax_cc.resize(ftsizev, ftsizeu);

		INFO("Size AX_cc: " << Ax_cc.rows() << " x " << Ax_cc.cols());
		INFO("Ax_cc(0,0) = " << Ax_cc(0,0) << " | Ax_cc(1,0) = " << Ax_cc(1,0) << " | Ax_cc(0,1) = " << Ax_cc(0,1));
		psi::t_real relative_error = (Ax_cc - Ax).norm()/Ax.norm();

		CHECK(relative_error < 1e-12);
	}


	SECTION("Test consistency visibilities (y)"){

		// Option 1
		psi::Vector<psi::t_complex> ycc = Phi * x0;
		CHECK(ycc.isApprox(y, 1e-12));

		// Option 2
		psi::Vector<psi::t_complex> ycc2 = Phi.degrid(x_c);
		CHECK(ycc2.isApprox(y, 1e-12));

		// Option 3
		psi::Matrix<psi::t_complex> out_hat = Phi.FFT(x_c);
		psi::Vector<psi::t_complex> ycc_sparse = Phi.G_function(out_hat);
		CHECK(ycc_sparse.isApprox(y, 1e-12));
	}

	SECTION("Test consistency visibilities (y)"){

		// Option 1
		psi::Vector<psi::t_complex> ycc = Phi * x0;
		CHECK(ycc.isApprox(y, 1e-12));

		// Option 2
		psi::Vector<psi::t_complex> ycc2 = Phi.degrid(x_c);
		CHECK(ycc2.isApprox(y, 1e-12));

		// Option 3
		psi::Matrix<psi::t_complex> out_hat = Phi.FFT(x_c);
	    int array_size = Phi.oversample_factor() * Phi.imsizey() * Phi.oversample_factor() * Phi.imsizex();

		Eigen::SparseMatrix<t_complex> x_hat_sparse = Eigen::SparseMatrix<t_complex>(array_size, 1);

		psi::Vector<psi::t_int> indices = Phi.get_fourier_indices();
		x_hat_sparse.reserve(indices.size());

		for (int k=0; k<indices.size(); k++){
			psi::t_complex data =  out_hat(indices(k),0);
			x_hat_sparse.insert(indices(k),0) = data;
		}
		x_hat_sparse.makeCompressed();
		psi::Vector<psi::t_complex> ycc_sparse = Phi.G_function(x_hat_sparse);
		CHECK(ycc_sparse.isApprox(y, 1e-12));
	}

	SECTION("Test consistency dirty image"){
		// Load dirty image
		H5File file(filename, H5F_ACC_RDONLY);
		DataSet dataset = file.openDataSet("dirty_image");
		DataSpace dataspace = dataset.getSpace();
		int rank = dataspace.getSimpleExtentNdims();
		hsize_t dims_dirty[rank];
		int ndims = dataspace.getSimpleExtentDims(dims_dirty, NULL);
		psi::Image<psi::t_complex> dirty_image = psi::Image<psi::t_complex>::Zero(dims_dirty[1], dims_dirty[0]);
		dataset.read(dirty_image.data(), complex_data_type);
		dirty_image.transposeInPlace();
		dataspace.close();
		dataset.close();
		file.close();
		INFO("Size dirty_image: " << dirty_image.rows() << " x " << dirty_image.cols());

		psi::Image<psi::t_complex> dirty_image_cc = Phi.grid(y);

		INFO("Size dirty_image_cc: " << dirty_image_cc.rows() << " x " << dirty_image_cc.cols());

		CHECK(dirty_image_cc.size() == dirty_image.size());
		CHECK(dirty_image_cc.rows() == dirty_image.rows());

		psi::t_real relative_error = (dirty_image_cc - dirty_image).matrix().norm()/dirty_image.matrix().norm();

		CHECK(relative_error < 1e-12);
	}

	SECTION("Test consistency dirty image (shared pointer)"){
		// Load dirty image
		H5File file(filename, H5F_ACC_RDONLY);
		DataSet dataset = file.openDataSet("dirty_image");
		DataSpace dataspace = dataset.getSpace();
		int rank = dataspace.getSimpleExtentNdims();
		hsize_t dims_dirty[rank];
		int ndims = dataspace.getSimpleExtentDims(dims_dirty, NULL);
		psi::Image<psi::t_complex> dirty_image = psi::Image<psi::t_complex>::Zero(dims_dirty[1], dims_dirty[0]);
		dataset.read(dirty_image.data(), complex_data_type);
		dirty_image.transposeInPlace();
		dataspace.close();
		dataset.close();
		file.close();
		INFO("Size dirty_image: " << dirty_image.rows() << " x " << dirty_image.cols());

		psi::Image<psi::t_complex> dirty_image_cc = shared_Phi->grid(y);

		INFO("Size dirty_image_cc: " << dirty_image_cc.rows() << " x " << dirty_image_cc.cols());

		CHECK(dirty_image_cc.size() == dirty_image.size());
		CHECK(dirty_image_cc.rows() == dirty_image.rows());

		psi::t_real relative_error = (dirty_image_cc - dirty_image).matrix().norm()/dirty_image.matrix().norm();

		CHECK(relative_error < 1e-12);
	}


	SECTION("Check gridding is the adjoint of degridding"){

		psi::t_uint nvis = Phi.M()[2];
		psi::Vector<psi::t_complex> y1 = psi::Vector<psi::t_complex>::Random(nvis);
		psi::Image<psi::t_complex> x1 = psi::Matrix<psi::t_complex>::Random(imsizey, imsizex);

		psi::Vector<psi::t_complex> Phix1 = Phi.degrid(x1);
		psi::Image<psi::t_complex> Phity1 = Phi.grid(y1);

		INFO("Size x1: " << x1.rows() << " x " << x1.cols());
		INFO("Size of Phity1: " << Phity1.rows() << " x " << Phity1.cols());

		psi::t_complex p1 = (Phix1.array().conjugate() * y1.array()).sum();
		psi::t_complex p2 = (x1.array().conjugate() * Phity1.array()).sum();
		psi::t_real relative_error = std::abs(p1 - p2)/std::abs(p2);

		CHECK(relative_error < 1e-12);
	}

	SECTION("Check gridding is the adjoint of degridding (operator)"){

		psi::t_uint nvis = Phi.M()[2];
		psi::Vector<psi::t_complex> y1 = psi::Vector<psi::t_complex>::Random(nvis);
		psi::Image<psi::t_complex> x1 = psi::Matrix<psi::t_complex>::Random(imsizey, imsizex);

		psi::Vector<psi::t_complex> tempx1 = Vector<t_complex>::Map(x1.data(), imsizey*imsizex);

		psi::Vector<psi::t_complex> Phix1 = Phi*tempx1;
		psi::Vector<psi::t_complex> temp_Phity1 = Phi.adjoint()*y1;
		psi::Image<psi::t_complex> Phity1 = psi::Image<psi::t_complex>::Map(temp_Phity1.data(), imsizey, imsizex);

		INFO("Size x1: " << x1.rows() << " x " << x1.cols());
		INFO("Size of Phity1: " << Phity1.rows() << " x " << Phity1.cols());

		psi::t_complex p1 = (Phix1.array().conjugate() * y1.array()).sum();
		psi::t_complex p2 = (x1.array().conjugate() * Phity1.array()).sum();
		psi::t_real relative_error = std::abs(p1 - p2)/std::abs(p2);

		CHECK(relative_error < 1e-12);
	}

	SECTION("Check gridding is the adjoint of degridding (shared pointer)"){

		psi::t_uint nvis = Phi.M()[2];
		psi::Vector<psi::t_complex> y1 = psi::Vector<psi::t_complex>::Random(nvis);
		psi::Image<psi::t_complex> x1 = psi::Matrix<psi::t_complex>::Random(imsizey, imsizex);

		psi::Vector<psi::t_complex> Phix1 = shared_Phi->degrid(x1);
		psi::Image<psi::t_complex> Phity1 = shared_Phi->grid(y1);

		INFO("Size x1: " << x1.rows() << " x " << x1.cols());
		INFO("Size of Phity1: " << Phity1.rows() << " x " << Phity1.cols());

		psi::t_complex p1 = (Phix1.array().conjugate() * y1.array()).sum();
		psi::t_complex p2 = (x1.array().conjugate() * Phity1.array()).sum();
		psi::t_real relative_error = std::abs(p1 - p2)/std::abs(p2);

		CHECK(relative_error < 1e-12);
	}

	SECTION("Check gridding is the adjoint of degridding (shared pointer)(operator)"){

		psi::t_uint nvis = Phi.M()[2];
		psi::Vector<psi::t_complex> y1 = psi::Vector<psi::t_complex>::Random(nvis);
		psi::Image<psi::t_complex> x1 = psi::Matrix<psi::t_complex>::Random(imsizey, imsizex);

		psi::Vector<psi::t_complex> tempx1 = Vector<t_complex>::Map(x1.data(), imsizey*imsizex);

		psi::Vector<psi::t_complex> Phix1 = (*shared_Phi)*tempx1;
		psi::Vector<psi::t_complex> temp_Phity1 = shared_Phi->adjoint()*y1;
		psi::Image<psi::t_complex> Phity1 = psi::Image<psi::t_complex>::Map(temp_Phity1.data(), imsizey, imsizex);

		INFO("Size x1: " << x1.rows() << " x " << x1.cols());
		INFO("Size of Phity1: " << Phity1.rows() << " x " << Phity1.cols());

		psi::t_complex p1 = (Phix1.array().conjugate() * y1.array()).sum();
		psi::t_complex p2 = (x1.array().conjugate() * Phity1.array()).sum();
		psi::t_real relative_error = std::abs(p1 - p2)/std::abs(p2);

		CHECK(relative_error < 1e-12);
	}

	SECTION("Check gridding is the adjoint of degridding (vector shared pointer)(operator)"){

		psi::t_uint nvis = Phi.M()[2];
		psi::Vector<psi::t_complex> y1 = psi::Vector<psi::t_complex>::Random(nvis);
		psi::Image<psi::t_complex> x1 = psi::Matrix<psi::t_complex>::Random(imsizey, imsizex);

		psi::Vector<psi::t_complex> tempx1 = Vector<t_complex>::Map(x1.data(), imsizey*imsizex);

		psi::Vector<psi::t_complex> temp_Phity1 = (*vector_Phi[0]).adjoint()*y1;
		psi::Vector<psi::t_complex> Phix1 = (*vector_Phi[0])*tempx1;
		psi::Image<psi::t_complex> Phity1 = psi::Image<psi::t_complex>::Map(temp_Phity1.data(), imsizey, imsizex);

		INFO("Size x1: " << x1.rows() << " x " << x1.cols());
		INFO("Size of Phity1: " << Phity1.rows() << " x " << Phity1.cols());

		psi::t_complex p1 = (Phix1.array().conjugate() * y1.array()).sum();
		psi::t_complex p2 = (x1.array().conjugate() * Phity1.array()).sum();
		psi::t_real relative_error = std::abs(p1 - p2)/std::abs(p2);

		CHECK(relative_error < 1e-12);

		Phix1 = (*vector_Phi[1])*tempx1;
		temp_Phity1 = (*vector_Phi[1]).adjoint()*y1;
		Phity1 = psi::Image<psi::t_complex>::Map(temp_Phity1.data(), imsizey, imsizex);

		INFO("Size x1: " << x1.rows() << " x " << x1.cols());
		INFO("Size of Phity1: " << Phity1.rows() << " x " << Phity1.cols());

		p1 = (Phix1.array().conjugate() * y1.array()).sum();
		p2 = (x1.array().conjugate() * Phity1.array()).sum();
		relative_error = std::abs(p1 - p2)/std::abs(p2);

		CHECK(relative_error < 1e-12);
	}

	SECTION("Test consistency dirty image"){

		// Load dirty image
		H5File file(filename, H5F_ACC_RDONLY);
		DataSet dataset = file.openDataSet("dirty_image");
		DataSpace dataspace = dataset.getSpace();
		int rank = dataspace.getSimpleExtentNdims();
		hsize_t dims_dirty[rank];
		int ndims = dataspace.getSimpleExtentDims(dims_dirty, NULL);
		psi::Image<psi::t_complex> dirty_image = psi::Image<psi::t_complex>::Zero(dims_dirty[1], dims_dirty[0]);
		dataset.read(dirty_image.data(), complex_data_type);
		dirty_image.transposeInPlace();
		dataspace.close();
		dataset.close();
		file.close();
		INFO("Size dirty_image: " << dirty_image.rows() << " x " << dirty_image.cols());

		psi::Image<psi::t_complex> dirty_image_cc = Phi.grid(y);

		INFO("Size dirty_image_cc: " << dirty_image_cc.rows() << " x " << dirty_image_cc.cols());

		CHECK(dirty_image_cc.size() == dirty_image.size());
		CHECK(dirty_image_cc.rows() == dirty_image.rows());

		psi::t_real relative_error = (dirty_image_cc - dirty_image).matrix().norm()/dirty_image.matrix().norm();

		CHECK(relative_error < 1e-12);

	}

	SECTION("Checking gridding and degridding"){

		t_complex p0 = (Phix.array().conjugate() * y.array()).sum();
		t_complex p1 = (Vector<t_complex>::Map(x0.data(), imsizey*imsizex).array().conjugate() * Phity.array()).sum();
		CHECK(std::abs(p0 - p1)/std::abs(p0) < 1e-13);

	}

	SECTION("Test FFT component"){
		psi::Vector<psi::t_complex> input = psi::Vector<psi::t_complex>::Random(imsizey*imsizex);
		psi::Vector<psi::t_complex> padded_input = psi::Vector<psi::t_real>::Zero(ftsizev*ftsizeu);
		psi::Vector<psi::t_real> zero_comp = psi::Vector<psi::t_real>::Zero(ftsizev*ftsizeu);
		padded_input.segment(0,imsizey*imsizex) =  input;
		padded_input.imag() = zero_comp;


		psi::Vector<psi::t_complex> FFT_out(ftsizev*ftsizeu);

		Phi.directFFT()(FFT_out, padded_input);

		psi::Vector<psi::t_complex> indirectFFT_out(imsizey*imsizex);
		Phi.indirectFFT()(indirectFFT_out, FFT_out);

		indirectFFT_out.imag() = zero_comp;
		psi::Vector<psi::t_complex> output = psi::Vector<psi::t_complex>::Zero(imsizey*imsizex);
		output.real() = indirectFFT_out.segment(0,imsizey*imsizex).real();
		output /= (ftsizev*ftsizeu);

		INFO("input sum: " << input.real().sum() << "  output sum: " << output.real().sum());
		CHECK(input.real().isApprox(output.real(), 1e-12));

	}

	SECTION("Testing preconditioning functionality"){

		psi::t_uint nvis = Phi.M()[2];
		psi::Vector<psi::t_complex> y1 = psi::Vector<psi::t_complex>::Random(nvis);
		psi::Image<psi::t_complex> x1 = psi::Matrix<psi::t_complex>::Random(imsizey, imsizex);

		psi::Vector<psi::t_complex> tempx1 = Vector<t_complex>::Map(x1.data(), imsizey*imsizex);

		psi::Vector<psi::t_complex> Phix1 = Phi*tempx1;
		psi::Vector<psi::t_complex> Phix2 = pre_Phi*tempx1;

		psi::t_complex p1 = Phix1.sum();
		psi::t_complex p2 = Phix2.sum();
		psi::t_real relative_error = std::abs(p1 - p2);

		CHECK(relative_error < 1e-15);

		psi::Vector<psi::t_complex> temp_Phity1 = Phi.adjoint()*y1;
		psi::Vector<psi::t_complex> temp_Phity2 = pre_Phi.adjoint()*y1;

		p1 = temp_Phity1.sum();
		p2 = temp_Phity2.sum();

		relative_error = std::abs(p1 - p2);

		CHECK(relative_error < 1e-15);

	}

	SECTION("Testing turning off preconditioning functionality"){

		psi::t_uint nvis = Phi.M()[2];
		psi::Vector<psi::t_complex> y1 = psi::Vector<psi::t_complex>::Random(nvis);
		psi::Image<psi::t_complex> x1 = psi::Matrix<psi::t_complex>::Random(imsizey, imsizex);

		psi::Vector<psi::t_complex> tempx1 = Vector<t_complex>::Map(x1.data(), imsizey*imsizex);

		psi::Vector<psi::t_complex> Phix1 = Phi*tempx1;
		random_pre_Phi.disable_preconditioning();
		psi::Vector<psi::t_complex> Phix2 = random_pre_Phi*tempx1;

		psi::t_complex p1 = Phix1.sum();
		psi::t_complex p2 = Phix2.sum();
		psi::t_real relative_error = std::abs(p1 - p2);

		CHECK(relative_error < 1e-15);

		psi::Vector<psi::t_complex> temp_Phity1 = Phi.adjoint()*y1;
		psi::Vector<psi::t_complex> temp_Phity2 = random_pre_Phi.adjoint()*y1;

		p1 = temp_Phity1.sum();
		p2 = temp_Phity2.sum();

		relative_error = std::abs(p1 - p2);

		CHECK(relative_error < 1e-15);
	}

	SECTION("Testing preconditioning functionality"){

		psi::t_uint nvis = Phi.M()[2];
		psi::Vector<psi::t_complex> y1 = psi::Vector<psi::t_complex>::Random(nvis);
		psi::Image<psi::t_complex> x1 = psi::Matrix<psi::t_complex>::Random(imsizey, imsizex);

		psi::Vector<psi::t_complex> tempx1 = Vector<t_complex>::Map(x1.data(), imsizey*imsizex);

		psi::Vector<psi::t_complex> Phix1 = Phi*tempx1;
		random_pre_Phi.enable_preconditioning();
		psi::Vector<psi::t_complex> Phix2 = random_pre_Phi*tempx1;

		psi::t_complex p1 = Phix1.sum();
		psi::t_complex p2 = Phix2.sum();

		CHECK(p1 != p2);

		psi::Vector<psi::t_complex> temp_Phity1 = Phi.adjoint()*y1;
		psi::Vector<psi::t_complex> temp_Phity2 = random_pre_Phi.adjoint()*y1;

		p1 = temp_Phity1.sum();
		p2 = temp_Phity2.sum();

		CHECK(p1 != p2);

	}

}
