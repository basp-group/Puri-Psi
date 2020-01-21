
# Puri-Psi

## Description

Puri-Psi is an open-source package that implements programs used to undertake radio interferometric imaging. The Puri-Psi programs can use standard radio astronomy visibility and fits files, is implemented in C++ and utilises MPI and OpenMP for parallelisation.
Puri-Psi uses [Psi](https://github.com/basp-group/Psi) as an implementation of the optimisation algorithms and wavelet operators required to reconstruct images for radio astronomy.

## Installation

### C++ pre-requisites and dependencies

- [CMake](http://www.cmake.org/): Program building software.
- [fftw3](www.fftw.org): Fastest Fourier Transform library.
- [tiff](http://www.libtiff.org/): Tag Image File Format library
- [OpenMP](http://openmp.org/wp/): Optional. Parallelises operations across multiple threads.
- [MPI](http://www.mpi-forum.org): Optional. Parallelises operations across multiple processes.
- [casacore](http://casacore.github.io/casacore/): Library for reading and writing Casa datasets
- [basp-group/Psi](https://github.com/basp-group/Psi): Optimisation library.
- [Eigen 3](http://eigen.tuxfamily.org/index.php?title=Main_Page): C++ linear algebra framework.
- [spdlog](https://github.com/gabime/spdlog): Logging library.
- [cfitsio](http://heasarc.gsfc.nasa.gov/fitsio/fitsio.html): library of functions for reading and writing data files in FITS (Flexible Image Transport System) format.
- [CCFits](http://heasarc.gsfc.nasa.gov/fitsio/ccfits/):Cc++ wrappers for cfitsio.
- [Catch2](https://github.com/catchorg/Catch2): A C++ unit-testing framework.
- [google/benchmark](https://github.com/google/benchmark):A C++ micro-benchmarking framework.

### Installing

Once the dependencies are present, the program can be built with:

```
cd /path/to/code
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```


## Contributors

Psi has been developed by:

- [Adrian Jackson](https://www.epcc.ed.ac.uk/about/staff/mr-adrian-jackson)
- Pierre-Antoine Thouvenin
- Ming Jiang
- Alex Onose
- [Yves Wiaux](http://basp.eps.hw.ac.uk/)

Puri-Psi started life as a fork of the [Purify](https://github.com/basp-group/purify) software package, developed in collaboration with UCL.

## References and citation

A. Repetti, M. Pereyra and Y. Wiaux. Uncertainty Quantification in Imaging: When Convex Optimization Meets Bayesian Analysis. In Proceedings of the 26th European Signal Processing Conference (EUSIPCO 2018), Rome, Italy, 3-8 Sept. 2018


## Acknowledgements

## License


```
Puri-Psi: Copyright (C) 2015-2020

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details (LICENSE.txt).

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
```
