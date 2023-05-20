/*
 *      Author: (C) Patrick Schäfer
 *      		patrick.schaefer@zib.de
 */
#include "DFT.h"
#include "TimeSeries.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

DFT::DFT(uint inSize) {
	this->fftSize = inSize;

	// create FFTW plan
	#pragma omp critical (make_plan)
	{
		double in[inSize];
		fftw_complex* out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (fftSize/2+1));
		this->plan = fftw_plan_dft_r2c_1d(inSize, in, out, FFTW_ESTIMATE);
		fftw_free(out);
	}
}

DFT::~DFT() {
	fftw_destroy_plan(plan);
	plan = NULL;
}


void DFT::transform(
		double* in, uint inSize,
		double* data, uint dataSize,
		uint startOffset,
		bool lowerBoundingNorm) {
//	if (inSize != fftSize) {
//		std::cout<<"Warning sizes different: " << inSize << "!=" << fftSize << std::endl;
//	}

	fftw_complex* out = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (fftSize/2+1));

	// perform the dft
	fftw_execute_dft_r2c(plan, in, out);

	// use only the first coefficients
	uint n = std::min(dataSize, this->fftSize*2-startOffset);

	out[0][1] = 0;

	// normalize the data
	if (lowerBoundingNorm) {
		double norm = 1.0/sqrt(fftSize);
		int offset = startOffset/2;
		for(uint i = 0; i < n; i+=2) {
			data[i] = out[i/2+offset][0]*norm;
			data[i+1] = -1*out[i/2+offset][1]*norm;
		}
	}
	else {
		int offset = startOffset/2;
		for(uint i = 0; i < n; i+=2) {
			data[i] = out[i/2+offset][0];
			data[i+1] = out[i/2+offset][1];
		}
	}

	fftw_free(out);
}

