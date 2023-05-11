/*
 *      Author: (C) Patrick Schäfer
 *      		patrick.schaefer@zib.de
 */
#include "MFT.h"
#include <math.h>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <iterator>

#define PI 3.14159265

MFT::MFT(uint windowSize, bool normMean, SFA* sfa) {
	this->windowSize = windowSize;
	this->normMean = normMean;
	this->startOffset = normMean? 2:0;
	this->sfa = sfa;
	if (sfa != NULL) {
		this->fft = sfa->transformation;
	}
	else {
		this->fft = new DFT(windowSize);
	}
	this->norm = 1.0 / sqrt(windowSize);
}

MFT::~MFT() {
}


void MFT::printData(double* data, int size) {
	std::copy(data, data + size, std::ostream_iterator<double>(std::cout, ","));
	std::cout << std::endl;
}

std::vector<int> MFT::transform(std::shared_ptr<TimeSeries> timeSeries, uint n){

	int size = n+this->startOffset;
	double phis[n+this->startOffset];
	for (int u = 0; u < size; u+=2) {
		double uHalve = -u/2;
		phis[u] = realephi(uHalve, windowSize);
		phis[u+1] = complexephi(uHalve, windowSize);
	}

	uint end = std::max((uint)1, timeSeries->getSize()-windowSize+1);

	// calculate means and stds for each sample
	std::vector< double > means;
	std::vector< double > stds;
	timeSeries->calcIncreamentalMeanStddev(
			std::min(timeSeries->getSize(), windowSize), means, stds);

	std::vector<int> transformed(end);

	uint arraySize = std::max(n+this->startOffset, windowSize);
	double mftData[arraySize];
	double mftData2[n];

	double* data = timeSeries->getData();
	unsigned short word[n];

	for (uint t = 0; t < end; t++) {
		if (t > 0) {
			// perform the MFT instead on a full DFT
			for (uint k = this->startOffset; k < n+this->startOffset; k+=2) {
				double real1 = (mftData[k] + data[t+windowSize-1] - data[t-1]);
				double imag1 = (mftData[k+1]);

				double real = complexMulReal(real1, imag1, phis[k], phis[k+1]);
				double imag = complexMulImag(real1, imag1, phis[k], phis[k+1]);

				mftData[k] = real;
				mftData[k+1] = imag;

				mftData2[k-this->startOffset] = mftData[k];
				mftData2[k-this->startOffset+1] = mftData[k+1];
			}
		}
		else {
			std::fill(mftData, mftData+arraySize, 0);
			std::fill(mftData2, mftData2+arraySize, 0);

			// perform one full DFT transform
			fft->transform(timeSeries->getData(), windowSize, mftData, arraySize, 0, false);

			// make a copy of the data
			std::copy(mftData+startOffset, mftData+n+startOffset, mftData2);
		}

		transformed[t] = createWord(mftData2, n, stds[t], word);
//		transformed[t] = createArray(mftData2, n, stds[t]);
//		std::cout << "Res : ";
//		printData(transformed[t], n);

	}

	return transformed;
}

// Thach: output each word as an array instead
std::vector<std::vector<unsigned short>> MFT::transform2Array(std::shared_ptr<TimeSeries> timeSeries, uint n) {
	int size = n+this->startOffset;
	double phis[n+this->startOffset];
	for (int u = 0; u < size; u+=2) {
		double uHalve = -u/2;
		phis[u] = realephi(uHalve, windowSize);
		phis[u+1] = complexephi(uHalve, windowSize);
	}

	uint end = std::max((uint)1, timeSeries->getSize()-windowSize+1);

	// calculate means and stds for each sample
	std::vector< double > means;
	std::vector< double > stds;
	timeSeries->calcIncreamentalMeanStddev(
			std::min(timeSeries->getSize(), windowSize), means, stds);

	std::vector<std::vector<unsigned short>> transformed(end);

	uint arraySize = std::max(n+this->startOffset, windowSize);
	double mftData[arraySize];
	double mftData2[n];

	double* data = timeSeries->getData();
	//unsigned short word[n];

	for (uint t = 0; t < end; t++) {
		unsigned short word[n];
		if (t > 0) {
			// perform the MFT instead on a full DFT
			for (uint k = this->startOffset; k < n+this->startOffset; k+=2) {
				double real1 = (mftData[k] + data[t+windowSize-1] - data[t-1]);
				double imag1 = (mftData[k+1]);

				double real = complexMulReal(real1, imag1, phis[k], phis[k+1]);
				double imag = complexMulImag(real1, imag1, phis[k], phis[k+1]);

				mftData[k] = real;
				mftData[k+1] = imag;

				mftData2[k-this->startOffset] = mftData[k];
				mftData2[k-this->startOffset+1] = mftData[k+1];
			}
		}
		else {
			std::fill(mftData, mftData+arraySize, 0);
			std::fill(mftData2, mftData2+arraySize, 0);

			// perform one full DFT transform
			fft->transform(timeSeries->getData(), windowSize, mftData, arraySize, 0, false);

			// make a copy of the data
			std::copy(mftData+startOffset, mftData+n+startOffset, mftData2);
		}

		//transformed[t] = createWord(mftData2, n, stds[t], word);
		double inverseStd = this->norm * stds[t];
		for (uint i = 0; i < n; i+=2) {
			mftData2[i] *= inverseStd;
			mftData2[i+1] *= -inverseStd;
		}
		sfa->lookup(mftData2, n, word);
		//std::cout << word[0] << std::endl;
		transformed[t] = std::vector<unsigned short> (word, word + sizeof word / sizeof word[0]);
		//std::copy(word, word+4, transformed[t]);
		//std::copy(std::begin(word), std::end(word), std::begin(transformed[t]));
//		transformed[t] = createArray(mftData2, n, stds[t]);
//		std::cout << "Res : ";
//		printData(transformed[t], n);

	}

	return transformed;
}

double* MFT::createArray(double* data, uint n, double std) {
	// norm the data
	double * newData = new double[n];
	double inverseStd = this->norm * std;
	for (uint i = 0; i < n; i+=2) {
		newData[i] = inverseStd * data[i];
		newData[i+1] = -1*inverseStd  * data[i+1];
	}
	return newData;
}

int MFT::createWord(double* data, uint n, double std, unsigned short* word) {
	// norm the data
	double inverseStd = this->norm * std;
	for (uint i = 0; i < n; i+=2) {
		data[i] *= inverseStd;
		data[i+1] *= -inverseStd;
	}

	// quantization
	sfa->lookup(data, n, word);


       
	

	uint usedBits = 2;
	uint shortsPerLong = 60 / usedBits;
	unsigned long bits = 0;
	uint start = 0;
	unsigned long shiftOffset = 1;

	// build a long
	for (uint i=start, end = std::min(n, shortsPerLong+start); i<end; i++) {
	  for (uint j = 0, shift = 1; j < usedBits; j++, shift <<= 1) {
		if ((word[i] & shift) != 0) {
		  bits |= shiftOffset;
		}
		shiftOffset <<= 1;
	  }
	}
	return (int) bits;
}

double MFT::complexMulReal(double r1, double im1, double r2, double im2) {
	return r1*r2 - im1*im2;
}

double MFT::complexMulImag(double r1, double im1, double r2, double im2) {
	return r1*im2 + r2*im1;
}

double MFT::realephi(double u, double M) {
	return cos(2*PI*u/M);
}

double MFT::complexephi(double u, double M) {
	return -sin(2*PI*u/M);
}

void MFT::test() {
	int size = 512;
    double* data = new double[size];
    for (int i = 0; i < size; i++) {
    	data[i] = i*i;
    }
    MFT::printData(data, size);
    std::shared_ptr<TimeSeries> ts = std::make_shared<TimeSeries>(data, size, 0);

    uint windowSize = 256;
    int n = 10;
	int symbols = 4;
	bool normMean = false;

    SFA sfa(SFA::EQUI_FREQUENCY, windowSize, n, symbols, normMean);

	// create histogramm + distribution
	std::vector<std::shared_ptr<TimeSeries> > windows;
	windows.reserve(size / windowSize);

	for (auto t2 : ts->getDisjointSequences(windowSize, normMean)) {
		t2->norm();
		windows.emplace_back(t2);
		MFT::printData(t2->getData(), t2->getSize());
	}

	std::cout<<"Divide histogram"<<std::endl;
	sfa.divideHistogram(windows, 0);
	// sfa.printHistogram();

    MFT mft(windowSize, normMean, &sfa);
    std::vector<int> words = mft.transform(ts, n);

    std::cout << "Resulting words" << std::endl;
    for (int d : words) {
    	std::cout << d << " ";
    }
    std::cout << std::endl;
}
