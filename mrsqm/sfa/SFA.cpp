/*
 *      Author: (C) Patrick Schäfer
 *      		patrick.schaefer@zib.de
 */
#include "SFA.h"

#include <math.h>
#include <cmath>
#include <string>
#include <limits>
#include <iomanip>
#include "MFT.h"
// #include <boost/lexical_cast.hpp>

SFA::SFA(unsigned int histogramType, unsigned int windowSize, unsigned int coefficients, unsigned int symbols, bool normMean) {
	this->histogramType = histogramType;
	this->transformation = new DFT(windowSize);
	this->multiHistogram.resize(coefficients);
	this->coefficients = coefficients;
	this->symbols = symbols;
	this->lookuptable.resize( coefficients, std::vector<double>( symbols-1 , INFINITY) );
	this->count = 0;
	this->startOffset = normMean? 2 : 0;
}

SFA::~SFA() {
	delete transformation;
}

void SFA::divideHistogram(std::vector< std::shared_ptr<TimeSeries> > & samples, int from) {
	createHistogram(samples, from);

	if (this->histogramType == EQUI_DEPTH) {
		divideEquiDepthHistogram(from);
	} else if (this->histogramType == EQUI_FREQUENCY) {
		divideEquiWidthHistogram(from);
	}
}

// void SFA::printHistogram() {
// 	std::cout << "[";
// 	for (auto element : this->lookuptable) {
// 		std::cout << "-Inf\t";
// 		for (double element2 : element) {
// 			std::cout << ","
// 					<< (element2 != INFINITY? boost::lexical_cast<std::string>(element2) : "Inf")
// 					<< "\t";
// 		}
// 		std::cout << ";" << std::endl;
// 	}
// 	std::cout << "]" << std::endl;
// }

void SFA::createHistogram(std::vector< std::shared_ptr<TimeSeries> > & samples, int from) {

	for (unsigned int i = from; i < samples.size(); i++) {
		// transform each sample using DFT
		double out[coefficients];

		// norm the time series
		samples[i]->norm();

		// transform the time series
		transformation->transform(
				samples[i]->getData(), samples[i]->getSize(), out, coefficients, startOffset, true);

//		MFT::printData(out, coefficients);

		for (unsigned int j=0; j < coefficients; j++) {
			double value = round(out[j]*100.0)/100.0;

			// increase the count
			auto iter = this->multiHistogram[j].find(value);
			if (iter == this->multiHistogram[j].end()) {
				this->multiHistogram[j][value] = 1;
			} else {
				iter->second++;
			}
		}
		this->count++;
	}
}

void SFA::divideByBreakPoints(double* data, unsigned int dataSize, unsigned short* word) {
	for (unsigned int a = 0; a < dataSize; a++) {
		double value = data[a];
		// search for the corresponding symbols
		short beta = 0;
		for (beta = 0; beta < this->symbols-1; beta++) {
			if (value < this->lookuptable[a][beta]) {
				break;
			}
		}
		word[a] = beta;
	}
}

void SFA::divideEquiDepthHistogram (int from) {
	// for each dimension
	for (unsigned int i = 0; i < this->lookuptable.size(); i++) {
		double intervalSize = this->count / (double)(this->symbols);

		// divide into equi-depth intervals of size intervalSize
		long currentIntervalDepth = 0;
		unsigned int beta = 0;
		for (auto it : this->multiHistogram[i]) {
			// add the counts for this dimension
			currentIntervalDepth +=  it.second;
			if (currentIntervalDepth > ceil(intervalSize*(beta+1))
					&& beta < this->lookuptable[i].size()) {
				this->lookuptable[i][beta++] = it.first;
			}
		}
	}
}

void SFA::divideEquiWidthHistogram (int from) {
	// for each dimension
	for (unsigned int i = 0; i < this->lookuptable.size(); i++) {
		double intervalSize = this->multiHistogram[i].size() / (double)(this->symbols);

		// divide into equi-depth intervals of size intervalSize
		long currentIntervalSize = 0;
		unsigned int beta = 0;
		for (auto it : this->multiHistogram[i]) {
			currentIntervalSize++;
			if (currentIntervalSize > intervalSize*(beta+1)
					&& beta < this->lookuptable[i].size()) {
				this->lookuptable[i][beta++] = it.first;
			}
		}
	}
}

void SFA::lookup(double* data, unsigned int dataSize, unsigned short* word) {
    // use the lookup table to discretise the words
    divideByBreakPoints(data, dataSize, word);
}

void SFA::lookup(TimeSeries & signature, unsigned short* word) {
    // use the lookup table to discretise the words
    divideByBreakPoints(signature.getData(), signature.getSize(), word);
}

void SFA::test() {

	int size = 512;
	double* data = new double[size];
	for (int i = 0; i < size; i++) {
		data[i] = i*i;
	}

	TimeSeries ts(data, size, 0);

	int windowSize = 64;
	int coefficients = 8;
	int symbols = 4;
	bool normMean = true;

	SFA sfa(EQUI_FREQUENCY, windowSize, coefficients, symbols, normMean);

	// create histogramm + distribution
	std::vector<std::shared_ptr<TimeSeries> > windows;
	windows.reserve(size / windowSize);

	for (auto t2 : ts.getDisjointSequences(windowSize, normMean)) {
		t2->norm();
		windows.emplace_back(t2);
		MFT::printData(t2->getData(), t2->getSize());
	}

	sfa.divideHistogram(windows, 0);
	// sfa.printHistogram();
}

