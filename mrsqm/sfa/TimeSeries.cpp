/*
 *      Author: (C) Patrick Schäfer
 *      		patrick.schaefer@zib.de
 */
#include "TimeSeries.h"
#include <math.h>
#include <stddef.h>

#include <algorithm>

TimeSeries::TimeSeries() {
	init(NULL, 0, -1);
}

TimeSeries::TimeSeries(double* data, int size, int label) {
	init(data, size, label);
}

TimeSeries::TimeSeries(const std::vector<double> & tsData, int label){
	int size = tsData.size();
	double* data = new double[size]; // convert vector to double[]
	std::copy(tsData.begin(), tsData.end(), data);
	init(data, size, label);
}

void TimeSeries::init(double* data, int size, int label) {
	this->data = data;
	this->size = size;
	this->stddev = 0;
	this->mean = 0;
	this->normed = false;
	this->label = label;
}

TimeSeries::~TimeSeries() {
	delete[] data;
}

double TimeSeries::getMean() {
	return this->mean;
}


double TimeSeries::calculateMean() {
	this->mean = 0.0;

	// calculate mean
	for (int i = 0; i < size; i++) {
		this->mean += data[i];
	}
	this->mean /= (double)size;

	return this->mean;
}

double TimeSeries::calculateStddev() {
	this->stddev = 0;

	// calculate stddev
	double var = 0;
	for (int i = 0; i < size; i++) {
		double d = data[i];
		var += d * d;
	}

	double norm = 1.0 / this->size;
	double buf = norm * var - this->mean*this->mean;
	if (buf > 0) {
		this->stddev = sqrt(buf);
	}

	return this->stddev;
}

double* TimeSeries::getData() {
	return this->data;
}

double TimeSeries::getStddev() {
	return this->stddev;
}

unsigned int TimeSeries::getSize() {
	return this->size;
}

double TimeSeries::getLabel() {
	return this->label;
}

void TimeSeries::setLabel(double label) {
	this->label = label;
}

bool TimeSeries::isNormed() {
	return this->normed;
}

void TimeSeries::norm() {
	norm(true);
}

void TimeSeries::norm (bool norm) {
	if (!isNormed()) {
		// calculate mean + stddev
		this->mean = calculateMean();
		this->stddev = calculateStddev();
		TimeSeries::norm(norm, this->mean, this->stddev);
	}
}

/**
 * Applies z-norming: sets mean to 0 and stddev to 1
 */
void TimeSeries::norm (bool normMean, double mean, double stddev) {
	if (!isNormed()) {
		this->mean = mean;
		this->stddev = stddev;

		double inverseStddev = 1.0 / (this->stddev>0? this->stddev : 1.0);

		if (normMean) {
			for (int i = 0; i < this->size; i++) {
				this->data[i] = (this->data[i] - this->mean) *  inverseStddev;
			}
			this->mean = 0.0;
		}
		else if (inverseStddev != 1.0) {
			for (int i = 0; i < this->size; i++) {
				this->data[i] *= inverseStddev;
			}
		}

		this->stddev = 1.0;

		this->normed = true;
	}
}

std::vector< std::shared_ptr<TimeSeries> > TimeSeries::getDisjointSequences(int w, bool normMean) {
	// extract subsequences
	int amount = (int)floor(this->size/(double)w);
	std::vector< std::shared_ptr<TimeSeries> > window(amount);

	for (int i=0; i < amount; i++) {
		int windowSize = std::min(w, this-> size - i*w);
		int startOffset = std::min(i*windowSize, this->size-windowSize);

		double* subsequenceData = new double[windowSize];
		std::copy(this->data+startOffset, this->data+startOffset+windowSize, subsequenceData);


		std::shared_ptr<TimeSeries> ts = std::make_shared<TimeSeries>(subsequenceData, windowSize, getLabel());
		ts->norm(normMean);
		window[i] = ts;
	}

	return window;
}

void TimeSeries::calcIncreamentalMeanStddev(int windowLength, std::vector< double > & means, std::vector< double > & stds) {
	double sum = 0;

	double squareSum = 0;
	// it is faster to multiply than to divide
	double rWindowLength = 1.0 / (double)windowLength;

	for (int ww = 0; ww < windowLength; ww++) {
		sum += data[ww];
		squareSum += data[ww]*data[ww];
	}
	means.push_back(sum * rWindowLength);
	double buf = squareSum * rWindowLength - means[0]*means[0];
	stds.push_back(buf > 0? 1.0 / sqrt(buf) : 1.0);

	for (int w = 1, end = size-windowLength+1; w < end; w++) {
		sum += data[w+windowLength-1] - data[w-1];
		means.push_back(sum * rWindowLength);

		squareSum += data[w+windowLength-1]*data[w+windowLength-1] - data[w-1]*data[w-1];
		buf = squareSum * rWindowLength - means[w]*means[w];
		stds.push_back(buf > 0? 1.0 / sqrt(buf) : 1.0);
	}
}


