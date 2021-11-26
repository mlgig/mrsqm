#ifndef SFA_H_
#define SFA_H_

#include "DFT.h"
#include "TimeSeries.h"

#include <iostream>
#include <map>
#include <vector>

class SFA {
private:

	std::vector< std::vector<double> > lookuptable;
	std::vector< std::map<double, int> > multiHistogram;

	unsigned int histogramType = EQUI_DEPTH;
	unsigned int count = 0;
	unsigned int coefficients;
	unsigned int startOffset = 0;

	int symbols;

	void initHistogram();
	void createHistogram( std::vector< std::shared_ptr<TimeSeries> > &samples, const int from );

	void divideByBreakPoints(double* data, unsigned int dataSize, unsigned short* signal);
	void divideEquiDepthHistogram (int from);
	void divideEquiWidthHistogram (int from);


public:
	static const unsigned int EQUI_FREQUENCY = 0;
	static const unsigned int EQUI_DEPTH = 1;

	DFT* transformation;

	SFA(unsigned int histogramType, unsigned int windowSize, unsigned int coefficients, unsigned int symbols, bool normMean);
	virtual ~SFA();

	void divideHistogram(std::vector< std::shared_ptr<TimeSeries> > & samples, int from);
	// void printHistogram();
	void lookup(TimeSeries & signature, unsigned short* word);
	void lookup(double* data, unsigned int dataSize, unsigned short* word);

	static void test();
};

#endif /* SFA_H_ */
