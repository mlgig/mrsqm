#ifndef MFT_H_
#define MFT_H_

#include "TimeSeries.h"
#include "DFT.h"
#include "SFA.h"

class MFT {

private:
	bool normMean = false;
	DFT* fft;
	SFA* sfa;

	uint windowSize = 0;
	uint startOffset = 0;

	double norm;

	int createWord(double* data, uint n, double std, unsigned short* word);
	double* createArray(double* data, uint n, double std);

	double realephi(double u, double M);
	double complexephi(double u, double M);

	double complexMulReal(double r1, double im1, double r2, double im2);
	double complexMulImag(double r1, double im1, double r2, double im2);

public:
	MFT(uint windowSize, bool normMean, SFA* sfa);
	virtual ~MFT();

	std::vector<int> transform(std::shared_ptr<TimeSeries> timeSeries, uint n);
	std::vector<std::vector<unsigned short>> transform2Array(std::shared_ptr<TimeSeries> timeSeries, uint n);

	static void printData(double* data, int size);
	static void test();

};

#endif /* MFT_H_ */
