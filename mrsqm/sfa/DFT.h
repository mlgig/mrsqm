#ifndef DFT_H_
#define DFT_H_

#include <fftw3.h>
#include <sys/types.h>

class DFT {
private:
	uint fftSize;
	fftw_plan plan;
//	fftw_complex *out;

public:
	DFT(uint fftSize);
	virtual ~DFT();

	void transform(double* in, uint inSize, double* out, uint outSize, uint startOffset, bool normalize);

};

#endif /* DFT_H_ */
