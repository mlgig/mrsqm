#ifndef TIMESERIES_H_
#define TIMESERIES_H_

#include <vector>
#include <memory>

class TimeSeries {

private:
	double* data;

	int size;

	double mean;
	double stddev;

	double label;

	bool normed;

	bool isNormed();

	TimeSeries& operator= (TimeSeries const& rhs);
	TimeSeries(TimeSeries const& original);

public:
	TimeSeries(const std::vector<double> & tsData, int label);
	TimeSeries(double* data, int size, int label);
	TimeSeries();
	virtual ~TimeSeries();

	void init(double* data, int size, int label);

	double calculateMean();
	double calculateStddev();

	double getMean();
	double getStddev();

	double* getData();
	unsigned int getSize();

    double getLabel();
    void setLabel(double label);

    void norm();
    void norm(bool norm);
    void norm(bool normMean, double mean, double stddev);

    std::vector< std::shared_ptr<TimeSeries> > getDisjointSequences(int w, bool normMean);

    void calcIncreamentalMeanStddev(int windowLength, std::vector< double > & means, std::vector< double > & stds);


};


#endif /* TIMESERIES_H_ */
