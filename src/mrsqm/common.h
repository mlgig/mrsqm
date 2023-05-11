#ifndef FREQT_COMMON_H
#define FREQT_COMMON_H

#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <sstream>
#include <cmath>

namespace constants
{
const int DEFAULT_MIN_WINDOW_SIZE = 16;
const int DEFAULT_WORD_LENGTH = 16;
const int DEFAULT_ALPHABET_SIZE = 4;

const std::string TIME_SERIES_DELIMITER = ",";
}

template <class Iterator>
static inline unsigned int tokenize (char *str, char *del, Iterator out, unsigned int max)
{
	char *stre = str + strlen (str);
	char *dele = del + strlen (del);
	unsigned int size = 1;

	while (size < max) {
		char *n = std::find_first_of (str, stre, del, dele);
		*n = '\0';
		*out++ = str;
		++size;
		if (n == stre) break;
		str = n + 1;
	}
	*out++ = str;

	return size;
}

size_t find_subsequence(std::string seq,std::string subseq,double distance, size_t pos){
	if (distance == 0){
		return seq.find(subseq, pos);
	} else {
		for (size_t i = pos; i <= seq.length() - subseq.length() ;i++){
			double budget = distance;
			for (size_t j = 0; j < subseq.length();j++){
				budget = budget - abs(seq.at(i+j) - subseq.at(j));
				if (budget < 0){
					break;
				}
			}
			if (budget >= 0){
				return i;
			}
		}
	}

	return std::string::npos;
}



std::vector<double> string_to_double_vector(std::string str,std::string delimiter){
	std::vector<double> numeric_ts;
	size_t pos = 0;
	std::string token;

	while ((pos = str.find(delimiter)) != std::string::npos) {
		token = str.substr(0, pos);
		//std::cout << token << " ";
		numeric_ts.push_back(atof(token.c_str()));
		str.erase(0, pos + delimiter.length());
	}
	if (!str.empty()){
		numeric_ts.push_back(atof(str.c_str()));
	}
	return numeric_ts;
}

std::vector<int> string_to_int_vector(std::string str,std::string delimiter){
	std::vector<int> numeric_ts;
	size_t pos = 0;
	std::string token;

	while ((pos = str.find(delimiter)) != std::string::npos) {
		token = str.substr(0, pos);
		//std::cout << token << " ";
		numeric_ts.push_back(atoi(token.c_str()));
		str.erase(0, pos + delimiter.length());
	}
	if (!str.empty()){
		numeric_ts.push_back(atoi(str.c_str()));
	}
	return numeric_ts;
}

template <class T>
void print_vector_of_vector(std::vector<std::vector<T>> input){
	for(std::vector<T> vt: input){
		for(T v: vt){
			std::cout << v << " ";
		}
		std::cout << std::endl;
	}
}

template <typename T>
std::string join(const T& v, const std::string& delim) {
	std::ostringstream s;
	for (const auto& i : v) {
		if (&i != &v[0]) {
			s << delim;
		}
		s << i;
	}
	return s.str();
}

double chi_square_score(std::vector<int>& observed, std::vector<double>& y_prob){
	int feature_count = 0;
	for (int c : observed){
		feature_count += c;
	}


	double chisq_score = 0.0;
	for (int i = 0; i < observed.size(); i++){
		double expected = y_prob[i] * feature_count;

		chisq_score += pow(observed[i] - expected,2.0) / expected;
	}


	return chisq_score;

}

double chi_square_bound(std::vector<int>& observed, std::vector<double>& y_prob){


	double bound = 0.0;
	for (int i = 0; i < observed.size(); i++){
		if (observed[i] > 0){
			std::vector<int> new_observed (observed.size(),0);
			new_observed[i] = observed[i];
			double new_chi2 = chi_square_score(new_observed, y_prob);
			if (new_chi2 >= bound){
				bound = new_chi2;
			}
		}

	}
	return bound;

}

double cp_entropy(double p){
	if (p > 0 && p < 1){
		return -(p * log(p) + (1.0 - p) * log(1.0 - p));
	}
	return 0.0;
}

/* lp: positive count on the left side of the split
 * ln: negative count on the left side of the split
 * rp: positive count on the right side of the split
 * rn: negative count on the right side of the split
 */
double split_entropy(int lp, int ln, int rp, int rn){
	double p_l = (lp > 0) ? lp * 1.0 / (lp + ln) : 0.0;
	double p_r = (rp > 0) ? rp * 1.0 / (rp + rn) : 0.0;
	return ((lp + ln) * cp_entropy(p_l) + (rp + rn) * cp_entropy(p_r))/ (lp + ln + rp + rn);
}

double compute_entropy_lowerbound(int lp, int ln, int rp, int rn){
	return std::min(split_entropy(lp, 0, rp, ln + rn),split_entropy(0, ln, rp + lp, rn));
}

static double igf(double S, double Z)
{
	if(Z < 0.0)
	{
		return 0.0;
	}
	double Sc = (1.0 / S);
	Sc *= pow(Z, S);
	Sc *= exp(-Z);

	double Sum = 1.0;
	double Nom = 1.0;
	double Denom = 1.0;

	for(int I = 0; I < 200; I++)
	{
		Nom *= Z;
		S++;
		Denom *= S;
		Sum += (Nom / Denom);
	}

	return Sum * Sc;
}

double approx_gamma(double Z)
{
	const double RECIP_E = 0.36787944117144232159552377016147;  // RECIP_E = (E^-1) = (1.0 / E)
	const double TWOPI = 6.283185307179586476925286766559;  // TWOPI = 2.0 * PI

	double D = 1.0 / (10.0 * Z);
	D = 1.0 / ((12 * Z) - D);
	D = (D + Z) * RECIP_E;
	D = pow(D, Z);
	D *= sqrt(TWOPI / Z);

	return D;
}

double chi2_pvalue(int Dof, double Cv)
{
	
	if(Cv < 0 || Dof < 1)
	{
		return 0.0;
	}
	if (Cv == 0)
	{
		return 1.0;

	}

	double K = ((double)Dof) * 0.5;
	double X = Cv * 0.5;
	if(Dof == 2)
	{
		return exp(-1.0 * X);
	}

	double PValue = igf(K, X);
	if(std::isnan(PValue) || std::isinf(PValue) || PValue <= 1e-8)
	{
		return 1e-14;
	}

	PValue /= tgamma(K);
	//PValue /= tgamma(K);

	return (1.0 - PValue);
}


#endif
