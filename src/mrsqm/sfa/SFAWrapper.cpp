#include "SFA.h"
#include "MFT.h"
#include "TimeSeries.h"
#include <string.h>

#include <time.h>
#include <stdio.h>
#include <iomanip>
#include <limits>
#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <string>
#include <sstream>
#include <array>
#include <utility>
#include <limits.h>
#include <math.h>
#include <stdlib.h>

class SFAWrapper
{
private:
    unsigned int windowSize;
    unsigned int maxFeatures; // word length
    unsigned int maxSymbols; // alphabet size
    bool normMean;
    bool normTS;
    SFA *sfa;

    std::vector<std::shared_ptr<TimeSeries>> toTimeSeriesData(std::vector<std::vector<double>> &X)
    {
        std::vector<std::shared_ptr<TimeSeries>> samples;
        for (int i = 0; i < X.size(); i++)
        {
            std::shared_ptr<TimeSeries> ts = std::make_shared<TimeSeries>(X[i], 0); // fake label as it's not important
            if (normTS == true) {
                ts->norm(true);
            }
            
            samples.emplace_back(ts);
        }
        return samples;
    }

    std::string word2string(std::vector<unsigned short> &word, int alphabetSize)
    {
        char startChar = '!';
        std::string strWord = "";
        for (uint i = 0; i < word.size(); i++)
        {
            strWord += startChar + i * alphabetSize + word[i];
        }
        return strWord;
    }

public:
    SFAWrapper(int window_size, int word_length, int alphabet_size, bool normalization, bool normTimeSeries)    
    {
        this->windowSize = window_size;
        this->maxFeatures = word_length;
        this->maxSymbols = alphabet_size;
        this->normMean = normalization;
        this->normTS = normTimeSeries;

    }
    void fit(std::vector<std::vector<double>> X)
    {
        std::vector<std::shared_ptr<TimeSeries>> samples = toTimeSeriesData(X);

        sfa = new SFA(SFA::EQUI_DEPTH, windowSize, maxFeatures, maxSymbols, normMean);

        // create histogramm + distribution
        std::vector<std::shared_ptr<TimeSeries>> windows;
        windows.reserve(samples.size() * samples[0]->getSize() / windowSize);

        for (auto t : samples)
        {
            for (auto t2 : t->getDisjointSequences(windowSize, normMean))
            {
                if (normTS == true){
                    t2->norm();
                }                
                windows.emplace_back(t2);
            }
        }

        sfa->divideHistogram(windows, 0);
    }

    std::vector<std::string> transform(std::vector<std::vector<double>> X)
    {
        std::vector<std::shared_ptr<TimeSeries>> samples = toTimeSeriesData(X);
        std::vector<std::string> seqs;
        MFT fft(windowSize, normMean, sfa);

        for (auto ts : samples)
        {
            std::vector<std::vector<unsigned short>> words = fft.transform2Array(ts, maxFeatures);
            std::string seq = word2string(words[0], maxSymbols);
            
            

            for (uint i = 1; i < words.size(); i++)
            {
                seq += " " + word2string(words[i], maxSymbols);
            }

            seqs.emplace_back(seq);
        }
        return seqs;
    }
};

void printHello()
{
    std::cout << "Hello" << std::endl;
}

