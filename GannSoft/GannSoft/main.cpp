#include "Definitions.h"
#include <ArtificialNeuralNetwork.h>
#include <GeneticAlgorithm.h>
#include "FitnessFunction.h"
#include "ReproductionFunction.h"

#include <boost/lexical_cast.hpp>

#include <random>
#include <iostream>
#include <ctime>
#include <fstream>
#include <iomanip>

inline double logisticFunction(const double &iInput)
{
  return 2.0 / (1.0 + std::exp(-iInput)) - 1.0;
};

inline double linearFunction(const double &iInput)
{
  return iInput;
}

auto wInitialWeightsDistribution = std::uniform_real_distribution<>(-1.0, 1.0);
std::mt19937 wMt19937(static_cast< std::mt19937::result_type >(std::time(nullptr)));
std::ofstream wOut;

auto wCreateAnn = [&]()
{
  return createAndInitializeArtificialNeuralNetwork< double, sInputLayerSize, sHiddenLayerSize, sOutputLayerSize >(logisticFunction, linearFunction, [&]() -> double {return wInitialWeightsDistribution(wMt19937); });
};

typedef decltype(wCreateAnn()) Ann;

std::ostream & operator<<(std::ostream &iOstream, const Ann &iAnn)
{
  iAnn.applyToInputToHiddenWeights([&](const Ann::InputToHiddenWeights &iWeights)
  {
    for (const auto &iLayer : iWeights)
    {
      for (const auto &iWeight : iLayer)
      {
        iOstream << iWeight << ",";
      }
      iOstream << std::endl;
    }
    iOstream << std::endl;
    iOstream << std::endl;
  });
  iAnn.applyToHiddenToOutputWeights([&](const Ann::HiddenToOutputWeights &iWeights)
  {
    for (const auto &iLayer : iWeights)
    {
      for (const auto &iWeight : iLayer)
      {
        iOstream << iWeight << ",";
      }
      iOstream << std::endl;
    }
    iOstream << std::endl;
    iOstream << std::endl;
  });
  return iOstream;
}

int main()
{
  auto wGa = createGeneticAlgorithm< double, Ann >(1000, wCreateAnn);

  typedef decltype(wGa.begin()) RatedIndividualIterator;

  auto wReproductionFunction = [&](RatedIndividualIterator iBegin, RatedIndividualIterator iEnd)
  {
    return reproductionFunction< Ann, RatedIndividualIterator >(iBegin, iEnd, wMt19937);
  };

  std::size_t wGeneration = 0;
  auto wPostGenerationEvaluation = [&](RatedIndividualIterator iBegin, RatedIndividualIterator iEnd) -> bool
  {
    if (++wGeneration % 100 == 0)
    {
      std::cout << wGeneration << ": " << iBegin->mScore << std::endl;
      std::ostringstream wFilename;
      wFilename << "Output/Network" << std::setw(10) << std::setfill('0') << boost::lexical_cast< std::string >(wGeneration) << ".csv";
      wOut.open(wFilename.str().c_str());
      wOut << iBegin->mIndividual;
      wOut.close();
    }
    return true;
  };

  wGa.runGenerations(10000, fitnessFunction< Ann >, wReproductionFunction, [](RatedIndividualIterator iBegin, RatedIndividualIterator iEnd) {}, wPostGenerationEvaluation);
  auto &&wBestAnn = wGa.begin()->mIndividual;

  {
    std::ofstream wCsv("best.csv");
    std::array< double, 1 > wInput;
    for (double wIter = 0; wIter < 6.28; wIter += 0.1)
    {
      wInput[0] = wIter;
      wCsv << wIter << "," << wBestAnn.compute(wInput)[0] << "\n";
    }
  }

  {
    auto wRandomAnn = wCreateAnn();
    std::ofstream wCsv("random.csv");
    std::array< double, 1 > wInput;
    for (double wIter = 0; wIter < 6.28; wIter += 0.1)
    {
      wInput[0] = wIter;
      wCsv << wIter << "," << wRandomAnn.compute(wInput)[0] << "\n";
    }
  }

  return 0;
}
