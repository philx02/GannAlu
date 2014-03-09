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

//std::ostream & operator<<(std::ostream &iOstream, const Ann &iAnn)
//{
//  iAnn.applyToInputToHiddenWeights([&](const Ann::InputToHiddenWeights &iWeights)
//  {
//    for (const auto &iLayer : iWeights)
//    {
//      for (const auto &iWeight : iLayer)
//      {
//        iOstream << iWeight << ",";
//      }
//      iOstream << std::endl;
//    }
//    iOstream << std::endl;
//    iOstream << std::endl;
//  });
//  iAnn.applyToHiddenToOutputWeights([&](const Ann::HiddenToOutputWeights &iWeights)
//  {
//    for (const auto &iLayer : iWeights)
//    {
//      for (const auto &iWeight : iLayer)
//      {
//        iOstream << iWeight << ",";
//      }
//      iOstream << std::endl;
//    }
//    iOstream << std::endl;
//    iOstream << std::endl;
//  });
//  return iOstream;
//}

int main()
{
  auto wInitialWeightsDistribution = std::uniform_real_distribution<>(-1.0, 1.0);
  std::mt19937 wMt19937(static_cast< std::mt19937::result_type >(std::time(nullptr)));
  std::ofstream wOut;

  std::size_t wGeneration = 0;
  //auto wPostGenerationEvaluation = [&](const Ga::Population &iSortedPopulation) -> bool
  //{
  //  if (++wGeneration % 100 == 0)
  //  {
  //    std::cout << wGeneration << ": " << iSortedPopulation.front().mScore << std::endl;
  //    std::ostringstream wFilename;
  //    wFilename << "Output/Network" << std::setw(10) << std::setfill('0') << boost::lexical_cast< std::string >(wGeneration) << ".csv";
  //    wOut.open(wFilename.str().c_str());
  //    wOut << iSortedPopulation.front().mIndividual;
  //    std::array< double, 2 > wInputs = {wInitialWeightsDistribution(wMt19937), wInitialWeightsDistribution(wMt19937)};
  //    std::cout << (wInputs[0] * wInputs[1]) << " <> " << iSortedPopulation.front().mIndividual.compute(wInputs)[0] << std::endl;
  //    wOut.close();
  //  }
  //  return true;
  //};

  auto wCreateAnn = [&]()
  {
    return createAndInitializeArtificialNeuralNetwork< double, sInputLayerSize, sHiddenLayerSize, sOutputLayerSize >(logisticFunction, linearFunction, [&]() -> double {return wInitialWeightsDistribution(wMt19937); });
  };

  typedef decltype(wCreateAnn()) Ann;

  auto wGa = createGeneticAlgorithm< double, Ann >(1000, wCreateAnn);

  typedef decltype(wGa.begin()) RatedIndividualIterator;

  auto wReproductionFunction = [&](RatedIndividualIterator iBegin, RatedIndividualIterator iEnd)
  {
    return reproductionFunction< Ann, RatedIndividualIterator >(iBegin, iEnd, wMt19937);
  };

  wGa.runGenerations(1000000, fitnessFunction< Ann >, wReproductionFunction, [](RatedIndividualIterator iBegin, RatedIndividualIterator iEnd) {}, [](RatedIndividualIterator iBegin, RatedIndividualIterator iEnd) {return true; });

  return 0;
}
