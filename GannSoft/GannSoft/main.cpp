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

template< typename T, std::size_t Span, std::size_t Offset >
inline T logisticFunction(const T &iInput)
{
  static const auto wSpan = static_cast< T >(Span);
  static const auto wOffset = static_cast< T >(Offset);
  static const auto wOne = static_cast< T >(1);
  return wSpan / (wOne + std::exp(-iInput)) + wOffset;
};

template< typename T >
inline double linearFunction(const T &iInput)
{
  return iInput;
}

template< typename T, std::size_t LowerLimit, std::size_t HigherLimit >
inline double saturatedLinearFunction(const T &iInput)
{
  static const auto wLowerLimit = static_cast< T >(LowerLimit);
  static const auto wHigherLimit = static_cast< T >(HigherLimit);
  return iInput > wHigherLimit ? wHigherLimit : (iInput < wLowerLimit ? wLowerLimit : iInput);
}

auto wInitialWeightsDistribution = std::uniform_real_distribution<>(-1.0, 1.0);
std::mt19937 wMt19937(static_cast< std::mt19937::result_type >(std::time(nullptr)));
std::ofstream wOut;

auto wCreateAnn = [&]()
{
  return createAndInitializeArtificialNeuralNetwork< double, sInputLayerSize, sHiddenLayerSize, sOutputLayerSize >(logisticFunction< double, 1, 0 >, logisticFunction< double, 1, 0 >, [&]() -> double {return wInitialWeightsDistribution(wMt19937); });
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

void caracterize()
{
  LongitudinalAircraft wLongitudinalAircraft(sRotationalInertia, sElevatorToCogDistance, sMass, sElevatorCommandToForceOnTailFactor, sThrottleCommandToThrustFactor, sLiftConstant, sDragConstant, 0, 0, 50, 0, 0, sEquilibriumHorizontalSpeed);

  double wDeltaT = 0.1;
  std::ofstream wFile("test.csv");
  wFile << "Time,Elevator,Throttle,Altitude,Speed,pitch\n";
  double KPpitch = 0.1;
  double KDpitch = -0.5;
  double KPthrottle = 5;
  for (double wTime = 0; wTime < 30; wTime += wDeltaT)
  {
    double wDesiredPitch = (100 - wLongitudinalAircraft.verticalPosition()) / 50 * bmc::pi/4;
    double wPitchError = wDesiredPitch - wLongitudinalAircraft.pitchAngle();
    double wElevator = KPpitch * wPitchError + KDpitch * wLongitudinalAircraft.pitchAngularSpeed();
    wElevator = wElevator > 1 ? 1 : wElevator < -1 ? -1 : wElevator;
    
    double wDesiredSpeed = 8;
    double wSpeedError = wDesiredSpeed - wLongitudinalAircraft.bodySpeed();
    double wThrottle = KPthrottle * wSpeedError;
    wThrottle = wThrottle > 1 ? 1 : wThrottle < 0 ? 0 : wThrottle;
    
    wLongitudinalAircraft.perform(wElevator, wThrottle, wDeltaT);
    wFile << wTime << "," << wElevator << "," << wThrottle << "," << wLongitudinalAircraft.verticalPosition() << "," << wLongitudinalAircraft.bodySpeed() << "," << wLongitudinalAircraft.pitchAngle() << "\n";
  }
  exit(0);
}

int main()
{
  //caracterize();
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

  std::ofstream wFile;
  auto wOutput = [&](double iTime, const Ann::Output &iOutput, const LongitudinalAircraft &iLongitudinalAircraft)
  {
    double wAltitudeContribution = std::abs(iLongitudinalAircraft.verticalPosition() - 100);
    double wSpeedContribution = std::abs(iLongitudinalAircraft.bodySpeed() - 8) * (50 / 8);
    double wPitchError = std::abs(iLongitudinalAircraft.pitchAngle());
    double wPitchContribution = std::min(0.25*std::exp(7.0*wPitchError), 200.0);
    wFile << iTime << "," << (iOutput[0] * 2 - 1) << "," << iOutput[1] << "," << iLongitudinalAircraft.verticalPosition() << "," << iLongitudinalAircraft.bodySpeed() << "," << iLongitudinalAircraft.pitchAngle() << "," << wAltitudeContribution << "," << wSpeedContribution << "," << wPitchContribution << "\n";
  };

  wFile.open("Performance/random.csv");
  fitnessFunction(wCreateAnn(), wOutput);
  wFile.close();

  while (true)
  {
    auto wNoOp = [](double, const Ann::Output &, const LongitudinalAircraft &) {};
    wGa.runGenerations(1000, [&](const Ann &iIndividual) {return fitnessFunction(iIndividual, wNoOp); }, wReproductionFunction, [](RatedIndividualIterator iBegin, RatedIndividualIterator iEnd) {}, wPostGenerationEvaluation);
    auto &&wBestAnn = wGa.begin()->mIndividual;

    std::ostringstream wFilename;
    wFilename << "Performance/best_of_gen" << std::setw(10) << std::setfill('0') << boost::lexical_cast< std::string >(wGeneration) << ".csv";
    wFile.open(wFilename.str().c_str());
    wFile << "Time,ElevatorCommand,ThrottleCommand,Altitude,Speed,Pitch\n";
    fitnessFunction(wBestAnn, wOutput);
    wFile.close();
  }

  return 0;
}
