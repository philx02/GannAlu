#pragma once

// This function must be thread safe
template< typename Ann >
double fitnessFunction(const Ann &iAnn)
{
  Ann::Input wInput;
  double wScore = 0;
  for (double wIter = 0; wIter < 6.28; wIter += 0.1)
  {
    wInput[0] = wIter;
    auto wOutput = iAnn.compute(wInput);
    wScore -= std::abs(wOutput[0] - wIter);
  }
  return wScore;
}
