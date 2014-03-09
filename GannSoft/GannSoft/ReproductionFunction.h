#pragma once

template< typename Ann, typename RandomNumberGenerator >
Ann cloneAndMutate(const Ann &iSource, RandomNumberGenerator &iRandomNumberGenerator)
{
  static auto wWeightMutationDistribution = std::uniform_real_distribution<>(-1.0, 1.0);
  static auto wWeightRangeDistribution = std::uniform_int_distribution<>(0, sTotalWeightSize - 1);
  auto wOffspring = iSource;
  for (auto wIter = 0; wIter < sTotalNumberOfMutation; ++wIter)
  {
    auto wIndexToMutate = wWeightRangeDistribution(iRandomNumberGenerator);
    if (wIndexToMutate < sInputToHiddenWeightSize)
    {
      // TODO: templatize?
      wOffspring.applyToInputToHiddenWeights([&](Ann::InputToHiddenWeights &iWeights)
      {
        iWeights[wIndexToMutate / sInputLayerSize][wIndexToMutate%sInputLayerSize] += wWeightMutationDistribution(iRandomNumberGenerator);
      });
    }
    else
    {
      wIndexToMutate -= sInputToHiddenWeightSize;
      wOffspring.applyToHiddenToOutputWeights([&](Ann::HiddenToOutputWeights &iWeights)
      {
        iWeights[wIndexToMutate / sHiddenLayerSize][wIndexToMutate%sHiddenLayerSize] += wWeightMutationDistribution(iRandomNumberGenerator);
      });
    }
  }
  return wOffspring;
}

template< typename Individual, typename Iterator, typename RandomNumberGenerator >
Individual reproductionFunction(Iterator iBegin, Iterator iEnd, RandomNumberGenerator &iRandomNumberGenerator)
{
  static auto wNormalDistribution = std::normal_distribution<>(0, 100);
  std::size_t wPopulationSize = iEnd - iBegin;
  assert(wPopulationSize > 0);
  std::size_t wIndividualIndex = 0;
  do
  {
    wIndividualIndex = static_cast< std::size_t >(std::abs(wNormalDistribution(iRandomNumberGenerator)));
  } while (wIndividualIndex >= wPopulationSize);

  return cloneAndMutate(iBegin[wIndividualIndex].mIndividual, iRandomNumberGenerator);
}
