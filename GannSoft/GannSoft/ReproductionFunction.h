#pragma once

#include "Definitions.h"
#include "GeneticAlgorithm.h"

template< typename Individual, typename Iterator >
Individual reproductionFunction(Iterator iBegin, Iterator iEnd)
{
  return iBegin->mIndividual;
}
