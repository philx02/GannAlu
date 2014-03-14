#pragma once

#include <boost/integer/static_min_max.hpp>

#include <cstddef>

static const std::size_t sInputLayerSize = 1;
static const std::size_t sHiddenLayerSize = 3;
static const std::size_t sOutputLayerSize = 1;

static const std::size_t sInputToHiddenWeightSize = sInputLayerSize*sHiddenLayerSize;
static const std::size_t sHiddenToOutputWeightSize = sHiddenLayerSize*sOutputLayerSize;
static const std::size_t sTotalWeightSize = sInputToHiddenWeightSize + sHiddenToOutputWeightSize;
static const std::size_t sTotalBiasesSize = sHiddenLayerSize + sOutputLayerSize;

static const std::size_t sTotalNumberOfWeightMutations = boost::static_unsigned_max< sTotalWeightSize / 20, 1 >::value;
static const std::size_t sTotalNumberOfBiasesMutations = boost::static_unsigned_max< sTotalBiasesSize / 20, 1 >::value;
