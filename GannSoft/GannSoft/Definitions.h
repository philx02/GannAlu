#pragma once

#include <boost/integer/static_min_max.hpp>

#include <cstddef>

static const std::size_t sInputLayerSize = 2;
static const std::size_t sHiddenLayerSize = 2;
static const std::size_t sOutputLayerSize = 1;

static const std::size_t sInputToHiddenWeightSize = sInputLayerSize*sHiddenLayerSize;
static const std::size_t sHiddenToOutputWeightSize = sHiddenLayerSize*sOutputLayerSize;
static const std::size_t sTotalWeightSize = sInputToHiddenWeightSize + sHiddenToOutputWeightSize;

static const std::size_t sTotalNumberOfMutation = boost::static_unsigned_max< sTotalWeightSize / 20, 1 >::value;
