#pragma once

#include <boost/integer/static_min_max.hpp>

#include <cstddef>

static const std::size_t sInputLayerSize = 6;
static const std::size_t sHiddenLayerSize = 8;
static const std::size_t sOutputLayerSize = 2;

static const std::size_t sTotalNumberOfWeightMutations = boost::static_unsigned_max< sTotalWeightSize / 20, 1 >::value;
static const std::size_t sTotalNumberOfBiasesMutations = boost::static_unsigned_max< sTotalBiasesSize / 20, 1 >::value;
