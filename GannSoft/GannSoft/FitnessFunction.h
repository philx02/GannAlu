#pragma once

#include <cmath>
#include <boost/math/constants/constants.hpp>
#include <boost/optional.hpp>
#include <algorithm>

#undef min

namespace bmc = boost::math::double_constants;

static const double sLocalGravitationalAcceleration = 9.81;

class LongitudinalAircraft
{
public:
  LongitudinalAircraft(double iRotationalInertia, double iElevatorToCogDistance, double iMass, double iElevatorCommandToForceOnTailFactor, double iThrottleCommandToThrustFactor, double iLiftConstant, double iDragConstant, double iPitchAngle, double iPitchAngularSpeed, double iVerticalPosition, double iVerticalSpeed, double iHorizontalPosition, double iHorizontalSpeed)
    : mRotationalInertia(iRotationalInertia)
    , mElevatorToCogDistance(iElevatorToCogDistance)
    , mMass(iMass)
    , mElevatorCommandToForceOnTailFactor(iElevatorCommandToForceOnTailFactor)
    , mThrottleCommandToThrustFactor(iThrottleCommandToThrustFactor)
    , mLiftConstant(iLiftConstant)
    , mDragConstant(iDragConstant)
  {
    initialize(iPitchAngle, iPitchAngularSpeed, iVerticalPosition, iVerticalSpeed, iHorizontalPosition, iHorizontalSpeed);
  }

  inline void initialize(double iPitchAngle, double iPitchAngularSpeed, double iVerticalPosition, double iVerticalSpeed, double iHorizontalPosition, double iHorizontalSpeed)
  {
    mPitchAngle = iPitchAngle;
    mPitchAngularSpeed = iPitchAngularSpeed;
    mVerticalPosition = iVerticalPosition;
    mVerticalSpeed = iVerticalSpeed;
    mHorizontalPosition = iHorizontalPosition;
    mHorizontalSpeed = iHorizontalSpeed;
  }

  inline double pitchAngle() const
  {
    return mPitchAngle;
  }
  inline double pitchAngularSpeed() const
  {
    return mPitchAngularSpeed;
  }

  inline double verticalPosition() const
  {
    return mVerticalPosition;
  }
  inline double verticalSpeed() const
  {
    return mVerticalSpeed;
  }

  inline double horizontalPosition() const
  {
    return mHorizontalPosition;
  }
  inline double horizontalSpeed() const
  {
    return mHorizontalSpeed;
  }

  inline double bodySpeed() const
  {
    return std::sqrt(mHorizontalSpeed * mHorizontalSpeed + mVerticalSpeed * mVerticalSpeed);
  }

  // iElevatorCommand => [-1, 1]; iThrottleCommand => [0, 1]
  inline void perform(double iElevatorCommand, double iThrottleCommand, double iDeltaT)
  {
    auto wForceOnTail = iElevatorCommand * mElevatorCommandToForceOnTailFactor;
    auto wThrust = iThrottleCommand * mThrottleCommandToThrustFactor;

    // Rotational dynamics
    auto wTorque = wForceOnTail * mElevatorToCogDistance;
    auto wPitchAngularAcceleration = wTorque / mRotationalInertia;
    mPitchAngularSpeed += wPitchAngularAcceleration * iDeltaT;
    mPitchAngle += mPitchAngularSpeed * iDeltaT;

    // Translational dynamics
    auto wBodySpeed = bodySpeed();
    // this is an approximation
    auto wLift = mLiftConstant * wBodySpeed;
    auto wDrag = mDragConstant * wBodySpeed;

    auto wTheta = mPitchAngle + bmc::half_pi;
    auto wVerticalLiftComponent = wLift * std::sin(wTheta);
    auto wHorizontalLiftComponent = wLift * std::cos(wTheta);
    auto wPsi = mPitchAngle + bmc::pi;
    auto wVerticalDragComponent = wDrag * std::sin(wPsi);
    auto wHorizontalDragComponent = wDrag * std::cos(wPsi);
    auto wVerticalThrustComponent = wThrust * std::sin(mPitchAngle);
    auto wHorizontalThrustComponent = wThrust * std::cos(mPitchAngle);

    auto wTotalVerticalForce = wVerticalLiftComponent + wVerticalDragComponent + wVerticalThrustComponent - mMass * sLocalGravitationalAcceleration;
    auto wTotalHorizontalForce = wHorizontalLiftComponent + wHorizontalDragComponent + wHorizontalThrustComponent;

    auto wVerticalAcceleration = wTotalVerticalForce / mMass;
    mVerticalSpeed += wVerticalAcceleration * iDeltaT;
    mVerticalPosition += mVerticalSpeed * iDeltaT;

    auto wHorizontalAcceleration = wTotalHorizontalForce / mMass;
    mHorizontalSpeed += wHorizontalAcceleration * iDeltaT;
    mHorizontalPosition += mHorizontalSpeed * iDeltaT;
  }

private:
  double mPitchAngle;
  double mPitchAngularSpeed;

  double mVerticalPosition;
  double mVerticalSpeed;

  double mHorizontalPosition;
  double mHorizontalSpeed;

  // aircraft performance data
  double mRotationalInertia;
  double mElevatorToCogDistance;
  double mMass;
  double mElevatorCommandToForceOnTailFactor;
  double mThrottleCommandToThrustFactor;
  double mLiftConstant;
  double mDragConstant;
};

static const double sRotationalInertia = 0.1;
static const double sElevatorToCogDistance = 1;
static const double sMass = 1;
static const double sElevatorCommandToForceOnTailFactor = 1;
static const double sThrottleCommandToThrustFactor = 10;
static const double sLiftConstant = 1.22625;
static const double sDragConstant = 1.22625/7;

static const auto sEquilibriumHorizontalSpeed = (sMass * sLocalGravitationalAcceleration) / sLiftConstant;

// This function must be thread safe
template< typename Ann, typename EndOfIterationFunction >
double fitnessFunction(const Ann &iAnn, EndOfIterationFunction &iEndOfIterationFunction)
{
  Ann::Input wInput;
  double wScore = 0;
  double wTargetAltitude = 100;
  double wTargetSpeed = 8;
  LongitudinalAircraft wLongitudinalAircraft(sRotationalInertia, sElevatorToCogDistance, sMass, sElevatorCommandToForceOnTailFactor, sThrottleCommandToThrustFactor, sLiftConstant, sDragConstant, 0, 0, 50, 0, 0, sEquilibriumHorizontalSpeed);
  double wDeltaT = 0.1;
  for (double wTime = 0; wTime < 30; wTime += wDeltaT)
  {
    wInput[0] = wLongitudinalAircraft.pitchAngle();
    wInput[1] = wLongitudinalAircraft.pitchAngularSpeed();
    wInput[2] = wLongitudinalAircraft.verticalPosition();
    wInput[3] = wLongitudinalAircraft.verticalSpeed();
    wInput[4] = wLongitudinalAircraft.horizontalPosition();
    wInput[5] = wLongitudinalAircraft.horizontalSpeed();

    auto wOutput = iAnn.compute(wInput);
    wLongitudinalAircraft.perform(wOutput[0]*2-1, wOutput[1], wDeltaT);
    wScore -= std::abs(wLongitudinalAircraft.verticalPosition() - wTargetAltitude);
    wScore -= std::abs(wLongitudinalAircraft.bodySpeed() - wTargetSpeed) * (50/8);
    auto wPitchError = std::abs(wLongitudinalAircraft.pitchAngle());
    wScore -= std::min(0.25*std::exp(7.0*wPitchError), 200.0);
    iEndOfIterationFunction(wTime, wOutput, wLongitudinalAircraft);
  }
  return wScore;
}
