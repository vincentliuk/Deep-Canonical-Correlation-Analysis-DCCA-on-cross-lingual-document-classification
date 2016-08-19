#pragma once

#include <assert.h>
#include <vector>

#include "Matrix.h"
#include "Random.h"

using namespace std;

class Layer { //kl: this class is just for one layer, input, weights, output, not for multiple layers
public:
	enum ActivationType { TANH, CUBIC, LINEAR };

private:
	AllocatingMatrix _a; // kl: layer output for all instances, so it's a matrix
	ActivationType _actType; //kl: which activation to use for calculating the outputs
	
    //kl: just compute the multiplication of inputs and weights in this one layer
	void ComputeInputs(const Matrix & weights, const Vector & biases, const Matrix & values, bool trans) {
		int numIns = values.NumC(); //kl: number of instances
		int size = trans ? weights.NumC() : weights.NumR(); //kl: the size of layer output
		_a.Resize(size, numIns);

		_a.AddProd(weights, trans, values, false, 1.0, 0.0);
		for (int i = 0; i < numIns; ++i) _a.GetCol(i) += biases; //kl: for each instance, add the same bias vector to the result. 
	}

	static double MySigmoid(double y) {
		bool negate = false;
		if (y < 0) { negate = true; y = -y; }
		double x = (y <= 20) ? y : pow(3 * y, 1.0/3);

		double newX;
		while (true) {
			double xSqr = x * x;
			newX = (0.66666666666666666 * xSqr * x + y) / (xSqr + 1.0);
			if (newX >= x) break;
			x = newX;
		}

		return negate ? -newX : newX;
	}

    // kl: decide which activation function to use for generating the layer output
	void ComputeActivations(ActivationType actType) {
		_actType = actType;
		switch (actType) {
		case TANH: _a.Tanh(); break;
		case CUBIC:
#ifdef __LINUX
			{
				_a.Apply(MySigmoid);
			}
			break;
#else
			_a.ModCubeRootSigmoid8SSE(); break;
#endif

		case LINEAR: break;
		default: abort();
		}
	}

public:
	Layer() { }

    //kl: activateup, calculating the layer outputs from the inputs
	const Matrix & ActivateUp(const Matrix & weights, const Vector & biases, const Matrix & lowerValues, ActivationType actType) {
		ComputeInputs(weights, biases, lowerValues, true); //kl: inputs multiplied weights
		ComputeActivations(actType); //kl: outputs = activationFunc(inputs X weights) for all instances, so in a matrix
		return _a;
	}
	
    //kl: vector matrix weights case, but not sure when to use ...
	const Matrix & ActivateUp(const vector<Matrix> & weights, const Vector & biases, const vector<Matrix> & lowerValues, ActivationType actType) {
		int numIns = lowerValues[0].NumC();
		int size = weights[0].NumC();
		_a.Resize(size, numIns);

		for (int i = 0; i < numIns; ++i) _a.GetCol(i).CopyFrom(biases);
		for (int l = 0; l < weights.size(); ++l) {
			_a.AddProd(weights[l], true, lowerValues[l], false);
		}

		ComputeActivations(actType);
		return _a;
	}
	
    //kl: calculate the free energy for this layer
	double ActivateUpAndGetNegFreeEnergy(const Matrix & weights, const Vector & biases, const Matrix & lowerValues) {
		ComputeInputs(weights, biases, lowerValues, true);

		double energy = 0;
		auto func = [&](double aVal)->double {
			if (aVal < -14) {
				energy += aVal;
				return -1;
			} else if (aVal > 14) {
				energy += aVal;
				return 1;
			} else {
				double e = exp(aVal);
				double er = 1.0 / e;
				energy += log (e + er);
				return (e - er) / (e + er);
			}
		};
		_a.Apply(func);

		return energy;
	}
	
	double ActivateDownAndGetNegLL(const Matrix & weights, const Vector & biases, const Matrix & upperValues, const Matrix & lowerValues) {
		ComputeInputs(weights, biases, upperValues, false);

		double negll = 0;
		auto func = [&](double aVal, double xVal)->double {
			if (aVal < -14) {
				double xP = (1 + xVal) / 2;
				negll -= 2 * xP * aVal;
				return -1;
			} else if (aVal > 14) {
				double xP = (1 + xVal) / 2;
				negll += 2 * (1 - xP) * aVal;
				return 1;
			} else {
				double a = tanh(aVal);
				double p = (1 + a) / 2;
				double xP = (1 + xVal) / 2;
				negll -= xP * log(p) + (1 - xP) * log (1.0 - p);
				return a;
			}
		};
		_a.ApplyIntoRef(lowerValues, func);

		return negll;
	}
	
    //kl: activate Down, 把网络倒过来看并计算，但并不是backpropagation, 也不是重建input
	const Matrix & ActivateDown(const Matrix & weights, const Vector & biases, const Matrix & upperValues, ActivationType actType) {
		ComputeInputs(weights, biases, upperValues, false);
		ComputeActivations(actType);
		return _a;
	}

    //kl: from upperErrors to get the lowerInErrors
	static void BackProp(const Matrix & weights, const Matrix & upperErrors, AllocatingMatrix & lowerInErrors) {
		lowerInErrors.Resize(weights.NumR(), upperErrors.NumC());
		lowerInErrors.AddProd(weights, false, upperErrors, false, 1.0, 0.0);
	}

    //kl: for two views
	static void BackProp(const Matrix & weights, const Matrix & upperErrors, const Matrix & weights2, const Matrix & upperErrors2, AllocatingMatrix & lowerInErrors) {
		lowerInErrors.Resize(weights.NumR(), upperErrors.NumC());
		lowerInErrors.AddProd(weights, false, upperErrors, false, 1.0, 0.0);
		lowerInErrors.AddProd(weights2, false, upperErrors2, false);
	}

    //kl: from lowerErrors to get the upperInErrors ?
	static void ReverseBackProp(const Matrix & weights, const Matrix & lowerErrors, AllocatingMatrix & upperInErrors) {
		upperInErrors.Resize(weights.NumC(), lowerErrors.NumC());
		upperInErrors.AddProd(weights, true, lowerErrors, false, 1.0, 0.0);
	}

    //kl: return an inference to _a, and can't be changed, why not just return _a, because it takes very long time,and also risky to be changed.
     //kl: 
	const Matrix & ComputeErrors(const Matrix & inError) {
		switch (_actType) {
		case TANH:
			{
				auto func = [](double aVal, double eVal) { return (1.0 - aVal * aVal) * eVal; };
				_a.ApplyIntoRef(inError, func);
			}
			break;
		case CUBIC:
			{
				auto func = [](double aVal, double eVal) { return eVal / (1.0 + aVal * aVal); };
				_a.ApplyIntoRef(inError, func);
			}
			break;
		case LINEAR: _a.CopyFrom(inError); break;
		default: abort();
		}

		return _a;
	}

	MutableMatrix & Activations() { return _a; }

	int Size() const { return _a.NumR(); }

	int Count() const { return _a.NumC(); }
	
    //kl: sample from _a, the value of aVal is the element value of matrix _a, and return a matrix with vaules 1 or -1 stored in "sample"
	void Sample(MutableMatrix & sample, Random & rand) const {
		auto func = [&](double aVal) { return (2 * rand.Uniform() - 1 < aVal ? 1.0 : -1.0); };
		sample.ApplyInto(_a, func);
	}

    //kl: add a Gaussian value to _a
	void SampleGaussian(MutableMatrix & sample, double stdDev, Random & rand) const {
		auto func = [&](double aVal) { return aVal + stdDev * rand.Normal(); };
		sample.ApplyInto(_a, func);
	}
	
	void Clear() {
		_a.Resize(0, 0);
	}
};
