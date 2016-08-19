#pragma once

#include <iostream>
#include <string>

#include "Globals.h"

using namespace std;

struct TrainModifiers {
	double LBFGS_tol;
	int LBFGS_M;
	bool testGrad;

	TrainModifiers()
		:
	LBFGS_tol(NaN), LBFGS_M(-1), testGrad(false) { }

	static TrainModifiers LBFGSModifiers(double tol, int M, bool testGrad) {
		TrainModifiers mod;
		mod.LBFGS_tol = tol;
		mod.LBFGS_M = M;
		mod.testGrad = testGrad;

		return mod;
	}

	void Print() const {
		cout << "tol: " << LBFGS_tol << endl;
		cout << "M: " << LBFGS_M << endl;
	}
};

struct PretrainHyperParams {
	static const int numParams = 6;

	double pretrainL2i, pretrainL2h, pretrainL2o;
	double gaussianStdDevI, gaussianStdDevH;
	double layerWidthH;

	PretrainHyperParams() {
		pretrainL2i = pretrainL2h = pretrainL2o = gaussianStdDevI = gaussianStdDevH = layerWidthH = NaN;
	}

	void Serialize(ostream & stream) const {
		stream.write((const char *)this, numParams * sizeof(double));
	}

	void Deserialize(istream & stream) {
		stream.read((char *)this, numParams * sizeof(double));
	}

	void Print(const string & prefix, ostream & stream = cout) const {
		printf("%spretrainL2i: %.4e\n", prefix.c_str(), pretrainL2i);
		printf("%spretrainL2h: %.4e\n", prefix.c_str(), pretrainL2h);
		printf("%spretrainL2o: %.4e\n", prefix.c_str(), pretrainL2o);
		printf("%sgaussianStdDevI: %.4e\n", prefix.c_str(), gaussianStdDevI);
		printf("%sgaussianStdDevH: %.4e\n", prefix.c_str(), gaussianStdDevH);
		printf("%slayerWidthH: %.4e\n", prefix.c_str(), layerWidthH);
	}

	void PrintArray(ostream & stream = cout) const {
		printf("%.4e, %.4e, %.4e, %.4e, %.4e, %.4e\n", pretrainL2i, pretrainL2h, pretrainL2o, gaussianStdDevI, gaussianStdDevH, layerWidthH);
	}

	int GetLayerWidthH() const {
#ifndef NDEBUG
		return 100;
#else
		return (int)ceil(layerWidthH);
#endif
	}
};

struct DCCAHyperParams {
	PretrainHyperParams params[2];
	double backpropReg;
	double ccaReg1, ccaReg2;

	static const int numParams = PretrainHyperParams::numParams * 2 + 3;

	DCCAHyperParams() {
		backpropReg = ccaReg1 = ccaReg2 = NaN;
	}
	
	void Serialize(ostream & stream) const {
		stream.write((const char *)this, numParams * sizeof(double));
	}

	void Deserialize(istream & stream) {
		stream.read((char *)this, numParams * sizeof(double));
	}

	operator Vector() const {
		return Vector((const double *)this, numParams, 1);
	}

	operator MutableVector() {
		return MutableVector((double *)this, numParams, 1);
	}

	void Print(ostream & stream = cout) {
		cout << "View 1 hyperparams: " << endl;
		params[0].Print("   ");
		cout << "View 2 hyperparams: " << endl;
		params[1].Print("   ");
		printf("backprop decay: %.4e\n", backpropReg);
		printf("CCA reg 1: %.4e\n", ccaReg1);
		printf("CCA reg 2: %.4e\n", ccaReg2);
	}
};
