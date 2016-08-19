#pragma once

#include <vector>
#include <deque>
#include <iostream>

#include "Globals.h"
#include "Matrix.h"

struct DifferentiableFunction {
	virtual double Eval(const Vector & input, MutableVector & gradient) = 0; // kl: A virtual member is a member function that can be redefined in a derived class, while preserving its calling properties through references
	virtual ~DifferentiableFunction() { }
};

#include "TerminationCriterion.h"

class LBFGS {
	bool quiet;
	bool responsibleForTermCrit;

public:
	TerminationCriterion *termCrit;

	LBFGS(bool quiet = false) : quiet(quiet) {
		termCrit = new RelativeMeanImprovementCriterion(5);
		responsibleForTermCrit = true;
	}

	LBFGS(TerminationCriterion *termCrit, bool quiet = false) : quiet(quiet), termCrit(termCrit) { 
		responsibleForTermCrit = false;
	}

	~LBFGS() {
		if (termCrit && responsibleForTermCrit) delete termCrit;
	}

	double Minimize(DifferentiableFunction& function, const Vector & initial, MutableVector & minimum, double tol = 1e-4, int m = 10, bool testGrad = false) const;
	void SetQuiet(bool q) { quiet = q; }

	static void TestGrad(DifferentiableFunction & function, const Vector & initial);
};

class OptimizerState {
	friend class LBFGS;

	AllocatingVector x, grad, newX, newGrad, dir;
	std::deque<MutableVector> sList, yList;
	AllocatingMatrix sAndY;
	std::deque<double> roList;
	std::vector<double> alphas;
	double value;
	int iter, m;
	const size_t dim;
	DifferentiableFunction& func;
	bool quiet;

	void MapDirByInverseHessian();
	bool WolfeLineSearch();
	void Shift();
	void TestDirDeriv();
	
	struct PointValueDeriv {
		double a, v, d;
		PointValueDeriv(double a = NaN, double value = NaN, double deriv = NaN) : a(a), v(value), d(deriv) { }
	};

	static double CubicInterp(const PointValueDeriv & p0, const PointValueDeriv & p1);

	OptimizerState(DifferentiableFunction& f, const Vector & init, int m, bool quiet) 
		:
	x(init), grad(init.Len()), newX(init), newGrad(init.Len()), dir(init.Len()),
	sAndY(init.Len(), 2 * m),
	alphas(m), iter(1), m(m), dim(init.Len()), func(f), quiet(quiet) {
		if (m <= 0) {
			std::cerr << "m must be an integer greater than zero." << std::endl;
			exit(1);
		}
		value = func.Eval(newX, newGrad);
		grad = newGrad;
	}

public:
	void Reset() {
		sList.clear();
		yList.clear();
		roList.clear();
	}

	Vector GetX() const { return newX; }
	Vector GetLastX() const { return x; }
	Vector GetGrad() const { return newGrad; }
	Vector GetLastGrad() const { return grad; }
	Vector GetLastDir() const { return dir; }
	double GetValue() const { return value; }
	int GetIter() const { return iter; }
	size_t GetDim() const { return dim; }
};
