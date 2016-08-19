#include "LBFGS.h"

#include "TerminationCriterion.h"

#include <vector>
#include <deque>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <assert.h>

using namespace std;

void OptimizerState::MapDirByInverseHessian() {
	dir.ScaleInto(grad, -1);

	int count = (int)sList.size();

	if (count != 0) {
		for (int i = count - 1; i >= 0; i--) {
			alphas[i] = -Vector::DotProduct(sList[i], dir) / roList[i];
			dir.Add(yList[i], alphas[i]);
		}

		Vector lastY = yList[count - 1];
		double yDotY = Vector::DotProduct(lastY, lastY);
		double scalar = roList[count - 1] / yDotY;
		dir *= scalar;

		for (int i = 0; i < count; i++) {
			double beta = Vector::DotProduct(yList[i], dir) / roList[i];
			dir.Add(sList[i], -alphas[i] - beta);
		}
	}
}

void OptimizerState::TestDirDeriv() {
	double dirNorm = sqrt(Vector::DotProduct(dir, dir));
	double eps = 1.05e-8 / dirNorm;
	newX.CopyFrom(x);
	newX.Add(dir, eps);
	newGrad.Clear();
	double val2 = func.Eval(newX, newGrad);
	double numDeriv = (val2 - value) / eps;
	double deriv = Vector::DotProduct(dir, grad);
	if (!quiet) cout << "  Grad check: " << numDeriv << " vs. " << deriv << "  ";
}

void LBFGS::TestGrad(DifferentiableFunction & func, const Vector & x) {
	cout << setprecision(4) << scientific << right;

	AllocatingVector newX(x), grad(x.Len()), dummy(x.Len());
	func.Eval(newX, grad);

	double eps = pow(2.2e-16, 1.0 / 3);
	double maxDiff = 0;
	cout << left << setw(5) << "i" << setw(20) << "analytic" << setw(20) << "numeric" << setw(20) << "difference" << endl;
	for (int i = 0; i < x.Len(); ++i) {
		double origVal = x[i];
		newX[i] = origVal - eps;
		double lVal = func.Eval(newX, dummy);
		newX[i] = origVal + eps;
		double rVal = func.Eval(newX, dummy);
		newX[i] = origVal;

		double numDeriv = (rVal - lVal) / (2 * eps);
		double diff = fabs(numDeriv - grad[i]);
		cout << setw(5) << i << setw(20) << grad[i] << setw(20) << numDeriv << setw(20) << diff << endl;
		if (diff > maxDiff) maxDiff = diff;
	}
	cout << "MaxDiff: " << maxDiff << endl;
}

bool OptimizerState::WolfeLineSearch() {
	double dirDeriv = Vector::DotProduct(dir, grad);
	double normDir = sqrt(Vector::DotProduct(dir, dir));

	// if a non-descent direction is chosen, the line search will break anyway, so throw here
	// The most likely reasons for this is a bug in your function's gradient computation,
	if (dirDeriv >= 0) {
		cerr << "L-BFGS chose a non-descent direction: check your gradient!" << endl;
		LBFGS::TestGrad(func, x);
		abort();
	}

	double c1 = 1e-4 * dirDeriv;
	double c2 = 0.9 * dirDeriv;

	double a = (roList.size() == 0 ? (1 / normDir) : 1.0);

	PointValueDeriv last(0, value, dirDeriv);
	PointValueDeriv aLo, aHi;
	bool done = false;

	if (!quiet) cout << " ";

	double oldValue = value;

	int steps = 0;
	for (;;) {
		newX.CopyFrom(x);
		newX.Add(dir, a);

		newGrad.Clear();
		value = func.Eval(newX, newGrad);

		if (IsNaN(value)) {
			cerr << "Got NaN." << endl;
			return false;
		}

		dirDeriv = Vector::DotProduct(dir, newGrad);
		PointValueDeriv curr(a, value, dirDeriv);

		if ((curr.v > oldValue + c1 * a) || (last.a > 0 && curr.v >= last.v)) {
			aLo = last;
			aHi = curr;
			break;
		} else if (fabs(curr.d) <= -c2) {
			done = true;
			break;
		} else if (curr.d >= 0) {
			aLo = curr;
			aHi = last;
			break;
		}

		if (++steps == 10) return false;

		last = curr;
		a *= 2;
		if (!quiet) cout << "+";
	}

	double minChange = 0.01;

	// this loop is the "zoom" procedure described in Nocedal & Wright
	steps = 0;
	while (!done) {
		if (++steps == 10) return false;
		if (aLo.a == aHi.a) return false;
		if (!quiet) cout << "-";
		PointValueDeriv left = aLo.a < aHi.a ? aLo : aHi;
		PointValueDeriv right = aLo.a < aHi.a ? aHi : aLo;
		if (IsInf(left.v) || IsInf(right.v)) {
			a = (aLo.a + aHi.a) / 2;
		} else if (left.d > 0 && right.d < 0) {
			// interpolating cubic would have max in range, not min (can this happen?)
			// set a to the one with smaller value
			a = aLo.v < aHi.v ? aLo.a : aHi.a;
		} else {
			a = CubicInterp(aLo, aHi);
		}

		// this is to ensure that the new point is within bounds
		// and that the change is reasonably sized
		double ub = (minChange * left.a + (1 - minChange) * right.a);
		if (a > ub) a = ub;
		double lb = (minChange * right.a + (1 - minChange) * left.a);
		if (a < lb) a = lb;

		newX.CopyFrom(x);
		newX.Add(dir, a);

		newGrad.Clear();
		value = func.Eval(newX, newGrad);
		if (IsNaN(value)) {
			cerr << "Got NaN." << endl;
			abort();
		}

		dirDeriv = Vector::DotProduct(dir, newGrad);

		PointValueDeriv curr(a, value, dirDeriv);

		if ((curr.v > oldValue + c1 * a) || (curr.v >= aLo.v)) {
			aHi = curr;
		} else if (fabs(curr.d) <= -c2) {
			done = true;
		} else {
			if (curr.d * (aHi.a - aLo.a) >= 0) aHi = aLo;
			aLo = curr;
		}
	}

	if (!quiet) cout << endl;
	return true;
}

/// <summary>
/// Cubic interpolation routine from Nocedal and Wright
/// </summary>
/// <param name="p0">first point, with value and derivative</param>
/// <param name="p1">second point, with value and derivative</param>
/// <returns>local minimum of interpolating cubic polynomial</returns>
double OptimizerState::CubicInterp(const PointValueDeriv & p0, const PointValueDeriv & p1) {
	double t1 = p0.d + p1.d - 3 * (p0.v - p1.v) / (p0.a - p1.a);
	double sign = (p1.a > p0.a) ? 1 : -1;
	double t2 = sign * sqrt(t1 * t1 - p0.d * p1.d);
	double num = p1.d + t2 - t1;
	double denom = p1.d - p0.d + 2 * t2;
	return p1.a - (p1.a - p0.a) * num / denom;
}

void OptimizerState::Shift() {
	MutableVector nextS, nextY;

	int listSize = (int)sList.size();

	if (listSize < m) {
		nextS = sAndY.GetCol(2 * listSize);
		nextY = sAndY.GetCol(2 * listSize + 1);
	} else {
		nextS = sList.front();
		sList.pop_front();
		nextY = yList.front();
		yList.pop_front();
		roList.pop_front();
	}

	nextS.SubtractInto(newX, x);
	nextY.SubtractInto(newGrad, grad);
	double ro = Vector::DotProduct(nextS, nextY);

	sList.push_back(nextS);
	yList.push_back(nextY);
	roList.push_back(ro);

	x.Swap(newX);
	grad.Swap(newGrad);

	iter++;
}

double LBFGS::Minimize(DifferentiableFunction& function, const Vector & initial, MutableVector & minimum, double tol, int m, bool testGrad) const {
	OptimizerState state(function, initial, m, quiet);
	if (testGrad) LBFGS::TestGrad(function, initial);

	if (!quiet) {
		cout << setprecision(4) << scientific << right;
		cout << "Optimizing function of " << state.dim << " variables with L-BFGS." << endl;
		cout << "   L-BFGS memory parameter (m): " << m << endl;
		cout << "   Convergence tolerance: " << tol << endl;
		cout << endl;
		cout << "Iter    n:  new_value    (conv_crit)   line_search" << endl << flush;
		cout << "Iter    0:  " << setw(10) << state.value << "  (***********) " << flush;
	}

	ostringstream str;
	termCrit->GetValue(state, str);

	while (true) {
		state.MapDirByInverseHessian();

		if (!state.WolfeLineSearch()) {
			if (!quiet) cout << "Line search failed. Resetting state." << endl;
			state.Reset();

			if (!state.WolfeLineSearch()) {
				if (!quiet) cout << endl << "Premature convergence." << endl;
				break;
			}
		}

//		if (testGrad) LBFGS::TestGrad(function, state.newX);

		ostringstream str;
		double termCritVal = termCrit->GetValue(state, str);
		if (!quiet) {
			cout << "Iter " << setw(4) << state.iter << ":  " << setw(10) << state.value;
			cout << str.str() << flush;			
		}

		if (termCritVal < tol) break;

		state.Shift();
	}

	if (!quiet) {
		cout << fixed << endl;
	}

	if (testGrad) LBFGS::TestGrad(function, state.newX);

	minimum.CopyFrom(state.newX);
	return state.value;
}
