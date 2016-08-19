#include <assert.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string.h>

#include "Matrix.h"

#include "mkl.h"
#include "mkl_lapacke.h"
#include "mkl_spblas.h"
#include "mkl_trans.h"
const long long MKL_MODE = VML_LA;

using namespace std;

void Vector::Print() const {
	const double * p = Start();
	while (p != End()) {
		cout << *(p++) << endl;
	}
}

void Vector::operator++() {
	if (_inc == 1) _start += _len;
	else ++_start;
}

void Vector::operator--() {
	if (_inc == 1) _start -= _len;
	else --_start;
}

double Vector::LogSum() const {
	const double * p = Start();
	int inc = Inc();

	int argMax = ArgMax();
	double max = operator[](argMax);

	double sumExp = 0.0;
	p = Start();
	for (int i = 0; i < argMax; ++i) {
		double val = *p - max;
		if (val > -30) sumExp += exp(val);
		p += inc;
	}

	for (int i = argMax + 1; i < _len; ++i) {
		p += inc;
		double val = *p - max;
		if (val > -30) sumExp += exp(val);
	}

	return max + log(1.0 + sumExp);
}

double Vector::Norm() const {
	return cblas_dnrm2(_len, Start(), Inc());
}

double Vector::MaxAbs() const {
	int idx = cblas_idamax(_len, Start(), Inc());
	return operator[](idx);
}

int Vector::ArgMax() const {
	const double * p = Start();
	int inc = Inc();

	int argMax = 0;
	double max = *p;
	for (int i = 1; i < _len; ++i) {
		p += inc;
		if (*p > max) {
			argMax = i;
			max = *p;
		}
	}

	return argMax;
}

double Vector::Sum() const {
	double result = 0;
	ApplyConst([&](double x) { result += x; });
	return result;
}

double Vector::DotProduct(const Vector & a, const Vector & b) {
	assert (a.Len() == b.Len());
	
	return cblas_ddot(a.Len(), a.Start(), a.Inc(), b.Start(), b.Inc());
}

Matrix Vector::AsMatrix(int numR, int numC) const {
	return Matrix(_start, numR, numC);
}

const Vector Vector::EmptyVector(0, 0, 0);

void MutableVector::MultInto(const Vector & a, const Vector & b) {
	assert (a.Len() == Len() && b.Len() == Len());
	
	if (a.Inc() == 1 && b.Inc() == 1 && Inc() == 1) {
		vdMul(_len, a.Start(), b.Start(), Start());
	} else {
		ApplyInto(a, b, [](double x, double y) { return x * y; });
	}
}

void MutableVector::DivideInto(const Vector & a, const Vector & b) {
	assert (a.Len() == Len() && b.Len() == Len());

	if (a.Inc() == 1 && b.Inc() == 1 && Inc() == 1) {
		vmdDiv(_len, a.Start(), b.Start(), Start(), MKL_MODE);
	} else {
		ApplyInto(a, b, [](double x, double y) { return x / y; });
	}
}

void MutableVector::MultInto(const Matrix & m, bool transM, const Vector & a, double alpha, double beta) {
	int mNumC = transM ? m.NumR() : m.NumC();
	int mNumR = transM ? m.NumC() : m.NumR();
	assert (a.Len() == mNumC && mNumR == Len());

    //kl: Multiplies a matrix by a vector (double precision). The vector will muliply matrix m, and store the result in the vector itself.
	cblas_dgemv(CblasColMajor, transM ? CblasTrans : CblasNoTrans, m.NumR(), m.NumC(), alpha, m.Start(), m.NumR(), a.Start(), a.Inc(), beta, Start(), Inc());
}
	
void MutableVector::AddInto(const Vector & a, const Vector & b) {
	assert (a.Len() == Len() && b.Len() == Len());

	if (a.Inc() == 1 && b.Inc() == 1 && Inc() == 1) {
		vdAdd(_len, a.Start(), b.Start(), Start());
	} else {
		ApplyInto(a, b, [](double x, double y) { return x + y; });
	}
}

double MutableVector::LogSumFastDestroyMe() {
	double max = Max();

	double * s = Start();
	double * p = s;
	double * e = End();
	double * tP = s;
	int inc = Inc();
	while (p != e) {
		double val = *p - max;
		if (val > -30) *tP++ = val;
		p += inc;
	}
	
	int nT = tP - s;
	vmdExp(nT, s, s, VML_EP);

	double sumExp = cblas_dasum(nT, s, 1);

	return max + log(sumExp);
}

void MutableVector::AddMult(const Vector & a, const Vector & b) {
	assert (a.Len() == Len() && b.Len() == Len());

	const double * pA = a.Start();
	const double * pB = b.Start();
	double *	p = Start();
	double const * end = End();
	
	int incA = a.Inc();
	int incB = b.Inc();
	int inc = Inc();

	while (p != end) {
		*p += (*pA) * (*pB);
		p += inc, pA += incA, pB += incB;
	}	
}

void MutableVector::SubtractInto(const Vector & a, const Vector & b) {
	assert (a.Len() == Len() && b.Len() == Len());
	
	if (a.Inc() == 1 && b.Inc() == 1 && Inc() == 1) {
		vdSub(_len, a.Start(), b.Start(), Start());
	} else {
		ApplyInto(a, b, [](double x, double y) { return x - y; });
	}
}

void MutableVector::Add(const Vector & a, double v) {
	assert (a.Len() == Len());

	cblas_daxpy(_len, v, a.Start(), a.Inc(), Start(), Inc());
}

void MutableVector::operator+=(const Vector & a) {
	// had vdAdd here, but it was much much slower! (?)
	Add(a, 1.0);
}

void MutableVector::operator*=(const Vector & a) {
	assert (a.Len() == Len());
	
	if (a.Inc() == 1 && Inc() == 1) {
		vdMul(_len, a.Start(), Start(), Start());
	} else {
		auto mut = [](double x, double y) { return x * y; };
		ApplyIntoRef(a, mut);
	}
}

void MutableVector::operator/=(const Vector & a) {
	DivideInto(*this, a);
}

void MutableVector::operator-=(const Vector & a) {
	Add(a, -1.0);
}

void MutableVector::Tanh() {
	vmdTanh(_len, Start(), Start(), MKL_MODE);
}

void MutableVector::LogSumWithNeg(MutableVector & temp) {
	assert (temp.Len() == Len());
	
	if (temp.Inc() == 1 && Inc() == 1) {
		vdAbs(_len, Start(), Start());
		cblas_daxpby(_len, -2.0, Start(), 1, 0, temp.Start(), 1);
		vmdExp(_len, temp.Start(), temp.Start(), MKL_MODE);
		vmdLog1p(_len, temp.Start(), temp.Start(), MKL_MODE); // note: might be faster to increment and log!
		vdAdd(_len, temp.Start(), Start(), Start());
	} else {
		auto mut = [](double x)->double {
			double absX = fabs(x);
			return absX + log(1.0 + exp(-2 * absX));
		};
		Apply(mut);
	}
}

void MutableVector::Exp() {
	return ExpInto(*this);
}

void MutableVector::ExpInto(const Vector & a) {
	assert (a.Len() == Len());
	
	if (a.Inc() == 1 && Inc() == 1) {
		vmdExp(_len, a.Start(), Start(), MKL_MODE);
	} else {
		ApplyInto(a, [](double x) { return exp(x); });
	}
}

void MutableVector::Log() {
	return LogInto(*this);
}

void MutableVector::LogInto(const Vector & a) {
	assert (a.Len() == Len());
	
	if (a.Inc() == 1 && Inc() == 1) {
		vmdLn(_len, a.Start(), Start(), MKL_MODE);
	} else {
		ApplyInto(a, [](double x) { return log(x); });
	}
}

void MutableVector::TanhInto(const Vector & a) {
	assert (a.Len() == Len());
	
	if (a.Inc() == 1 && Inc() == 1) {
		vmdTanh(_len, a.Start(), Start(), MKL_MODE);
	} else {
		ApplyInto(a, [](double x) { return tanh(x); });
	}
}

void MutableVector::SquareInto(const Vector & a) {
	assert (a.Len() == Len());
	
	if (a.Inc() == 1 && Inc() == 1) {
		vdSqr(_len, a.Start(), Start());
	} else {
		ApplyInto(a, [](double x) { return x * x; });
	}
}

void MutableVector::ScaleInto(const Vector & a, double d) {
	Axpby(a, d, 0);
}

void MutableVector::Axpby(const Vector & x, double a, double b) {
	assert (x.Len() == Len());

	cblas_daxpby(_len, a, x.Start(), x.Inc(), b, Start(), Inc());
}

void MutableVector::operator*=(double d) {
	cblas_dscal(_len, d, Start(), Inc());
}

void MutableVector::Assign(double d) {
  Apply([d](double x) { return d; });
}

void MutableVector::InvNormCDF() {
	vmdCdfNormInv(Len(), Start(), Start(), MKL_MODE);
}

void MutableVector::CopyFrom(const Vector & a) {
	assert(_len == a.Len());
	if (&a == this) return;
	
	cblas_dcopy(_len, a.Start(), a.Inc(), Start(), Inc());
}

void MutableVector::Sqrt() {
	vmdSqrt(_len, Start(), Start(), MKL_MODE);
}

void MutableVector::Shrink(double d) {
  Apply([d](double x) { return (x>d) ? x-d : (x<-d) ? x+d : 0; });
}

void MutableVector::Trunc(double d) {
  Apply([d](double x) { return x > d ? d : (x < -d ? -d : x); });
}

Matrix MutableVector::AsMatrix(int numR, int numC) const {
	return Matrix(_start, numR, numC);
}

MutableMatrix MutableVector::AsMatrix(int numR, int numC) {
	return MutableMatrix(Start(), numR, numC);
}

const Matrix Matrix::EmptyMatrix(0, 0, 0);

void Matrix::SolvePSDSystem(MutableMatrix & A, MutableVector & b) {
	LAPACKE_dposv(LAPACK_COL_MAJOR, 'u', A._numC, 1, A.Start(), A.NumR(), b.Start(), b.Len());
}

Vector Matrix::GetRow(int r) const {
	if (r < 0) r += NumR();
	assert (r >= 0 && r < NumR());
	
	return Vector(_start + r, _numC, _numR);
}

void Matrix::Print() const {
	cout << fixed << left;
	cout << "Matrix: " << NumR() << "x" << _numC << endl;
	for (int r = 0; r < _numR; ++r) {
		cout << setw(10) << At(r,0);
		for (int c = 1; c < _numC; ++c) {
			cout << setw(10) << At(r,c);
		}
		cout << endl;
	}
}

Vector Matrix::GetCol(int c) const {
	if (c < 0) c += _numC;
	assert (c >= 0 && c < _numC);

	return Vector(_start + c * _numR, NumR(), 1);
}

Vector MutableMatrix::GetCol(int c) const { return Matrix::GetCol(c); }

Vector MutableMatrix::GetRow(int r) const { return Matrix::GetRow(r); }

MutableVector MutableMatrix::GetCol(int c) {
	if (c < 0) c += _numC;
	assert (c >= 0 && c < _numC);

	return MutableVector(Start() + c * _numR, NumR(), 1);
}

MutableVector MutableMatrix::GetRow(int r) {
	if (r < 0) r += NumR();
	assert (r >= 0 && r < NumR());
	
	return MutableVector(Start() + r, _numC, _numR);
}

void MutableMatrix::CopyFrom(const Matrix & other) { //kl: copy a matrix
	if (&other == this) return;

	assert (_numR == other.NumR() && _numC == other.NumC());
    
	//kl: Copies a vector to another vector (double-precision).
	cblas_dcopy(_numR * _numC, other.Start(), 1, Start(), 1);
}

void MutableMatrix::EigenDecompose(MutableVector & eigenvals) {
	assert (NumR() == _numC);

	int info = LAPACKE_dsyevd(CblasColMajor, 'V', 'U', NumR(), Start(), NumR(), eigenvals.Start());

	assert (info == 0);
}
	
void MutableMatrix::AddTranspose(const Matrix & a, double scaleA, double scaleSelf) {
	assert (NumR() == a.NumC() && _numC == a.NumR());

	char ordering = 'c';
	char transa = 't';
	char transb = 'f';

	mkl_domatadd(ordering, transa, transb, NumR(), _numC, scaleA, a.Start(), a.NumR(), scaleSelf, Start(), NumR(), Start(), NumR());
}

void MutableMatrix::TransposeInPlace(double scale) {
	mkl_dimatcopy('c', 't', _numR, _numC, scale, Start(), _numR, _numC);
	swap(_numR, _numC);
}

void MutableMatrix::CopyTransposed(const Matrix & a) {
	assert (NumR() == a.NumC() && _numC == a.NumR());

	char ordering = 'c';
	char trans = 't';
	mkl_domatcopy(ordering, trans, a.NumR(), a.NumC(), 1.0, a.Start(), a.NumR(), Start(), NumR());
}

//kl: Matrix←α*a*b + β*Matrix. first muliply two matrix and then add to itself
void MutableMatrix::AddProd(const Matrix & a, bool aTrans, const Matrix & b, bool bTrans, double alpha, double beta) {
	int k;
	if (aTrans) { k = a.NumR(); assert(_numR == a.NumC()); }
	else { k = a.NumC(); assert (_numR == a.NumR()); }
	if (bTrans) { assert(b.NumC() == k && b.NumR() == _numC); }
	else { assert(b.NumR() == k && b.NumC() == _numC); }

	if (_len == 0) return;
    //kl: Multiplies two matrices (double-precision).Matrix←α*a*b + β*Matrix
	cblas_dgemm(CblasColMajor, (CBLAS_TRANSPOSE)(CblasNoTrans + aTrans), (CBLAS_TRANSPOSE)(CblasNoTrans + bTrans), _numR, _numC, k,
		alpha, a.Start(), a.NumR(), b.Start(), b.NumR(), beta, Start(), _numR);
}

//kl: the matrix who call this function, getting matrix = mult*b*c' + matrix
void MutableMatrix::RankOneUpdate(const Vector & b, const Vector & c, double mult) {
	assert (b.Len() == _numR && c.Len() == _numC);
	if (_numR == 0 || _numC == 0) return;

    //kl: Multiplies vector b by the transform of vector c, then adds matrix itself (double precison).
     // kl: A = alpha*x*y' + A
	cblas_dger(CblasColMajor, _numR, _numC, mult, b.Start(), b.Inc(), c.Start(), c.Inc(), Start(), _numR);
}

void MutableMatrix::SymmetricRankOneUpdate(const Vector & b, double mult) {
	assert (b.Len() == _numR && _numR == _numC);

    //kl: Calculates A + mult*b*bT and stores the result in A.
	cblas_dsyr(CblasColMajor, CblasUpper, _numR, mult, b.Start(), b.Inc(), Start(), _numR);
}

//kl: Matrix = alpha*other*ohterT + beta*Matrix
void MutableMatrix::Syrk(const Matrix & other, double alpha, double beta) {
	assert (other.NumR() == _numR && _numR == _numC);
    //kl: Rank-k update—multiplies a symmetric matrix by its transpose and adds a second matrix (double precision).
    //kl: Matrix = alpha*other*ohterT + beta*Matrix
	cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans,
		_numR, other.NumC(), alpha, other.Start(),
		_numR, beta, Start(), _numR);
}

void AllocatingMatrix::push_back(const Vector & col) {
	assert (col.Len() == _numR);
	Resize(_numR, _numC + 1);

	memcpy(Start() + (_numR * (_numC - 1)), col.Start(), _numR * sizeof(double));
}

void SolveLeastSquares(const Matrix & X, const Matrix & Winv, const Vector & y, vector<double> & temp, MutableVector & result) {
	int d = X.NumC();
	int n = X.NumR();
	assert(Winv.NumC() == n && Winv.NumR() == n && y.Len() == n);

	int minusOne = -1;
	int info;

	if (temp.size() < 1) temp.resize(1);

	// workspace query
	dggglm(&n, &d, &n, const_cast<double *>(X.Start()), &n, const_cast<double *>(Winv.Start()), &n, const_cast<double *>(y.Start()),
		result.Start(), &temp[0], &temp[0], &minusOne, &info);

	int workSize = temp[0];
	temp.resize(n + workSize);
		
	dggglm(&n, &d, &n, const_cast<double *>(X.Start()), &n, const_cast<double *>(Winv.Start()), &n, const_cast<double *>(y.Start()),
		result.Start(), &temp[0], &temp[n], &workSize, &info);
}

void Whiten(const Matrix & images, AllocatingMatrix & whiteningMat, double eps) {
	int numPixels = images.NumR(), numImages = images.NumC();
	AllocatingMatrix cov(images.NumR(), images.NumR());
	cov.Syrk(images, 1.0 / (numImages - 1), 0.0);

	AllocatingVector eigs(numPixels);
	cov.EigenDecompose(eigs);

	int numFeatures = 0;
	for (int f = 0; f < numPixels; ++f) {
		if (eigs[f] >= eps) {
			eigs[numFeatures] = eigs[f];
			cov.GetCol(numFeatures++).ScaleInto(cov.GetCol(f), 1.0 / sqrt(eigs[f]));
		}
	}
	cout << "discarding " << (numPixels - numFeatures) << " dimensions." << endl;

	whiteningMat.Resize(numFeatures, numPixels);
	whiteningMat.CopyTransposed(cov.SubMatrix(0, numFeatures));
}
