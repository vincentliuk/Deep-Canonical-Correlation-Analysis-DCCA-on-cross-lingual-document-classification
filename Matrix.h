#pragma once

#include "Random.h"
#include <vector>
#include <iostream>
#include <string>

#include "Globals.h"


class Matrix;
class MutableMatrix;
class MutableVector;

class Vector {

protected:
	const double* _start;
	int _len;
	int _inc;

	template <class EltType>
	class Iterator {
	protected:
		EltType * p;
		int inc; //kl: increment, usually = 1

	public:
		Iterator(EltType * p, int inc) : p(p), inc(inc) { }
		Iterator & operator++() { p += inc; return *this; }
		Iterator operator++(int) { Iterator ret(*this); ++(*this); return ret; }
		EltType & operator*() const { return *p; }

		template <class OtherEltType>
		bool operator==(const Iterator<OtherEltType> & other) { return p == other.p; }

		template <class OtherEltType>
		bool operator!=(const Iterator<OtherEltType> & other) { return p != other.p; }

		template <class OtherEltType>
		int operator-(const Iterator<OtherEltType> &other) { return (p - other.p) / inc; }
	};

public:
	typedef Matrix MatType;

	const static Vector EmptyVector;

	Vector(const double * start, int len, int inc)
		:
	_start(start),
	_len(len),
	_inc(inc) { }

	Vector()
		:
	_start(0),
	_len(0),
	_inc(1) { }

	Vector(const std::vector<double> & v)
		:
	_start(v.size() == 0 ? 0 : &v[0]),
	_len(v.size()),
	_inc(1) { }

	Vector(const Vector & other) 
		:
	_start(other._start),
	_len(other._len),
	_inc(other._inc)
	{ }

	const Vector & operator=(const Vector & other) {
		_start = other._start;
		_len = other._len;
		_inc = other._inc;
		return *this;
	}
	
	void Serialize(std::ostream & outStream) const {
		outStream.write((const char *) &_len, sizeof(int));
		// don't serialize _inc because on deserialization _inc will always be 1
		outStream.write((const char *) _start, _len * sizeof(double));
	}

	typedef Iterator<const double> const_iterator;

	const_iterator cbegin() const { return const_iterator(Start(), _inc); }
	const_iterator cend() const { return const_iterator(End(), _inc); }

	static double DotProduct(const Vector & a, const Vector & b);

	const double* Start() const { return _start; }

	int Len() const { return _len; }

	const double* End() const { return _start + _len * _inc; }

	int Inc() const { return _inc; }

	double operator[](int i) const {
		if (i < 0) i += _len;
		return *(_start + i * _inc);
	}

	double Sum() const;

	void Print() const;

	void operator++();

	void operator--();

	int ArgMax() const;

	double Norm() const;
	
	double LogSum() const;
	
	template <class Visitor>
	void ApplyConst(Visitor visit) const {
		const double *	p = Start();
		const double * const end = End();

		int inc = Inc();

		while (p != end) {
			visit(*p);
			p += inc;
		}
	}

	template <class Visitor>
	void ApplyConstWith(const Vector & a, Visitor visit) const {
		assert (_len == a.Len());

		const double *	p = Start();
		const double * pA = a.Start();
		const double * const end = End();

		int inc = Inc();
		int aInc = a.Inc();

		while (p != end) {
			visit(*p, *pA);
			p += inc;
			pA += aInc;
		}
	}

	bool AllSafe() const {
		bool allSafe = true;
		ApplyConst([&](double x) { allSafe &= !IsDangerous(x); });
		return allSafe;
	}

	bool AlmostEquals(const Vector & a) const {
		bool equal = true;
		ApplyConstWith(a, [&](double x, double y) { equal &= IsClose(x, y); });
		return equal;
	}

	double MaxAbs() const;

	double Max() const {
		double max = -INFTY;
		ApplyConst([&](double x) { if (x > max) max = x; });
		return max;
	}

	Vector SubVector(int start, int end) const {
		if (end < 0) end = _len - ~end;
		return Vector(_start + (start * _inc), end - start, _inc);
	}

	Matrix AsMatrix(int numR, int numC) const;

	int Sample(Random & rand, double sum = NaN) const {
		if (IsNaN(sum)) sum = Sum();

		for (;;) {
			double r = rand.Uniform() * sum;
			Vector::const_iterator it = cbegin(), end = cend();
			int i = 0;
			while (it != end) {
				r -= *it;
				if (r < 0) return i;
				++i, ++it;
			}
			assert (IsClose(Sum(), sum));
		}
	}

	void WriteToFile(const string & filename) {
		ofstream outputStream(filename, ios::out|ios::binary);	
		if (!outputStream.is_open()) {
			cout << "Couldn't open feature file " << filename.c_str() << endl;
			exit(1);
		}

		outputStream.write((char*)Start(), Len() * sizeof(double));
		outputStream.close();
	}
};

class MutableVector : virtual public Vector {
	friend class Matrix;
	friend class MutableMatrix;
	
public:
	typedef MutableMatrix MatType;

	MutableVector(double * start, int len, int inc)
		:
	Vector(start, len, inc) { }

	MutableVector()
		:
	Vector() { }

	MutableVector(std::vector<double> & vec)
		:
	Vector(vec) { }
	
	MutableVector & operator=(const MutableVector & other) {
		_start = other._start;
		_len = other._len;
		_inc = other._inc;
		return *this;
	}

	typedef Iterator<double> iterator;

	iterator begin() { return iterator(Start(), _inc); }
	const_iterator cbegin() const { return Vector::cbegin(); }
	iterator end() { return iterator(End(), _inc); }
	iterator rend() { return iterator(End() - _inc, -_inc); }
	
	void MultInto(const Vector & a, const Vector & b);
	
	void MultInto(const Matrix & m, bool transM, const Vector & a, double alpha = 1.0, double beta = 0.0);
	
	void AddInto(const Vector & a, const Vector & b);
		
	void LogSumWithNeg(MutableVector & temp);
	
	double LogSumFastDestroyMe();
	
	void AddMult(const Vector & a, const Vector & b);
	
	void SubtractInto(const Vector & a, const Vector & b);
	
	void DivideInto(const Vector & a, const Vector & b);
	
	void operator+=(const Vector & a);
	
	void Add(const Vector & a, double v);
	
	void operator*=(const Vector & a);

	void operator/=(const Vector & a);

	void operator-=(const Vector & a);
	
	void operator*=(double d);
	
	void operator/=(double d) {
		return operator*=(1.0 / d);
	}

	void operator+=(double d) {
		Apply([d](double x) { return x + d; });
	}

	void operator-=(double d) {
		operator+=(-d);
	}
	
	void Exp();

	void ExpInto(const Vector & a);

	void Log();

	void LogInto(const Vector & a);

	void InvNormCDF();
	
	void Tanh();
				
	void TanhInto(const Vector & a);

	void SquareInto(const Vector & a);

	void Sqrt();
	
	void ScaleInto(const Vector & a, double d);

	void Axpby(const Vector & x, double a, double b);

	virtual void CopyFrom(const Vector & a);

	void Assign(double val);

	void Shrink(double d);

	void Trunc(double max);

	double * Start() { return const_cast<double*>(_start); }

	double * End() { return const_cast<double*>(Vector::End()); }

	const double * Start() const { return Vector::Start(); }

	const double * End() const { return Vector::End(); }
	
	double & operator[](int i) { 
		if (i < 0) i += _len;
		return *(Start() + i * _inc);
	}

	double operator[](int i) const { return Vector::operator[](i); }
	
	void Clear() { Assign(0); }
	
	template <class Mutator>
	void Apply(Mutator mutator) {	
		double *p = Start();
		double const *end = End();

		int inc = Inc();

		while (p != end) {
			*p = mutator(*p);
			p += inc;
		}
	}
	
	template <class Mutator>
	void ApplyInto(const Vector & a, Mutator mutator) {	
		assert (_len == a.Len());

		double *	p = Start();
		const double * pA = a.Start();
		double const * end = End();

		int inc = Inc();
		int incA = a.Inc();

		while (p != end) {
			*p = mutator(*pA);
			p += inc, pA += incA;
		}
	}
	
	template <class Mutator>
	void ApplyIntoRef(const Vector & a, Mutator & mutator) {	
		assert (_len == a.Len());

		double *	p = Start();
		const double * pA = a.Start();
		double const * end = End();

		int inc = Inc();
		int incA = a.Inc();

		while (p != end) {
			*p = mutator(*p, *pA);
			p += inc, pA += incA;
		}
	}
	
	template <class Mutator>
	void ApplyInto(const Vector & a, const Vector & b, Mutator mutator) {	
		assert (_len == a.Len() && _len == b.Len());

		double *	p = Start();
		const double * pA = a.Start();
		const double * pB = b.Start();
		double const * end = End();

		int inc = Inc();
		int incA = a.Inc();
		int incB = b.Inc();

		while (p != end) {
			*p = mutator(*pA, *pB);
			p += inc, pA += incA, pB += incB;
		}
	}
	
	MutableVector SubVector(int start, int end) {
		if (end < 0) end = _len - ~end;
		return MutableVector(Start() + (start * _inc), end - start, _inc);
	}

	Vector SubVector(int start, int end) const {
		return Vector::SubVector(start, end);
	}
	
	Matrix AsMatrix(int numR, int numC) const;

	MutableMatrix AsMatrix(int numR, int numC);
	
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

#ifndef __LINUX	
	void ModCubeRootSigmoid8SSE() {
		ASSERT(_inc == 1);
		double *p = Start();
		double *end = End();

		// align to 16-byte boundary
		while ((long)p % 16 != 0) {
			*p = MySigmoid(*p);
			++p;
		}

		const __m128d twoThirds = _mm_set1_pd(2.0 / 3.0),
			one = _mm_set1_pd(1.0);

		_CRT_ALIGN(16) union U {
	       __int64 i[2];
	       __m128d m;
		};
		
		const U absMask = {0x7fffffffffffffff, 0x7fffffffffffffff};
		U x[4], newX[4];

		__m128d sign[4];

		while ((p+7) < end) {
			__m128d y[] = { _mm_load_pd(p),
				_mm_load_pd(p + 2),
				_mm_load_pd(p + 4),
				_mm_load_pd(p + 6) };

			sign[0] = _mm_andnot_pd(absMask.m, y[0]);
			sign[1] = _mm_andnot_pd(absMask.m, y[1]);
			sign[2] = _mm_andnot_pd(absMask.m, y[2]);
			sign[3] = _mm_andnot_pd(absMask.m, y[3]); // save sign
			
			y[0] = _mm_and_pd(absMask.m, y[0]);
			y[1] = _mm_and_pd(absMask.m, y[1]);
			y[2] = _mm_and_pd(absMask.m, y[2]);
			y[3] = _mm_and_pd(absMask.m, y[3]); // take absolute value

			x[0].m = y[0], x[1].m = y[1], x[2].m = y[2], x[3].m = y[3];

			while (true) {
				__m128d xSqr[] = {
					_mm_mul_pd(x[0].m, x[0].m), _mm_mul_pd(x[1].m, x[1].m),
					_mm_mul_pd(x[2].m, x[2].m), _mm_mul_pd(x[3].m, x[3].m) };

				newX[0].m = _mm_mul_pd(xSqr[0], x[0].m);
				newX[1].m = _mm_mul_pd(xSqr[1], x[1].m);
				newX[2].m = _mm_mul_pd(xSqr[2], x[2].m);
				newX[3].m = _mm_mul_pd(xSqr[3], x[3].m);

				newX[0].m = _mm_mul_pd(twoThirds, newX[0].m);
				newX[1].m = _mm_mul_pd(twoThirds, newX[1].m);
				newX[2].m = _mm_mul_pd(twoThirds, newX[2].m);
				newX[3].m = _mm_mul_pd(twoThirds, newX[3].m);

				newX[0].m = _mm_add_pd(newX[0].m, y[0]);
				newX[1].m = _mm_add_pd(newX[1].m, y[1]);
				newX[2].m = _mm_add_pd(newX[2].m, y[2]);
				newX[3].m = _mm_add_pd(newX[3].m, y[3]);

				xSqr[0] = _mm_add_pd(xSqr[0], one);
				xSqr[1] = _mm_add_pd(xSqr[1], one);
				xSqr[2] = _mm_add_pd(xSqr[2], one);
				xSqr[3] = _mm_add_pd(xSqr[3], one);

				newX[0].m = _mm_div_pd(newX[0].m, xSqr[0]);
				newX[1].m = _mm_div_pd(newX[1].m, xSqr[1]);
				newX[2].m = _mm_div_pd(newX[2].m, xSqr[2]);
				newX[3].m = _mm_div_pd(newX[3].m, xSqr[3]);

				newX[0].m = _mm_min_pd(newX[0].m, x[0].m);
				newX[1].m = _mm_min_pd(newX[1].m, x[1].m);
				newX[2].m = _mm_min_pd(newX[2].m, x[2].m);
				newX[3].m = _mm_min_pd(newX[3].m, x[3].m);

				if (newX[0].i[0] == x[0].i[0] && newX[0].i[1] == x[0].i[1] &&
					newX[1].i[0] == x[1].i[0] && newX[1].i[1] == x[1].i[1] &&
					newX[2].i[0] == x[2].i[0] && newX[2].i[1] == x[2].i[1] &&
					newX[3].i[0] == x[3].i[0] && newX[3].i[1] == x[3].i[1]) break;
				
				x[0] = newX[0];
				x[1] = newX[1];
				x[2] = newX[2];
				x[3] = newX[3];
			}

			x[0].m = _mm_or_pd(sign[0], newX[0].m);
			x[1].m = _mm_or_pd(sign[1], newX[1].m);
			x[2].m = _mm_or_pd(sign[2], newX[2].m);
			x[3].m = _mm_or_pd(sign[3], newX[3].m); // restore sign

			_mm_store_pd(p, x[0].m);
			_mm_store_pd(p+2, x[1].m);
			_mm_store_pd(p+4, x[2].m);
			_mm_store_pd(p+6, x[3].m);

			p += 8;
		}

		while (p < end) {
			*p = MySigmoid(*p);
			++p;
		}
	}
#endif

};



class AllocatingVector : virtual public MutableVector {
	void ResetStart() {
		if (_arr.size() > 0) _start = &_arr[0];
		assert (_len == _arr.size());
	}

	std::vector<double> _arr;
	
public:
	AllocatingVector(int len = 0, double init = 0)
		:
	Vector(0, len, 1),
	MutableVector(0, len, 1),
	_arr(len, init)
	{
		ResetStart();
	}
	
	AllocatingVector(const AllocatingVector & a)
		:
	Vector(0, a.Len(), 1),
	MutableVector(0, a.Len(), 1),
	_arr(a._arr)
	{
		ResetStart();
	}

	AllocatingVector(const Vector & a)
		:
	Vector(0, a.Len(), 1),
	MutableVector(0, a.Len(), 1),
	_arr(a.Len())
	{
		ResetStart();
		MutableVector::CopyFrom(a);
	}
	
	void Deserialize(std::istream & inStream) {
		inStream.read((char *) &_len, sizeof(int));
		_arr.resize(_len);
		ResetStart();
		if (_len > 0) inStream.read((char *) _start, _len * sizeof(double));
	}

	void Resize(int len) {
		_len = len;
		_arr.resize(len);
		ResetStart();
	}

	void CopyFrom(const Vector & a) {
		Resize(a.Len());
		MutableVector::CopyFrom(a);
	}

	AllocatingVector & operator=(const AllocatingVector & a) {
		_arr = a._arr;
		_len = a.Len();
		_inc = 1;
		ResetStart();
		return *this;
	}

	AllocatingVector & operator=(const Vector & a) {
		CopyFrom(a);
		return *this;
	}

	void Swap(AllocatingVector & other) {
		std::swap(_len, other._len);
		std::swap(_inc, other._inc);
		std::swap(_start, other._start);
		_arr.swap(other._arr);
	}

	void Assign(int len, double val) {
		Resize(len);
		MutableVector::Assign(val);
	}

	void Assign(double val) {
		MutableVector::Assign(val);
	}
};

class AllocatingMatrix;

class Matrix : virtual public Vector {
protected:
	int _numR, _numC;

	const double* PtrAt(int r, int c) const {
		return _start + c * _numR + r;
	}
	
public:

	Matrix(const double * start, int numR, int numC)
		:
	Vector(start, numR * numC, 1),
	_numR(numR), _numC(numC) { }

	typedef AllocatingMatrix AllocatingType;

	Matrix() : Vector(), _numR(0), _numC(0) { }

	Matrix(const Matrix & other)
		:
	Vector(other), _numR(other._numR), _numC(other._numC) { }

	Matrix(const std::vector<double> & v, int numR)
		:
	Vector(v), _numR(numR), _numC(v.size() / numR) { }

	Matrix(const Vector & col)
		:
	Vector(col), _numR(col.Len()), _numC(1) { }

	void Serialize(std::ostream & outStream) const {
		Vector::Serialize(outStream);
		outStream.write((const char *) &_numR, sizeof(int));
		outStream.write((const char *) &_numC, sizeof(int));
	}

	const static Matrix EmptyMatrix;

	const Matrix & operator=(const Matrix & other) {
		Vector::operator=(other);
		_numR = other.NumR();
		_numC = other.NumC();
		return *this;
	}

    //kl: begin: beginning column number of the instances, end: end column number of instances
     //kl: cause one column is one instance
	Matrix SubMatrix(int begin, int end) const {
		if (end < 0) end = _numC - ~end;
		return Matrix(_start + begin * _numR, _numR, end - begin);
	}

	int NumR() const { return _numR; }

	int NumC() const { return _numC; }

	const double & At(int r, int c) const {
		assert(r >= 0 && r < _numR);
		assert(c >= 0 && c < _numC);
		return *PtrAt(r,c);
	}

	Vector GetCol(int c) const;

	Vector GetRow(int r) const;

	void Print() const;
	
	void GetRowCol(int p, int & row, int & col) const {
		row = p % NumC();
		col = p / NumC();
	}

	static void SolvePSDSystem(MutableMatrix & A, MutableVector & b);
	
	void LogSumCols(MutableVector & result, std::vector<double> & temp) const;
	void LogSumCols2(MutableVector & result, std::vector<double> & temp) const;
};

class MutableMatrix : public Matrix, virtual public MutableVector {
	
protected:
	double * PtrAt(int r, int c) { return const_cast<double*>(Matrix::PtrAt(r, c)); }

public:
	MutableMatrix(double* start, int numR, int numC)
		:
	Matrix(start, numR, numC),
	Vector(start, numR * numC, 1),
	MutableVector(start, numR * numC, 1)
	{ }
		
	MutableMatrix() : Matrix() { }
		
	MutableMatrix(std::vector<double> & v, int numR)
		:
	Matrix(v, numR), Vector(v), MutableVector(v) { }

	void AddProd(const Matrix & a, bool aTrans, const Matrix & b, bool bTrans, double alpha = 1.0, double beta = 1.0);

	void AddTranspose(const Matrix & a, double scaleA = 1, double scaleSelf = 1);
	
	void SubtractInto(const Matrix & a, const Matrix & b) {
		MutableVector::SubtractInto(a, b);
	}

	void TransposeInPlace(double scale = 1.0);

	void EigenDecompose(MutableVector & eigenvals);
	
	// adds outer product of b and c
	void RankOneUpdate(const Vector & b, const Vector & c, double mult = 1.0);

	void SymmetricRankOneUpdate(const Vector & b, double mult = 1.0);

	void Syrk(const Matrix & other, double alpha, double beta);

	double & At(int r, int c) {
		assert(r >= 0 && r < _numR);
		assert(c >= 0 && c < _numC);
		return *PtrAt(r,c);
	}

	const double & At(int r, int c) const {
		return Matrix::At(r, c);
	}

    //kl: begin is the start instance number, end is the end instance number
	MutableMatrix SubMatrix(int begin, int end) {
		if (end < 0) end = _numC - ~end;
		return MutableMatrix(Start() + begin * _numR, _numR, end - begin);
	}

	Matrix SubMatrix(int begin, int end) const {
		return Matrix::SubMatrix(begin, end);
	}

	MutableVector GetCol(int c);

	Vector GetCol(int c) const;

	MutableVector GetRow(int r);

	Vector GetRow(int r) const;
	
	virtual void CopyFrom(const Matrix & other);
	
	void CopyTransposed(const Matrix & a);
};

class AllocatingMatrix : public MutableMatrix, virtual public AllocatingVector {
public:
	AllocatingMatrix(int numR = 0, int numC = 0)
		:
	MutableMatrix(0, numR, numC),
	Vector(0, numR * numC, 1),
	MutableVector(0, numR * numC, 1),
	AllocatingVector(numR * numC) { }
	
	AllocatingMatrix(int numR, int numC, const std::vector<double> & init)
		:
	MutableMatrix(0, numR, numC),
	Vector(0, numR * numC, 1),
	MutableVector(0, numR * numC, 1),
	AllocatingVector(init) {
		assert(numR * numC == init.size());
	}
	
	AllocatingMatrix(const Matrix & other)
		:
	MutableMatrix(0, other.NumR(), other.NumC()),
	Vector(other),
	MutableVector(0, other.NumR(), other.NumC()),
	AllocatingVector(other) { }
	
	AllocatingMatrix & operator=(const Matrix & other) {
		Matrix::operator=(other);
		AllocatingVector::operator=(other);
		return *this;
	}
	
	void Deserialize(std::istream & inStream) {
		AllocatingVector::Deserialize(inStream);
		inStream.read((char *) &_numR, sizeof(int));
		inStream.read((char *) &_numC, sizeof(int));
		assert(_numR * _numC == _len);
	}

	void Resize(int numR, int numC) {
		_numR = numR;
		_numC = numC;
		AllocatingVector::Resize(numR * numC);
	}

	void Resize(const Matrix & other) {
		Resize(other.NumR(), other.NumC());
	}

	void push_back(const Vector & col);

	void Swap(AllocatingMatrix & other) {
		if (this == &other) return;

		std::swap(_numR, other._numR);
		std::swap(_numC, other._numC);

		AllocatingVector::Swap(other);
	}

	void CopyFrom(const Matrix & other) {
		Resize(other.NumR(), other.NumC());
		MutableMatrix::CopyFrom(other);
	}

	void CopyTransposed(const Matrix & other) {
		Resize(other.NumC(), other.NumR());
		MutableMatrix::CopyTransposed(other);
	}	
};

void SolveLeastSquares(const Matrix & X, const Matrix & w, const Vector & y, std::vector<double> & temp, MutableVector & result);

void Whiten(const Matrix & images, AllocatingMatrix & whiteningMat, double eps = 1e-8);
