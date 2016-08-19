#pragma once

#include <iostream>

#include "mkl_lapacke.h"
#include "mkl.h"
#include "Matrix.h"

using namespace std;

class WhitenTransform {
	AllocatingMatrix _w;
	AllocatingVector _v;

public:
	void Init(const Matrix & data, int maxD = -1) {
		int m = data.NumC(), n = data.NumR();
		_v.Resize(n);

		AllocatingVector ones(m, 1.0);
		_v.MultInto(data, false, ones, 1.0 / m, 0.0);
		
		AllocatingMatrix centered(data);
		centered.RankOneUpdate(_v, ones, -1.0);

		int k = min(m,n);
		AllocatingVector singularValues(k);
		AllocatingVector superb(k);
		int info = LAPACKE_dgesvd(CblasColMajor, 'O', 'N', n, m, centered.Start(), n, singularValues.Start(),
			0, 1, 0, 1, superb.Start());

		if (info != 0) {
			cout << "dgesvd returned error code " << info << endl;
		}

		// centered was overwritten with left singular vectors. alias for clarity
		MutableMatrix U = centered;
		
		// NOTE: this might be clearer if we use the fact that the singularValues are decreasing

		const double eps = 1e-8 * singularValues[0];
		const double scale = sqrt(m-1.0);
		int n_new = 0;
		for (int f = 0; f < k; ++f) {
			if (singularValues[f] >= eps) {
				singularValues[n_new] = singularValues[f];
				U.GetCol(n_new++).ScaleInto(U.GetCol(f), scale / singularValues[f]);
				if (n_new == maxD) break;
			}
		}
//		cout << "whitening reduced dimensionality from " << n << " to " << n_new << endl;

		_w.Resize(n_new, n);
		_w.CopyTransposed(U.SubMatrix(0, n_new));
	}

	void InitAndWhiten(AllocatingMatrix & data, int maxD = -1) {
		int m = data.NumC(), n = data.NumR();
		_v.Resize(n);

		AllocatingVector ones(m, 1.0);
		_v.MultInto(data, false, ones, 1.0 / m, 0.0);
		
		AllocatingMatrix centered(data);
		centered.RankOneUpdate(_v, ones, -1.0);

		int k = min(m,n);
		AllocatingVector singularValues(k);
		AllocatingVector superb(k);
		int info = LAPACKE_dgesvd(CblasColMajor, 'O', 'S', n, m, centered.Start(), n, singularValues.Start(),
			0, 1, data.Start(), n, superb.Start());

		data *= sqrt(m-1.0);

		if (info != 0) {
			cout << "dgesvd returned error code " << info << endl;
		}

		// centered was overwritten with left singular vectors. alias for clarity
		MutableMatrix U = centered;
		
		// NOTE: this might be clearer if we use the fact that the singularValues are decreasing

		const double eps = 1e-8 * singularValues[0];
		const double scale = sqrt(m-1.0);
		int n_new = 0;
		for (int f = 0; f < k; ++f) {
			if (singularValues[f] >= eps) {
				singularValues[n_new] = singularValues[f];
				U.GetCol(n_new++).ScaleInto(U.GetCol(f), scale / singularValues[f]);
				if (n_new == maxD) break;
			}
		}

		if (n_new < n) {
			AllocatingMatrix reduced(n_new, m);
			for (int r = 0; r < n_new; ++r) reduced.GetRow(r).CopyFrom(data.GetRow(r));
			reduced.Swap(data);
		}

		_w.Resize(n_new, n);
		_w.CopyTransposed(U.SubMatrix(0, n_new));
	}

	void Serialize(ostream & stream) const {
		_w.Serialize(stream);
		_v.Serialize(stream);
	}

	void Deserialize(istream & stream) {
		_w.Deserialize(stream);
		_v.Deserialize(stream);
	}

	void Transform(AllocatingMatrix & data) const {
		Transform(data, data);
	}

	void Transform(const Matrix & data, AllocatingMatrix & transformedData) const {
		int m = data.NumC();
		AllocatingVector ones(m, 1.0);
		AllocatingMatrix centered(data);
		centered.RankOneUpdate(_v, ones, -1.0);
		transformedData.Resize(_w.NumR(), m);
		transformedData.AddProd(_w, false, centered, false, 1.0, 0.0);
	}

	Matrix GetW() { return _w; }

	Vector GetV() { return _v; }

	static void TestWhitened(const Matrix & whitened) {
		int m = whitened.NumC(), n = whitened.NumR();
		AllocatingVector ones(m, 1.0), mu(n);
		mu.MultInto(whitened, false, ones, 1.0 / m, 0.0);

		cout << "mu error: " << mu.Norm() << endl;

		AllocatingMatrix sigma(n,n);
		sigma.AddProd(whitened, false, whitened, true, 1.0 / (m-1));

		for (int i = 0; i < n; ++i) sigma.At(i,i) -= 1;
		cout << "sigma error: " << sigma.Norm() << endl;
	}

	int InSize() const { return _w.NumC(); }

	int OutSize() const { return _w.NumR(); }
};