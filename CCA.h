#pragma once

#include "Matrix.h"
#include <assert.h>
#include <iostream>

#include "WhitenTransform.h"

class CCA {
	vector<AllocatingMatrix> _w; //kl: finally _w[0] = A1_*, _w[1] = A2_* in the paper page2, use this in the "Map" method to generate the pairs of linear projections of the two views (w1'X1, w2'X2) in the paper page2, these projections are the outputs from CCA
	vector<AllocatingVector> _mu; //kl: _mu[i], the average over whole instances

    //kl: get the mean over the number of instances for the matrix X, and stores in vector mu
	static void GetMean(const Matrix & X, AllocatingVector & mu) { 
		int m = X.NumC(); //kl: number of instances
		AllocatingVector _ones(m, 1.0); //kl: a m length vector with all being 1.0
		mu.Resize(X.NumR()); //kl: X.NumR() gives the number of features
		mu.MultInto(X, false, _ones, 1.0 / m, 0.0); //kl: mu = mu * X(matrix) * 1/m to get the mean over the number of instances for the matrix X.
	}

    ////kl: "substracting mean from the matrix". the matrix who call this function, getting barX = barX - mu*_ones' in which the origin barX = X is the training dataset for all instances,basically here barX is the matrix who substracting the mean for each instances datasets. 
	static void Translate(const Matrix & X, const Vector & mu, AllocatingMatrix & barX) {
		int m = X.NumC();
		AllocatingVector _ones(m, 1.0);
		barX.CopyFrom(X); //kl: copy matrix X into barX
		barX.RankOneUpdate(mu, _ones, -1.0); //kl: the matrix who call this function, getting matrix = mult*b*c' + matrix
	}

public:
	CCA() : _w(2), _mu(2) { }

    //kl: inner class TraceNormObjective for CCA class
	class TraceNormObjective {
		vector<AllocatingMatrix> _barX;
		AllocatingMatrix _S11, _S12, _S22; //kl: _S11 = sigma_11{-1/2}, _S12=sigma_12, _S22=sigma_22^{-1/2}
		AllocatingMatrix _nabla11, _nabla12, _nabla22; //kl: equation (12), (13) in the paper
		vector<AllocatingVector> _mu;
		AllocatingMatrix _U, _Vt; // kl: _U * 
		AllocatingVector _D;
		AllocatingVector _superB;
		vector<double> _lambda;

		// computes the Cholesky factor of [(HH' + rI)/(m-1)]^{1/2} on page 4
		// Sigma_ii^{1/2} in the paper
          //kl: 《矩阵论》p191, A = GGt, 实对称矩阵A的Cholesky分解，平方根分解，对称三角分解，G是下三角矩阵
          //kl: get the Cholesky decomposition matrix G, and stored into the matrix output. output = Sigma_ii^{1/2} in the paper
		void MakeSqrtCovCholesky(const Matrix & input, AllocatingMatrix & output, double lambda) {
			int n = input.NumR(), m = input.NumC();
			int k = min(n,m);

			_U.Resize(n, n);
            
            //kl: covariance matrix here. _U = 1/(m-1)*input*inputT, ******, helping to get the covariance matrix for the averaged input
			_U.Syrk(input, 1.0 / (m-1), 0.0);
			for (int i = 0; i < n; ++i) _U.At(i,i) += lambda / (m-1); // kl: diagonal element re-assigned

			_D.Resize(n);
            //kl: DSYEVD computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix A.
             //kl: egienvalues of _U stored in a double precision array _D, and the egienvectors stored in _U itself (replace original _U)
			int info = LAPACKE_dsyevd(CblasColMajor, 'V', 'U', n, _U.Start(), n, _D.Start());

			if (info != 0) {
				cout << "dsyevd returned error code " << info << endl;
			}

			output.Resize(n,n);
			output.Clear();
			for (int i = 0; i < n; ++i) {
                //kl: Calculates "output + sqrt(_D[i])*_U.GetCol(i)*_U.GetCol(i)T" and stores the result in "output".
				output.SymmetricRankOneUpdate(_U.GetCol(i), sqrt(_D[i]));
			}

            //kl: DPOTRF computes the Cholesky factorization of a real symmetric positive definite matrix A.
             //kl: Upper triangle of output is stored into the updated output. output = Sigma_ii^{1/2} in the paper
			info = LAPACKE_dpotrf(CblasColMajor, 'U', n, output.Start(), n);

			if (info != 0) {
				cout << "dpotrf returned error code " << info << endl;
			}
		}

	public:
        //kl: const inference lambda can't be changed 
		TraceNormObjective(const vector<double> & lambda) : _mu(2), _barX(2), _lambda(lambda) { }

        //kl: evaluate Trace of equation (10)
         //kl: importantly, generate the DCCA gradients at top for backpropagation, D1 and D2 are these gradients, as errors1, errors2 for modifying the neural networks
		double EvalTrace(const Matrix & X1, const Matrix & X2, MutableMatrix & D1, MutableMatrix & D2) {
			const Matrix X[] = { X1, X2 };
			int n1 = X1.NumR(), n2 = X2.NumR();
			int k = min(n1, n2);
			int m = X1.NumC(); //kl: NumC is the number of instances
			assert (m == X2.NumC());

			bool shallowView1 = (D1.Len() == 0);
			bool shallowView2 = (D2.Len() == 0);
			
			for (int i = 0; i < 2; ++i) {
				GetMean(X[i], _mu[i]); // kl: mean of X into _mu
				Translate(X[i], _mu[i], _barX[i]); //kl: substract _mu from X and stored into _barX
			}

			_S12.Resize(n1, n2);
            //kl: _S12 = _barX[0] * _barX[1] * 1/(m-1); the cross covariance matrix between X1 and X2
			_S12.AddProd(_barX[0], false, _barX[1], true, 1.0 / (m-1), 0.0);

			MakeSqrtCovCholesky(_barX[0], _S11, _lambda[0]); //kl: cholesky decomposition of _barX[0], the upper triangle matrix stored into _S11 = Sigma_11^{1/2} in the paper, covariance matrix by _barX[0] formed in calling the MakeSqrtCovCholesky method
			MakeSqrtCovCholesky(_barX[1], _S22, _lambda[1]); //kl: cholesky decomposition of _barX[1], the upper triangle matrix stored into _S22 =Sigma_22^{1/2} in the paper

			// set _S12 = S11^{-1/2} S12 S22^{-1/2} kl: 
			// first multiply by S22^{-1/2} on right
            //kl: DPOTRS solves a system of linear equations A*X = B with a symmetric positive definite matrix A using the Cholesky factorization A = U_T*U or A = L*L_T computed by DPOTRF.                                          X = _S12 * _S22(ie. sigma_22^{-1/2}) stored in _S12
			LAPACKE_dpotrs(CblasRowMajor, 'L', n2, n1, _S22.Start(), n2, _S12.Start(), n1);
			// then multiply by S11^{-1/2} on left
			LAPACKE_dpotrs(CblasColMajor, 'U', n1, n2, _S11.Start(), n1, _S12.Start(), n1);
            //kl: so far now, _S12 = T = sigma_11^{-1/2}*sigma_12*sigma_22^{-1/2} in the paper

			_U.Resize(n1, k);
			_Vt.Resize(k, n2);

			// get SVD of S11^{-1/2} S12 S22^{-1/2}
			_D.Resize(k);
			_superB.Resize(k);
            //kl: DGESVD computes the singular value decomposition (SVD) of a real M-by-N matrix _S12=T, optionally computing the left and/or right singular vectors.
            //kl: Arguments: JOBU 'S':  the first min(m,n) columns of U (the left singular vectors) are returned in the array _U; JOBVT 'S': the first min(m,n) rows of Vt (the right singular vectors) are returned in the array _Vt; n1:  The number of rows of the input matrix T.; n2:The number of columns of the input matrix T.; _S12: input matrix for SVD, after SVD, its content is destroyed; n1: The leading dimension of the matrix _S12. ; _D: (output) Singular values stored. DOUBLE PRECISION array, dimension (min(M,N)) The singular values of T, sorted so that _D(i) >= _D(i+1). ; _U:  U contains the first min(m,n) columns of U (the left singular vectors, stored columnwise); n1: The leading dimension of the array _U ; _Vt: VT contains the first min(m,n) rows of V_T (the right singular vectors, stored rowwise); k: The leading dimension of the array VT; _superB: 
			int info = LAPACKE_dgesvd(CblasColMajor, 'S', 'S', n1, n2, _S12.Start(), n1, _D.Start(),
				_U.Start(), n1, _Vt.Start(), k, _superB.Start());

			if (info != 0) {
				cout << "dgesvd returned error code " << info << endl;
			}

			double val = _D.Sum(); //kl: sum up all the singular values from T's SVD, which equals to the total correlation. 
			
			// put S11^{-1/2} U in _U
             //kl: _U = sigma_11^{-1/2} * _U ＝ A1_* in paper page2
			LAPACKE_dpotrs(CblasColMajor, 'U', n1, k, _S11.Start(), n1, _U.Start(), n1);
			// put S11^{-1/2} V in _V
             //kl: _Vt = sigma_22^{-1/2} * _Vt = A2_* in paper page2
			LAPACKE_dpotrs(CblasRowMajor, 'L', n2, k, _S22.Start(), n2, _Vt.Start(), k);

			// form nabla12
			_nabla12.Resize(n1, n2);
             //kl: _nabla12 = sigma_11^{-1/2}*U*Vt*sigma_22^{-1/2}, equation (12) in the paper
			_nabla12.AddProd(_U, false, _Vt, false, 1.0, 0.0);

            //kl: get _nabla11 = equation (13) in the paper
             //kl: more important, here we get the errors1 from equation (11), stored in D1, for backpropagation 
			if (!shallowView1) {
				for (int i = 0; i < k; ++i) {
					_U.GetCol(i) *= sqrt(_D[i]); //kl: introduce the singular values from D here for equation (13)
				}
				_nabla11.Resize(n1, n1);
                //kl: get _nabla11 = _U *_Ut in equation (13) pay attention at each step, the _U hold different values
				cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, n1, k, -1.0/2, _U.Start(), n1, 0, _nabla11.Start(), n1);
                //kl: D1 = _nabla12 * _barX[1] * (-1/2)
				D1.AddProd(_nabla12, false, _barX[1], false, -1.0/2, 0.0);
                //kl: D1 = D1 - _nabla11 * _barX[0]
				cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, n1, m, -1.0, _nabla11.Start(), n1, _barX[0].Start(), n1, 1.0, D1.Start(), n1);
			}

            //kl: get _nabla22
             //kl: meanwhile, errors2 in D2 similar to equation (12) in the paper
			if (!shallowView2) {
				for (int i = 0; i < k; ++i) {
					_Vt.GetRow(i) *= sqrt(_D[i]);
				}
				_nabla22.Resize(n2, n2);
                //kl: Calculates alpha*A*AT + beta*C
                 //kl: _nabla22 = _Vt *_Vtt
				cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, n2, k, -1.0/2, _Vt.Start(), k, 0, _nabla22.Start(), n2);
				D2.AddProd(_nabla12, true, _barX[0], false, -1.0/2, 0.0);
				cblas_dsymm(CblasColMajor, CblasLeft, CblasUpper, n2, m, -1.0, _nabla22.Start(), n2, _barX[1].Start(), n2, 1.0, D2.Start(), n2);
			}

			return 0.5 * (m-1) * (k - val);
		}
	};
	
	template<class ArrayOfMatrices>
	static double TestCorr(const ArrayOfMatrices & X) {
		if (X[0].NumC() == 0 || X[1].NumC() == 0) return NaN;

		int m = X[0].NumC(); //kl: number of instances
		assert (X[1].NumC() == m); //kl: ensure both views have the same number of instances.

		int n[] = { X[0].NumR(), X[1].NumR() }; // kl: number of feature 

		vector<AllocatingMatrix> temp(2);
		for (int i = 0; i < 2; ++i) {
			// put centered in temp
			AllocatingVector mu(n[i]);
			GetMean(X[i], mu);       
			Translate(X[i], mu, temp[i]); //kl: centered X into temp

			// compute SVD
			int k = min(n[0], m);
			AllocatingMatrix U(n[i], k), Vt(k, m);
			AllocatingVector singularValues(k), superb(k);

            //kl: SVD on barX to get U and Vt
			int info = LAPACKE_dgesvd(CblasColMajor, 'S', 'S', n[i], m, temp[i].Start(), n[i], singularValues.Start(),
				U.Start(), n[i], Vt.Start(), k, superb.Start());

			if (info != 0) {
				cout << "dgesvd returned error code " << info << endl;
			}

			// put UV' in temp
			temp[i].AddProd(U, false, Vt, false, 1.0, 0.0);
		}

		AllocatingMatrix temp2(n[0], n[1]);
		temp2.AddProd(temp[0], false, temp[1], true, 1.0, 0.0);

		int k = min(n[0],n[1]);
		AllocatingVector singularValues(k);
		AllocatingVector superb(k);
		int info = LAPACKE_dgesvd(CblasColMajor, 'N', 'N', n[0], n[1], temp2.Start(), n[0], singularValues.Start(),
			0, n[0], 0, k, superb.Start());

		if (info != 0) {
			cout << "dgesvd returned error code " << info << endl;
		}

		return singularValues.Sum();
	}

	template<class ArrType1>
	void Map(ArrType1 & X) const {
		Map(X, X);
	}

	template<class ArrType1, class ArrType2>
	void Map(const ArrType1 & X, ArrType2 & mapped) const {
		AllocatingMatrix barX;
		for (int i = 0; i < 2; ++i) {
			Translate(X[i], _mu[i], barX);
			mapped[i].Resize(_w[i].NumR(), X[i].NumC());
			mapped[i].AddProd(_w[i], false, barX, false, 1.0, 0.0);
		}
	}

    //kl: Map is for generating the CCA projection outputs for two views. 
	void Map(const Matrix & X, int which, AllocatingMatrix & mapped) const {
        cout<<"even get here"<< endl;
		AllocatingMatrix barX;
		Translate(X, _mu[which], barX);
        
        cout<<endl;
        cout<<"centered X"<<which<<" start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(barX.Start()+k)<<" ";
        }
        
		mapped.Resize(_w[which].NumR(), X.NumC());
		mapped.AddProd(_w[which], false, barX, false, 1.0, 0.0);
	}
    
    // kl: this MapForDoc is added for the NLP task we are doing for Documents classification. 
    void MapForDoc(const Matrix & X, int which, AllocatingMatrix & mapped) const {
		//AllocatingMatrix barX;
		//Translate(X, _mu[which], barX);
        
		mapped.Resize(_w[which].NumR(), X.NumC());
		mapped.AddProd(_w[which], false, X, false, 1.0, 0.0);
        cout <<"view "<<which<<" output's NumC: "<< mapped.NumC()<< endl;
        cout << "view "<<which<<" output's NumR: "<< mapped.NumR()<< endl;
	}

    //kl: output and store the parameters
	void Serialize(ostream & outStream) const {
		for (int i = 0; i < 2; ++i) {
			_mu[i].Serialize(outStream);
			_w[i].Serialize(outStream);
		}
	}
	
    //kl: input the parameters
	void Deserialize(istream & inStream) {
		for (int i = 0; i < 2; ++i) {
			_mu[i].Deserialize(inStream);
			_w[i].Deserialize(inStream);
		}
	}

    //kl: this "InitWeights" method could be as the candidate for the pure CCA.
     //kl: which gives the A1_*, A2_* in the paper page 2, to get the new projection output for X1, X2, use the mapup to muliply the X1 * A1_*, X2 * A2_*
	template <class ArrayOfMatrices>
	double InitWeights(const ArrayOfMatrices & X, double reg1 = 0, double reg2 = NaN) {
		if (IsNaN(reg2)) reg2 = reg1;

		int m = X[0].NumC();
		int n[] = { X[0].NumR(), X[1].NumR() };
		assert (X[1].NumC() == m);
		double reg[] = { reg1, reg2 };

		vector<AllocatingMatrix> U(2);
		vector<AllocatingMatrix> centered(2);

		for (int i = 0; i < 2; ++i) {
			GetMean(X[i], _mu[i]);
			Translate(X[i], _mu[i], centered[i]); //kl: center X

			U[i].Resize(n[i], n[i]); //kl: square matrix
			U[i].Syrk(centered[i], 1.0 / (m - 1), 0.0); //kl: get covariance matrix in U
			double thisReg = max(reg[i] / m, 1e-8);
			for (int r = 0; r < n[i]; ++r) U[i].At(r,r) += thisReg; //kl: add regularization to the covariance matrix
            //kl: DPOTRF computes the Cholesky factorization of a real symmetric positive definite matrix A.
            //kl: Upper triangle of U[i] is stored into the updated U[i] = Sigma_ii^{1/2} in the paper
			LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n[i], U[i].Start(), n[i]);
		}

		AllocatingMatrix S12(n[0], n[1]);
		S12.AddProd(centered[0], false, centered[1], true, 1.0 / (m-1), 0.0); //kl: covariance matrix in S12

		// set S12 = U[0]^{-1}' S12 U[1]^{-1}
         //kl: right-multiply U[0]^{-1}, where U[1]= signma_22^{1/2}
		LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N', n[1], n[0], U[1].Start(), n[1], S12.Start(), n[0]);
         //kl: left-multiply U[1]^{-1}, where U[0]= signma_11^{1/2}
		LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', n[0], n[1], U[0].Start(), n[0], S12.Start(), n[0]);
        //kl: so now S12 = sigma_11^{-1/2}*sigma_12*sigma_22^{-1/2} = T in the paper

		int k = min(n[0], n[1]);
		_w[0].Resize(n[0], k);
		_w[1].Resize(k, n[1]);
		AllocatingVector singularValues(k);
		AllocatingVector superb(k);
         //kl: SVD on T in the paper to get U and Vt, U=_w[0], Vt=_w[1]
		int info = LAPACKE_dgesvd(CblasColMajor, 'S', 'S', n[0], n[1], S12.Start(), n[0], singularValues.Start(),
			_w[0].Start(), n[0], _w[1].Start(), k, superb.Start());

		if (info != 0) {
			cout << "dgesvd returned error code " << info << endl;
		}

		_w[1].TransposeInPlace(); //kl: transpose of _w[0]
       
       
        
        //kl: _w[i] = U[i]^{-1}*_w[i] = sigma_11^{-1/2}*U or sigma_22^{-1/2}*V, the A1_*, A2_* in page2
		for (int i = 0; i < 2; ++i) {
           LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', k, n[i], U[i].Start(), n[i], _w[i].Start(), k);
            cout << "NumC: " << _w[i].NumC() << endl;
            cout << "NumR: " << _w[i].NumR() << endl;
        }
        
        int kp = 50;
        for(int i=0;i<2;i++){
            _w[i].Resize(n[0], kp);
            
            _w[i].CopyFrom(_w[i].SubMatrix(0, kp));
            _w[i].TransposeInPlace();
            cout<<"_A "<<i<< "'s NumC: "<< _w[i].NumC()<<endl;
            cout<<"_A "<<i<< "'s NumR: "<< _w[i].NumR()<<endl;
        }
        
        for(int i=0;i<2;i++){
            
            //_w[i].WriteToFile(outputW[i]);
            cout<<"A "<<i<< "'s NumC: "<< _w[i].NumC()<<endl;
            cout<<"A "<<i<< "'s NumR: "<< _w[i].NumR()<<endl;
        }
        vector<string> outputA(2);
        outputA[0] = "outputData2/A_Out1_Centered.dat";
        outputA[1] = "outputData2/A_Out2_Centered.dat";
        
        for(int i=0;i<2;i++){
         
         _w[i].WriteToFile(outputA[i]);
         }
        
        cout<<"A1 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(_w[0].Start()+k)<<" ";
        }
        cout<<endl;
        cout<<"A2 start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(_w[1].Start()+k)<<" ";
        }
        
        cout<<endl;
        cout<<"X1 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(X[0].Start()+k)<<" ";
        }
        
        cout<<endl;
        cout<<"X2 start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(X[1].Start()+k)<<" ";
        }
        
        vector<string> outputData(2);
        outputData[0] = "outputData2/trainOut1_Centered.dat";
        outputData[1] = "outputData2/trainOut2_Centered.dat";
        
        AllocatingMatrix mapped1;
        AllocatingMatrix mapped2;
        
        Map(X[0], 0, mapped1);
        
        cout<<endl;
        cout<<"mapped1 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(mapped1.Start()+k)<<" ";
        }
        
        cout <<"view "<<0<<" output's NumC: "<< mapped1.NumC()<< endl;
        cout << "view "<<0<<" output's NumR: "<< mapped1.NumR()<< endl;
        
        mapped1.WriteToFile(outputData[0]);
        
        Map(X[1], 1, mapped2);
        
        cout<<"mapped2 start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(mapped2.Start()+k)<<" ";
        }
        cout <<"view "<<1<<" output's NumC: "<< mapped2.NumC()<< endl;
        cout << "view "<<1<<" output's NumR: "<< mapped2.NumR()<< endl;
        mapped2.WriteToFile(outputData[1]);
        
		return singularValues.Sum();
	}
    
    //kl: this is for nlp task without data-centering. 
    template <class ArrayOfMatrices>
	double InitWeightsDoc(const ArrayOfMatrices & X, const ArrayOfMatrices & Y, double reg1 = 0, double reg2 = NaN) {
		if (IsNaN(reg2)) reg2 = reg1;
        
		int m = X[0].NumC();
        cout<< "Number of instance: "<< m <<endl;
		int n[] = { X[0].NumR(), X[1].NumR() };
		assert (X[1].NumC() == m);
        
        
        
		double reg[] = { reg1, reg2 };
        
		vector<AllocatingMatrix> U(2);
		vector<AllocatingMatrix> centered(2);
        
		for (int i = 0; i < 2; ++i) {
			//GetMean(X[i], _mu[i]);
			//Translate(X[i], _mu[i], centered[i]); //kl: center X
            
			U[i].Resize(n[i], n[i]); //kl: square matrix
			U[i].Syrk(X[i], 1.0 / (m - 1), 0.0); //kl: get covariance matrix in U
			double thisReg = max(reg[i] / m, 1e-8);
			for (int r = 0; r < n[i]; ++r) U[i].At(r,r) += thisReg; //kl: add regularization to the covariance matrix
            //kl: DPOTRF computes the Cholesky factorization of a real symmetric positive definite matrix A.
            //kl: Upper triangle of U[i] is stored into the updated U[i] = Sigma_ii^{1/2} in the paper
			LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n[i], U[i].Start(), n[i]);
		}
        
        vector<string> outputU(2);
        outputU[0] = "outputData/Sigma11output.dat";
        outputU[1] = "outputData/Sigma22output.dat";
        
        for(int i=0;i<2;i++){
            
            U[i].WriteToFile(outputU[i]);
            cout<<"Sigma"<<i+1<<i+1<<" 's NumC: "<<U[i].NumC()<<endl;
            cout<<"Sigma"<<i+1<<i+1<<" 's NumR: "<<U[i].NumR()<<endl;
        }
        
		AllocatingMatrix S12(n[0], n[1]);
		S12.AddProd(X[0], false, X[1], true, 1.0 / (m-1), 0.0); //kl: covariance matrix in S12
        
        /*string outputS12 = "outputData/outputS12";
        S12.WriteToFile(outputS12);
        cout<<"Sigma12's NumC: "<<S12.NumC()<<endl;
        cout<<"Sigma12's NumR: "<<S12.NumR()<<endl;*/
        
		// set S12 = U[0]^{-1}' S12 U[1]^{-1}
        //kl: right-multiply U[1]^{-1}, where U[1]= signma_22^{1/2}
		LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N', n[1], n[0], U[1].Start(), n[1], S12.Start(), n[0]);
        //kl: left-multiply U[0]^{-1}, where U[0]= signma_11^{1/2}
		LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', n[0], n[1], U[0].Start(), n[0], S12.Start(), n[0]);
        //kl: so now S12 = sigma_11^{-1/2}*sigma_12*sigma_22^{-1/2} = T in the paper
        
		int k = min(n[0], n[1]);
		_w[0].Resize(n[0], k);
		_w[1].Resize(k, n[1]);
		AllocatingVector singularValues(k);
		AllocatingVector superb(k);
        //kl: SVD on T in the paper to get U and Vt, U=_w[0], Vt=_w[1]
		int info = LAPACKE_dgesvd(CblasColMajor, 'S', 'S', n[0], n[1], S12.Start(), n[0], singularValues.Start(),
                                  _w[0].Start(), n[0], _w[1].Start(), k, superb.Start());
        
        
		if (info != 0) {
			cout << "dgesvd returned error code " << info << endl;
		}
        
        //_w[1].TransposeInPlace();
      
        
        _w[1].TransposeInPlace(); //kl: transpose of _w[0], since V is already Vt, so _w[1] not necessary to be transposed.
       
        vector<string> outputW(2);
        outputW[0] = "outputDataAE/W_Out1_noCentered.dat";
        outputW[1] = "outputDataAE/W_Out2_noCentered.dat";
        
        
        for(int i=0;i<2;i++){
            
            _w[i].WriteToFile(outputW[i]);
            cout<<"_W "<<i<< "'s NumC: "<< _w[i].NumC()<<endl;
            cout<<"_W "<<i<< "'s NumR: "<< _w[i].NumR()<<endl;
        }
        
        cout<<"W1 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(_w[0].Start()+k)<<" ";
        }
        cout<<endl;
        
        cout<<"W2 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(_w[1].Start()+k)<<" ";
        }
        cout<<endl;
        
        
        
        //kl: _w[i] = U[i]^{-1}*_w[i] = sigma_11^{-1/2}*Ut or sigma_22^{-1/2}*Vt, the A1_*, A2_* in page2
		for (int i = 0; i < 2; ++i) {
            LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', k, n[i], U[i].Start(), n[i], _w[i].Start(), k);
            //cout <<"A's NumC: "<< _w[i].NumC()<< endl;
            //cout << "A's NumR: "<< _w[i].NumR()<< endl;
        }
         /*for(int i=0;i<2;i++){
         _w[i].TransposeInPlace();
         }*/
        int kp = 50;
         for(int i=0;i<2;i++){
         _w[i].Resize(n[i], kp);
         
         _w[i].CopyFrom(_w[i].SubMatrix(0, kp));
         _w[i].TransposeInPlace();
         cout<<"_A "<<i<< "'s NumC: "<< _w[i].NumC()<<endl;
         cout<<"_A "<<i<< "'s NumR: "<< _w[i].NumR()<<endl;
         }
        
        
        for(int i=0;i<2;i++){
            
            //_w[i].WriteToFile(outputW[i]);
            cout<<"A "<<i<< "'s NumC: "<< _w[i].NumC()<<endl;
            cout<<"A "<<i<< "'s NumR: "<< _w[i].NumR()<<endl;
        }
        vector<string> outputA(2);
        outputA[0] = "outputDataAE/A_Out1_noCentered.dat";
        outputA[1] = "outputDataAE/A_Out2_noCentered.dat";
        
        for(int i=0;i<2;i++){
            
            _w[i].WriteToFile(outputA[i]);
        }
        
        cout<<"A1 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(_w[0].Start()+k)<<" ";
        }
        cout<<endl;
        cout<<"A2 start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(_w[1].Start()+k)<<" ";
        }
        
        cout<<endl;
        cout<<"X1 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(X[0].Start()+k)<<" ";
        }
        
        cout<<endl;
        cout<<"X2 start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(X[1].Start()+k)<<" ";
        }
        
        //kl: generate train outputs
        vector<string> outputData(2);
        outputData[0] = "outputDataAE/trainOut1_noCentered.dat";
        outputData[1] = "outputDataAE/trainOut2_noCentered.dat";
        
        AllocatingMatrix mapped1;
        AllocatingMatrix mapped2;
        
        mapped1.Resize(_w[0].NumR(), X[0].NumC());
		mapped1.AddProd(_w[0], false, X[0], false, 1.0, 0.0);
        
        cout<<endl;
        cout<<"mapped1 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(mapped1.Start()+k)<<" ";
        }
        
        cout<<endl;
        cout <<"Trainview "<<0<<" output's NumC: "<< mapped1.NumC()<< endl;
        cout << "Train view "<<0<<" output's NumR: "<< mapped1.NumR()<< endl;
        
        mapped1.WriteToFile(outputData[0]);
        
        
        mapped2.Resize(_w[1].NumR(), X[1].NumC());
		mapped2.AddProd(_w[1], false, X[1], false, 1.0, 0.0);
        
        cout<<"mapped2 start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(mapped2.Start()+k)<<" ";
        }
        cout<<endl;
        cout <<"view "<<1<<" output's NumC: "<< mapped2.NumC()<< endl;
        cout << "view "<<1<<" output's NumR: "<< mapped2.NumR()<< endl;
        mapped2.WriteToFile(outputData[1]);
        
        //kl: generate the test outputs
        vector<string> outputDataTest(2);
        outputDataTest[0] = "outputDataAE/testOut1_noCentered.dat";
        outputDataTest[1] = "outputDataAE/testOut2_noCentered.dat";
        
        AllocatingMatrix mapped1Test;
        AllocatingMatrix mapped2Test;
        
        mapped1Test.Resize(_w[0].NumR(), Y[0].NumC());
		mapped1Test.AddProd(_w[0], false, Y[0], false, 1.0, 0.0);
        
        cout<<endl;
        cout<<"mapped1Test start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(mapped1Test.Start()+k)<<" ";
        }
        cout<<endl;
        cout <<"Testview "<<0<<" output's NumC: "<< mapped1Test.NumC()<< endl;
        cout << "Testview "<<0<<" output's NumR: "<< mapped1Test.NumR()<< endl;
        
        mapped1Test.WriteToFile(outputDataTest[0]);
        
        
        mapped2Test.Resize(_w[1].NumR(), Y[1].NumC());
		mapped2Test.AddProd(_w[1], false, Y[1], false, 1.0, 0.0);
        
        cout<<"mapped2Test start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(mapped2Test.Start()+k)<<" ";
        }
        cout<<endl;
        cout <<"Testview "<<1<<" output's NumC: "<< mapped2Test.NumC()<< endl;
        cout << "Testview "<<1<<" output's NumR: "<< mapped2Test.NumR()<< endl;
        mapped2Test.WriteToFile(outputDataTest[1]);

        
        AllocatingMatrix test1;
        AllocatingMatrix test2;
        
        test1.Resize(2, 3);
        test2.Resize(2, 3);
        
        *(test1.Start()) = 1.0;
        *(test1.Start()+1) = 0.0;
        *(test1.Start()+2) = 0.0;
        *(test1.Start()+3) = 2.0;
        *(test1.Start()+4) = 0.0;
        *(test1.Start()+5) = 1.0;
        
        *(test2.Start()) = 1.0;
        *(test2.Start()+1) = 0.0;
        *(test2.Start()+2) = 0.0;
        *(test2.Start()+3) = 0.0;
        *(test2.Start()+4) = 1.0;
        *(test2.Start()+5) = 1.0;
        
        AllocatingMatrix mappedtest;
        mappedtest.Resize(test1.NumR(), test2.NumR());
		mappedtest.AddProd(test1, false, test2, true, 1.0, 0.0);
        
        cout<< "Addprod test: "<<endl;
        for(int k=0;k<mappedtest.NumR()*mappedtest.NumC();k++){
            cout<<*(mappedtest.Start()+k)<<" ";
        }
        cout<<endl;
                
		return singularValues.Sum();
	}
    
    //kl: this is for nlp task without data-centering.
    template <class ArrayOfMatrices>
	double InitWeightsNLP(const ArrayOfMatrices & X, double reg1 = 0, double reg2 = NaN) {
		if (IsNaN(reg2)) reg2 = reg1;
        
		int m = X[0].NumC();
        cout<< "Number of instance: "<< m <<endl;
		int n[] = { X[0].NumR(), X[1].NumR() };
		assert (X[1].NumC() == m);
        
        
        
		double reg[] = { reg1, reg2 };
        
		vector<AllocatingMatrix> U(2);
		vector<AllocatingMatrix> centered(2);
        
		for (int i = 0; i < 2; ++i) {
			//GetMean(X[i], _mu[i]);
			//Translate(X[i], _mu[i], centered[i]); //kl: center X
            
			U[i].Resize(n[i], n[i]); //kl: square matrix
			U[i].Syrk(X[i], 1.0 / (m - 1), 0.0); //kl: get covariance matrix in U
			double thisReg = max(reg[i] / m, 1e-8);
			for (int r = 0; r < n[i]; ++r) U[i].At(r,r) += thisReg; //kl: add regularization to the covariance matrix
            //kl: DPOTRF computes the Cholesky factorization of a real symmetric positive definite matrix A.
            //kl: Upper triangle of U[i] is stored into the updated U[i] = Sigma_ii^{1/2} in the paper
			LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n[i], U[i].Start(), n[i]);
		}
        
        vector<string> outputU(2);
        outputU[0] = "outputData2/Sigma11output.dat";
        outputU[1] = "outputData2/Sigma22output.dat";
        
        for(int i=0;i<2;i++){
            
            //U[i].WriteToFile(outputU[i]);
            cout<<"Sigma"<<i+1<<i+1<<" 's NumC: "<<U[i].NumC()<<endl;
            cout<<"Sigma"<<i+1<<i+1<<" 's NumR: "<<U[i].NumR()<<endl;
        }
        
		AllocatingMatrix S12(n[0], n[1]);
		S12.AddProd(X[0], false, X[1], true, 1.0 / (m-1), 0.0); //kl: covariance matrix in S12
        
        /*string outputS12 = "outputData/outputS12";
         S12.WriteToFile(outputS12);
         cout<<"Sigma12's NumC: "<<S12.NumC()<<endl;
         cout<<"Sigma12's NumR: "<<S12.NumR()<<endl;*/
        
		// set S12 = U[0]^{-1}' S12 U[1]^{-1}
        //kl: right-multiply U[1]^{-1}, where U[1]= signma_22^{1/2}
		LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N', n[1], n[0], U[1].Start(), n[1], S12.Start(), n[0]);
        //kl: left-multiply U[0]^{-1}, where U[0]= signma_11^{1/2}
		LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', n[0], n[1], U[0].Start(), n[0], S12.Start(), n[0]);
        //kl: so now S12 = sigma_11^{-1/2}*sigma_12*sigma_22^{-1/2} = T in the paper
        
		int k = min(n[0], n[1]);
		_w[0].Resize(n[0], k);
		_w[1].Resize(k, n[1]);
		AllocatingVector singularValues(k);
		AllocatingVector superb(k);
        //kl: SVD on T in the paper to get U and Vt, U=_w[0], Vt=_w[1]
		int info = LAPACKE_dgesvd(CblasColMajor, 'S', 'S', n[0], n[1], S12.Start(), n[0], singularValues.Start(),
                                  _w[0].Start(), n[0], _w[1].Start(), k, superb.Start());
        
        
		if (info != 0) {
			cout << "dgesvd returned error code " << info << endl;
		}
      
        
        
        _w[1].TransposeInPlace(); //kl: transpose of _w[0], since V is already Vt, so _w[1] not necessary to be transposed.
        
        vector<string> outputW(2);
        outputW[0] = "outputData2/W_Out1_noCentered.dat";
        outputW[1] = "outputData2/W_Out2_noCentered.dat";
        
        
        for(int i=0;i<2;i++){
            
            _w[i].WriteToFile(outputW[i]);
            cout<<"_W "<<i<< "'s NumC: "<< _w[i].NumC()<<endl;
            cout<<"_W "<<i<< "'s NumR: "<< _w[i].NumR()<<endl;
        }
        
        cout<<"W1 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(_w[0].Start()+k)<<" ";
        }
        cout<<endl;
        
        cout<<"W2 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(_w[1].Start()+k)<<" ";
        }
        cout<<endl;
        
        
        
        //kl: _w[i] = U[i]^{-1}*_w[i] = sigma_11^{-1/2}*Ut or sigma_22^{-1/2}*Vt, the A1_*, A2_* in page2
		for (int i = 0; i < 2; ++i) {
            LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', k, n[i], U[i].Start(), n[i], _w[i].Start(), k);
            //cout <<"A's NumC: "<< _w[i].NumC()<< endl;
            //cout << "A's NumR: "<< _w[i].NumR()<< endl;
        }
        /*for(int i=0;i<2;i++){
         _w[i].TransposeInPlace();
         }*/
        int kp = 50;
        for(int i=0;i<2;i++){
            _w[i].Resize(n[i], kp);
            
            _w[i].CopyFrom(_w[i].SubMatrix(0, kp));
            _w[i].TransposeInPlace();
            cout<<"_A "<<i<< "'s NumC: "<< _w[i].NumC()<<endl;
            cout<<"_A "<<i<< "'s NumR: "<< _w[i].NumR()<<endl;
        }
        
        
        for(int i=0;i<2;i++){
            
            //_w[i].WriteToFile(outputW[i]);
            cout<<"A "<<i<< "'s NumC: "<< _w[i].NumC()<<endl;
            cout<<"A "<<i<< "'s NumR: "<< _w[i].NumR()<<endl;
        }
        vector<string> outputA(2);
        outputA[0] = "outputData2/A_Out1_noCentered.dat";
        outputA[1] = "outputData2/A_Out2_noCentered.dat";
        
        for(int i=0;i<2;i++){
            
            _w[i].WriteToFile(outputA[i]);
        }
        
        cout<<"A1 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(_w[0].Start()+k)<<" ";
        }
        cout<<endl;
        cout<<"A2 start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(_w[1].Start()+k)<<" ";
        }
        
        cout<<endl;
        cout<<"X1 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(X[0].Start()+k)<<" ";
        }
        
        cout<<endl;
        cout<<"X2 start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(X[1].Start()+k)<<" ";
        }
        
        vector<string> outputData(2);
        outputData[0] = "outputData2/trainOut1_noCentered.dat";
        outputData[1] = "outputData2/trainOut2_noCentered.dat";
        
        AllocatingMatrix mapped1;
        AllocatingMatrix mapped2;
        
        mapped1.Resize(_w[0].NumR(), X[0].NumC());
		mapped1.AddProd(_w[0], false, X[0], false, 1.0, 0.0);
        
        cout<<endl;
        cout<<"mapped1 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(mapped1.Start()+k)<<" ";
        }
        
        cout<<endl;
        cout <<"Trainview "<<0<<" output's NumC: "<< mapped1.NumC()<< endl;
        cout << "Train view "<<0<<" output's NumR: "<< mapped1.NumR()<< endl;
        
        mapped1.WriteToFile(outputData[0]);
        
        
        mapped2.Resize(_w[1].NumR(), X[1].NumC());
		mapped2.AddProd(_w[1], false, X[1], false, 1.0, 0.0);
        
        cout<<"mapped2 start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(mapped2.Start()+k)<<" ";
        }
        cout<<endl;
        cout <<"view "<<1<<" output's NumC: "<< mapped2.NumC()<< endl;
        cout << "view "<<1<<" output's NumR: "<< mapped2.NumR()<< endl;
        mapped2.WriteToFile(outputData[1]);
        
		return singularValues.Sum();
	}
    
    //kl: supervised Pipeline CCA
    template <class ArrayOfMatrices>
    double supPipelineCCADoc(const ArrayOfMatrices & X, const ArrayOfMatrices & Y, double reg1 = 0, double reg2 = NaN) {
		//kl: in X are the traindata, which has trainData[0] in X[0], trainData[1] in X[0] are train.en.tok.data and train.es.tok.data, however, trainData[2] in X[2] stores the labeldata for English for superCCA
        //kl: so do the two steps Pipeline CCA in this one method called.
        
        if (IsNaN(reg2)) reg2 = reg1;
        
        // kl: NumC is the number of the instances; NumR is the number of features;
		int m = X[0].NumC();
        cout<< "Number of instance: "<< m <<endl;
		int n[] = { X[0].NumR(), X[2].NumR() }; //kl: in our case n[0] = 5000, n[1] = 50;
		assert (X[0].NumC() == m);
        
        
        
		double reg[] = { reg1, reg2 };
        
        // kl: supCCA(train.en.tok.data, 50dlabelVector.data)=> generate BNx for the new representation of train.en.tok.data
        
		vector<AllocatingMatrix> U(2);
        int z=0;
        // i = i+2, take the trainData[0](train.en.tok.data) and trainData[2](50dlabelVector.data) to do CCA firstly
		for (int i = 0; i < 3; i=i+2) {
			//GetMean(X[i], _mu[i]);
			//Translate(X[i], _mu[i], centered[i]); //kl: center X
            
			U[z].Resize(n[z], n[z]); //kl: square matrix
			U[z].Syrk(X[i], 1.0 / (m - 1), 0.0); //kl: get covariance matrix in U
			double thisReg = max(reg[z] / m, 1e-8);
			for (int r = 0; r < n[z]; ++r) U[z].At(r,r) += thisReg; //kl: add regularization to the covariance matrix
            //kl: DPOTRF computes the Cholesky factorization of a real symmetric positive definite matrix A.
            //kl: Upper triangle of U[i] is stored into the updated U[i] = Sigma_ii^{1/2} in the paper
			LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n[z], U[z].Start(), n[z]);
            z = z + 1;
		}
        
        cout<<"931 debugging"<<endl;
        
		AllocatingMatrix S12(n[0], n[1]);
		S12.AddProd(X[0], false, X[2], true, 1.0 / (m-1), 0.0); //kl: covariance matrix in S12
        
       
        
		// set S12 = U[0]^{-1}' S12 U[1]^{-1}
        //kl: right-multiply U[1]^{-1}, where U[1]= signma_22^{1/2}
		LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N', n[1], n[0], U[1].Start(), n[1], S12.Start(), n[0]);
        //kl: left-multiply U[0]^{-1}, where U[0]= signma_11^{1/2}
		LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', n[0], n[1], U[0].Start(), n[0], S12.Start(), n[0]);
        //kl: so now S12 = sigma_11^{-1/2}*sigma_12*sigma_22^{-1/2} = T in the paper
        
        cout<<"945 debugging"<<endl;
        
		int k = min(n[0], n[1]); //kl: k = min(5000, 50) = 50;
		_w[0].Resize(n[0], k);
		_w[1].Resize(k, n[1]);
		AllocatingVector singularValues(k);
		AllocatingVector superb(k);
        //kl: SVD on T in the paper to get U and Vt, U=_w[0], Vt=_w[1]
		int info = LAPACKE_dgesvd(CblasColMajor, 'S', 'S', n[0], n[1], S12.Start(), n[0], singularValues.Start(),
                                  _w[0].Start(), n[0], _w[1].Start(), k, superb.Start());
        
        cout<<"956 debugging"<<endl;
		if (info != 0) {
			cout << "dgesvd returned error code " << info << endl;
		}
                
        _w[1].TransposeInPlace(); //kl: transpose of _w[0], since V is already Vt, so _w[1] not necessary to be transposed.
        
                
        
        //kl: _w[i] = U[i]^{-1}*_w[i] = sigma_11^{-1/2}*Ut or sigma_22^{-1/2}*Vt, the A1_*, A2_* in page2
		for (int i = 0; i < 2; ++i) {
            LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', k, n[i], U[i].Start(), n[i], _w[i].Start(), k);
            //cout <<"A's NumC: "<< _w[i].NumC()<< endl;
            //cout << "A's NumR: "<< _w[i].NumR()<< endl;
        }
        for(int i=0;i<2;i++){
           _w[i].TransposeInPlace();
         }
        cout<<"974 debugging"<<endl;
        /*int kp = 50;
        for(int i=0;i<2;i++){
            _w[i].Resize(n[i], kp);
            
            _w[i].CopyFrom(_w[i].SubMatrix(0, kp));
            _w[i].TransposeInPlace();
            cout<<"_A "<<i<< "'s NumC: "<< _w[i].NumC()<<endl;
            cout<<"_A "<<i<< "'s NumR: "<< _w[i].NumR()<<endl;
        }*/
                
        //kl: generate train outputs
        vector<string> outputData(2);
        outputData[0] = "outputData/trainEN.BNx_noCentered.dat";
        outputData[1] = "outputData/trainEn.testBNx_noCentered.dat";
        
        AllocatingMatrix BNx;
        AllocatingMatrix testEnBNx;
        
        BNx.Resize(_w[0].NumR(), X[0].NumC());
		BNx.AddProd(_w[0], false, X[0], false, 1.0, 0.0);
        
        cout<<"BNx start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(BNx.Start()+k)<<" ";
        }
        
        cout<<endl;
        cout <<"BNx "<<0<<" output's NumC: "<< BNx.NumC()<< endl;
        cout << "BNx "<<0<<" output's NumR: "<< BNx.NumR()<< endl;
        
        BNx.WriteToFile(outputData[0]);
        
        testEnBNx.Resize(_w[0].NumR(), Y[0].NumC());
		testEnBNx.AddProd(_w[0], false, Y[0], false, 1.0, 0.0);
        
        testEnBNx.WriteToFile(outputData[1]);
        
        cout<<"testEnBNx start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(testEnBNx.Start()+k)<<" ";
        }
        
        vector<string> outputWLDA(2);
        outputWLDA[0] = "outputData/w_LDA1_noCentered.dat";
        outputWLDA[1] = "outputData/w_LDA2_noCentered.dat";
        
        for(int i=0;i<2;i++){
            
            _w[i].WriteToFile(outputWLDA[i]);
        }
        
        
        
        
        
        
        //kl: now mapped1 stores the new representation （BNx） of train.en.tok.data after supCCA(train.en.tok.data, 50dlabelVector.data)
        //kl: apply 2nd CCA on BNx of English generated from (CCA(train.en.tok.data, 50dlabelVector.data)) and train.es.tok.data to generate final representations
        //kl: concatenate the train.en. from X[0] and BNx
        AllocatingMatrix BNy;
        AllocatingMatrix temp;
        temp = BNx;
        temp.TransposeInPlace();
        //BNx = temp.SubMatrix(0, 25);
        
        BNy = temp.SubMatrix(0, 25);
        //BNx.TransposeInPlace();
        BNy.TransposeInPlace();
        temp.TransposeInPlace();
        AllocatingMatrix concateBNx = concatenateMatrix(X[0], BNx);
        
        /*double* pdata = concateBNx.Start();
        
        //cout << maxCols << "\t" << numR << endl;
        for(int i = 0; i < concateBNx.NumC(); i++) {
            for (int j = 0; j < concateBNx.NumR(); j++)
                cout << pdata[i * concateBNx.NumR() + j] << " ";
            cout << endl;
        }*/
        concateBNx.WriteToFile("outputData/concateBNx.dat");
        
        //int m2 = BNx.NumC();
        int m2 = concateBNx.NumC();
        cout<< "Number of instance: "<< m2 <<endl;
		//int n2[] = { BNx.NumR(), X[1].NumR() };
        int n2[] = { concateBNx.NumR(), X[1].NumR() };
		assert(X[1].NumC() == m2);
        vector<AllocatingMatrix> U2(2);
        
		
            
			U2[0].Resize(n2[0], n2[0]); //kl: square matrix
			//U2[0].Syrk(BNx, 1.0 / (m2 - 1), 0.0); //kl: get covariance matrix in U
            U2[0].Syrk(concateBNx, 1.0 / (m2 - 1), 0.0);
			double thisReg = max(reg[0] / m2, 1e-8);
			for (int r = 0; r < n2[0]; ++r) U2[0].At(r,r) += thisReg; //kl: add regularization to the covariance matrix
            //kl: DPOTRF computes the Cholesky factorization of a real symmetric positive definite matrix A.
            //kl: Upper triangle of U[i] is stored into the updated U[i] = Sigma_ii^{1/2} in the paper
			LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n2[0], U2[0].Start(), n2[0]);
		
            U2[1].Resize(n2[1], n2[1]); //kl: square matrix
            U2[1].Syrk(X[1], 1.0 / (m2 - 1), 0.0); //kl: get covariance matrix in U
            double thisReg2 = max(reg[1] / m2, 1e-8);
            for (int r = 0; r < n2[1]; ++r) U2[1].At(r,r) += thisReg2; //kl: add regularization to the covariance matrix
            //kl: DPOTRF computes the Cholesky factorization of a real symmetric positive definite matrix A.
            //kl: Upper triangle of U[i] is stored into the updated U[i] = Sigma_ii^{1/2} in the paper
            LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', n2[1], U2[1].Start(), n2[1]);
        
        
		AllocatingMatrix S12_2(n2[0], n2[1]);
		//S12_2.AddProd(BNx, false, X[1], true, 1.0 / (m2-1), 0.0); //kl: covariance matrix in S12
        S12_2.AddProd(concateBNx, false, X[1], true, 1.0 / (m2-1), 0.0);
        
        
		// set S12 = U[0]^{-1}' S12 U[1]^{-1}
        //kl: right-multiply U[1]^{-1}, where U[1]= signma_22^{1/2}
		LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N', n2[1], n2[0], U2[1].Start(), n2[1], S12_2.Start(), n2[0]);
        //kl: left-multiply U[0]^{-1}, where U[0]= signma_11^{1/2}
		LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'T', 'N', n2[0], n2[1], U2[0].Start(), n2[0], S12_2.Start(), n2[0]);
        //kl: so now S12 = sigma_11^{-1/2}*sigma_12*sigma_22^{-1/2} = T in the paper
        
		int k2 = min(n2[0], n2[1]);
		_w[0].Resize(n2[0], k2);
		_w[1].Resize(k2, n2[1]);
		AllocatingVector singularValues2(k2);
		AllocatingVector superb2(k2);
        cout<<"1087 debugging"<<endl;
        //kl: SVD on T in the paper to get U and Vt, U=_w[0], Vt=_w[1]
		int info2 = LAPACKE_dgesvd(CblasColMajor, 'S', 'S', n2[0], n2[1], S12_2.Start(), n2[0], singularValues2.Start(),
                                  _w[0].Start(), n2[0], _w[1].Start(), k2, superb2.Start());
        
        //kl: print out the singular values
        cout << "singular values print start"<< endl;
        double* sing = singularValues2.Start();
         
         //cout << maxCols << "\t" << numR << endl;
         for(int i = 0; i < singularValues2.Len(); i++) {
         
            cout << sing[i] << " ";
            
         }
        cout<< endl;
        cout << "singlar values print end"<<endl;
        
        
        _w[1].TransposeInPlace(); //kl: transpose of _w[0], since V is already Vt, so _w[1] not necessary to be transposed.
        
        
        
        //kl: _w[i] = U[i]^{-1}*_w[i] = sigma_11^{-1/2}*Ut or sigma_22^{-1/2}*Vt, the A1_*, A2_* in page2
         //kl: U2[i] is the upper triangle matrix, is consistent to the DCCA paper and Raman's handwriting.
         //kl: which means so far the changes to the original CCA code is correct. 
		for (int i = 0; i < 2; ++i) {
            LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', k2, n2[i], U2[i].Start(), n2[i], _w[i].Start(), k2);
            //cout <<"A's NumC: "<< _w[i].NumC()<< endl;
            //cout << "A's NumR: "<< _w[i].NumR()<< endl;
        }
        for(int i=0;i<2;i++){
           _w[i].TransposeInPlace();
         }
        //kl: below code is commented out, because right now BNx is 50 dimensional, the output is 50 dimensional not necessary to extract 
        /*int kp2 = 50;
        for(int i=0;i<2;i++){
            _w[i].Resize(n2[i], kp2);
            
            _w[i].CopyFrom(_w[i].SubMatrix(0, kp2));
            _w[i].TransposeInPlace();
            cout<<"_A "<<i<< "'s NumC: "<< _w[i].NumC()<<endl;
            cout<<"_A "<<i<< "'s NumR: "<< _w[i].NumR()<<endl;
        }*/
        
        //kl: generate train outputs for the 2nd CCA(train.en.tok.data, labels)
        vector<string> outputData2(2);
        outputData2[0] = "outputData/trainPipeOutput1_noCentered.dat";
        outputData2[1] = "outputData/trainPipeOutput2_noCentered.dat";
        
        AllocatingMatrix mapped1;
        AllocatingMatrix mapped2;
        
        //mapped1.Resize(_w[0].NumR(), BNx.NumC());
		//mapped1.AddProd(_w[0], false, BNx, false, 1.0, 0.0);
        mapped1.Resize(_w[0].NumR(), concateBNx.NumC());
		mapped1.AddProd(_w[0], false, concateBNx, false, 1.0, 0.0);
        
        cout<<"mapped1 start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(mapped1.Start()+k)<<" ";
        }
        
        cout<<endl;
        cout <<"Trainview "<<0<<" output's NumC: "<< mapped1.NumC()<< endl;
        cout << "Train view "<<0<<" output's NumR: "<< mapped1.NumR()<< endl;
        
        mapped1.WriteToFile(outputData2[0]);
        
        mapped2.Resize(_w[1].NumR(), X[1].NumC());
		mapped2.AddProd(_w[1], false, X[1], false, 1.0, 0.0);
        
        cout<<"mapped2 start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(mapped2.Start()+k)<<" ";
        }
        cout<<endl;
        cout <<"view "<<1<<" output's NumC: "<< mapped2.NumC()<< endl;
        cout << "view "<<1<<" output's NumR: "<< mapped2.NumR()<< endl;
        mapped2.WriteToFile(outputData2[1]);
        
        cout<<"A1 start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(_w[0].Start()+k)<<" ";
        }
        cout<<endl;
        cout<<"A2 start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(_w[1].Start()+k)<<" ";
        }
        
        vector<string> outputW(2);
        outputW[0] = "outputData/w_Out1_noCentered.dat";
        outputW[1] = "outputData/w_Out2_noCentered.dat";
        
        for(int i=0;i<2;i++){
            
            _w[i].WriteToFile(outputW[i]);
        }
        
        
        
        //kl: generate the test outputs
        vector<string> outputDataTest(2);
        outputDataTest[0] = "outputData/testOut1_noCentered.dat";
        outputDataTest[1] = "outputData/testOut2_noCentered.dat";
        
        AllocatingMatrix mapped1Test;
        AllocatingMatrix mapped2Test;
        
        //kl: use the concatenated (Y[0], testEnBNx)
        AllocatingMatrix concateTestBNx = concatenateMatrix(Y[0],testEnBNx);
        //mapped1Test.Resize(_w[0].NumR(), testEnBNx.NumC());
		//mapped1Test.AddProd(_w[0], false, testEnBNx, false, 1.0, 0.0);
        mapped1Test.Resize(_w[0].NumR(), concateTestBNx.NumC());
		mapped1Test.AddProd(_w[0], false, concateTestBNx, false, 1.0, 0.0);
        
        cout<<endl;
        cout<<"mapped1Test start:"<<endl;
        for(int k=0;k<=50;k++){
            cout<<*(mapped1Test.Start()+k)<<" ";
        }
        cout<<endl;
        cout <<"Testview "<<0<<" output's NumC: "<< mapped1Test.NumC()<< endl;
        cout << "Testview "<<0<<" output's NumR: "<< mapped1Test.NumR()<< endl;
        
        mapped1Test.WriteToFile(outputDataTest[0]);
        
        
        mapped2Test.Resize(_w[1].NumR(), Y[1].NumC());
		mapped2Test.AddProd(_w[1], false, Y[1], false, 1.0, 0.0);
        
        cout<<"mapped2Test start:"<<endl;
        
        for(int k=0;k<=50;k++){
            cout<<*(mapped2Test.Start()+k)<<" ";
        }
        cout<<endl;
        cout <<"Testview "<<1<<" output's NumC: "<< mapped2Test.NumC()<< endl;
        cout << "Testview "<<1<<" output's NumR: "<< mapped2Test.NumR()<< endl;
        mapped2Test.WriteToFile(outputDataTest[1]);
        
        
                
		return singularValues2.Sum();
	}
    
    //kl: this method is to concatenate two matrices 
    AllocatingMatrix concatenateMatrix(const AllocatingMatrix & mat1, const AllocatingMatrix & mat2) const{
        
        //kl: concatenate two matrices by columns the same, but adding rows
        
        int numR1 = mat1.NumR();
        int numC1 = mat1.NumC();
        int numR2 = mat2.NumR();
        int numC2 = mat2.NumC();
        
        //kl: numC: trainSize
        assert (numC1 == numC2);
        
        AllocatingMatrix newMatrix;
        newMatrix.Resize(numR1+numR2, numC1);
        
        for (int i=0;i<numC1;i++){
            
            //kl: copy the first matrix
            for (int j=0;j<numR1;j++){
                *(newMatrix.Start()+i*(numR1+numR2)+j) = *(mat1.Start()+i*numR1+j);
            }
           
            //kl: stack the 2nd matrix below the first one
            for (int h=0;h<numR2;h++){
                *(newMatrix.Start()+i*(numR1+numR2)+numR1+h) = *(mat2.Start()+i*numR2+h);
            }
            
        }
        
        return newMatrix;
    }

};
