#pragma once

#include <vector>
#include <algorithm>

#include "Layer.h"
#include "LBFGS.h"
#include "HyperParams.h"

using namespace std;

class DBN {
	static void DistRanges(const Vector & v, int numBins) {
		vector<double> vals(v.Start(), v.End());
		for (int i = 0; i < vals.size(); ++i) vals[i] = fabs(vals[i]);
		sort(vals.begin(), vals.end());
		for (int i = 0; i < numBins; ++i) {
			double binMin = vals[vals.size() * i / numBins];
			double binMax = vals[vals.size() * (i+1) / numBins - 1];
			printf("%-5d %-18.15f %-18.15f\n", i, binMin, binMax);
		}
		printf("\n");
	}

	int _iSize, _hSize, _oSize;
	int _numLayers;

	mutable vector<Layer> _layers;

	vector<Matrix> _readW; //kl: weight
	vector<Vector> _readB; //kl: bias
	vector<MutableMatrix> _writeW; //kl: weight
	vector<MutableVector> _writeB; //kl: bias

	mutable AllocatingMatrix _tempTopSample, _tempBottomSample;

	mutable AllocatingMatrix _tempMat;
	mutable AllocatingVector _tempVec;
	mutable Layer _tempTopLayer, _tempBottomLayer;
	Layer::ActivationType _hActType, _oActType;

	MutableMatrix _emptyMatrix;

    //kl: noise distorter
	class Distorter {
		double _noiseLevel;
		Random & _rand;

	public:
		Distorter(double noiseLevel, Random & rand) : _noiseLevel(noiseLevel), _rand(rand) { }

		double operator()(double x) {
			return x + _noiseLevel * _rand.Normal();
		}
	};
	
	class TrainingFunction : public DifferentiableFunction {
	protected:
		DBN & _dbn;
		double _alpha;
		Matrix _input, _output;
		int _numParams;
		AllocatingMatrix _tempInput, _tempOutput;

		TrainingFunction(DBN & dbn, const Matrix & input, const Matrix & output, double alpha, int numParams)
			:
		_dbn(dbn), _alpha(alpha), _input(input), _output(output), _numParams(numParams) { }
		
		virtual double PrivateEval(const Vector & params, MutableVector & gradient, Matrix inputMiniBatch, Matrix outputMiniBatch, bool regularize) = 0;

		MutableVector _emptyVector;

	public:
		int NumIns() { return _output.NumC(); } // kl: number of instances
		
		double Eval(const Vector & params) {
			return PrivateEval(params, _emptyVector, _input, _output, true);
		}

		double Eval(const Vector & params, MutableVector & gradient) {
			return PrivateEval(params, gradient, _input, _output, true);
		}

		int NumParams() const { return _numParams; } //kl: return number of parameters

        //kl: use the selected instances indicated by "indices" for PrivateEval()
		double Eval(const Vector & params, MutableVector & gradient, const vector<int> & indices) {
			int n = indices.size();
			if (_input.Len() > 0) _tempInput.Resize(_input.NumR(), n);
			_tempOutput.Resize(_output.NumR(), n);

			for (int i = 0; i < n; ++i) {
				if (_input.Len() > 0) _tempInput.GetCol(i).CopyFrom(_input.GetCol(indices[i]));
				_tempOutput.GetCol(i).CopyFrom(_output.GetCol(indices[i]));
			}
			
			return PrivateEval(params, gradient, _tempInput, _tempOutput, true);
		}

        //kl: minibatch case for Evaluation
		double Eval(const Vector & params, MutableVector & gradient, int start, int count) {
			int numIns = NumIns();
			if (count == -1) count = numIns;
			start %= numIns;

			Matrix inputMiniBatch, outputMiniBatch;
			int end = start + count;
			if (end <= numIns) {
				inputMiniBatch = (_input.Len() > 0) ? _input.SubMatrix(start, end) : _input;
				outputMiniBatch = _output.SubMatrix(start, end);
			} else {
				if (_input.Len() > 0) {
					_tempInput.Resize(_input.NumR(), count);
					_tempInput.SubMatrix(0, (numIns - start)).CopyFrom(_input.SubMatrix(start, -1));
					_tempInput.SubMatrix(numIns - start, -1).CopyFrom(_input.SubMatrix(0, end - numIns));
					inputMiniBatch = _tempInput;
				} else inputMiniBatch = _input;

				_tempOutput.Resize(_output.NumR(), count);
				_tempOutput.SubMatrix(0, (numIns - start)).CopyFrom(_output.SubMatrix(start, -1));
				_tempOutput.SubMatrix(numIns - start, -1).CopyFrom(_output.SubMatrix(0, end - numIns));
				outputMiniBatch = _tempOutput;
			}
			
			return PrivateEval(params, gradient, inputMiniBatch, outputMiniBatch, true);
		}
	};

    //kl: AutoEncoder Training Function inherited from TrainingFunction class
	template <class Distorter>
	class AETrainingFunction : public TrainingFunction {
		int _layer;
		AllocatingMatrix _distortedInput, _inputBiases;
		Distorter & _distorter;

	protected:
		virtual double PrivateEval(const Vector & params, MutableVector & gradient, Matrix inputMiniBatch, Matrix outputMiniBatch, bool regularize) {
			_dbn.SetReadParamsForLayer(params, _layer);

			if (gradient.Len() > 0) {
				gradient.Clear();
				_dbn.SetWriteParamsForLayer(gradient, _layer);
			} else {
				_dbn.ClearWriteParamsForLayer(_layer);
			}

			if (inputMiniBatch.Len() == 0) {
				_distortedInput.Resize(outputMiniBatch.NumR(), outputMiniBatch.NumC());
				_distortedInput.ApplyInto(outputMiniBatch, _distorter);
				inputMiniBatch = _distortedInput;
			}

			double val = _dbn.UpdateAE(inputMiniBatch, _inputBiases, outputMiniBatch, _layer);

			int m = outputMiniBatch.NumC();
			val /= m;
			gradient /= m;

			if (regularize) {
				double alphaEff = _alpha * m / NumIns();
				val += _dbn.RegularizeLayer(_layer, alphaEff);
			}

			return val + 1;
		}

	public:		
		AETrainingFunction(DBN & dbn, const Matrix & trainData, int layer, double alpha, bool fixTrainDistortion, Distorter & distorter)
			: TrainingFunction(dbn, Matrix::EmptyMatrix, trainData, alpha, dbn.NumParamsInLayer(layer)), _layer(layer),
			_distorter(distorter)
		{
			if (fixTrainDistortion) {
				_distortedInput.Resize(trainData);
				_distortedInput.ApplyInto(trainData, _distorter);
				_input = _distortedInput;
			}

			dbn.InitializeInputBiases(trainData, _inputBiases, 1e-3);
		}
	};
		
	void PretrainLayerAE(int layer, const Matrix & train, MutableVector & layerParams, Random & rand, double pretrainL2, double noiseLevel, bool quiet, TrainModifiers trainModifiers) {
		if (!quiet) {
			cout << "AE Pretraining layer " << layer << " of size " << LayerInSize(layer) << "x" << LayerOutSize(layer) << endl;
		}

		Distorter distorter(noiseLevel, rand);

		AETrainingFunction<Distorter> aeFunc(*this, train, layer, pretrainL2, true, distorter);

		LBFGS opt(quiet);
		opt.Minimize(aeFunc, layerParams, layerParams, trainModifiers.LBFGS_tol, trainModifiers.LBFGS_M, trainModifiers.testGrad);

		SetReadParamsForLayer(layerParams, layer);
	}

public:
	DBN() { }

	DBN(int numLayers, int iSize, int hSize, int oSize, Layer::ActivationType hActType, Layer::ActivationType oActType)
		:
	_iSize(iSize), _hSize(hSize), _oSize(oSize), _numLayers(numLayers), _layers(numLayers),
	_readW(numLayers), _readB(numLayers), _writeW(numLayers), _writeB(numLayers), _hActType(hActType), _oActType(oActType)
	{ }

	void Initialize(int numLayers, int iSize, int hSize, int oSize)
	{
		_iSize = iSize;
		_hSize = hSize;
		_oSize = oSize;
		_numLayers = numLayers;
		_layers.resize(numLayers);
		_readW.resize(numLayers);
		_readB.resize(numLayers);
		_writeW.resize(numLayers);
		_writeB.resize(numLayers);
		_hActType = _oActType = Layer::CUBIC;
	}

	void Deserialize(istream & inStream) {
		inStream.read((char *) &_iSize, sizeof(int));
		inStream.read((char *) &_hSize, sizeof(int));
		inStream.read((char *) &_oSize, sizeof(int));
		inStream.read((char *) &_numLayers, sizeof(int));
		inStream.read((char *) &_hActType, sizeof(Layer::ActivationType));
		inStream.read((char *) &_oActType, sizeof(Layer::ActivationType));

		_layers.resize(_numLayers);
		_readW.resize(_numLayers);
		_readB.resize(_numLayers);
		_writeW.resize(_numLayers);
		_writeB.resize(_numLayers);
	}

	void Serialize(ostream & outStream) const {
		outStream.write((const char *) &_iSize, sizeof(int));
		outStream.write((const char *) &_hSize, sizeof(int));
		outStream.write((const char *) &_oSize, sizeof(int));
		outStream.write((const char *) &_numLayers, sizeof(int));
		outStream.write((const char *) &_hActType, sizeof(Layer::ActivationType));
		outStream.write((const char *) &_oActType, sizeof(Layer::ActivationType));
	}

	int NumLayers() const { return _numLayers; }

	int NumParams(int numLayers = -1) const {
		if (numLayers == -1) numLayers = _numLayers;

		int numParams = 0;
		for (int layer = 0; layer < numLayers; ++layer) numParams += NumParamsInLayer(layer);

		return numParams;
	}
	
	int NumParamsInLayer(int layer) const {
		assert (layer < _numLayers);
		if (_numLayers == 1) return (_iSize + 1) * _oSize;
		else if (layer == 0) return (_iSize + 1) * _hSize;
		else if (layer == _numLayers - 1) return (_hSize + 1) * _oSize;
		else return (_hSize + 1) * _hSize;
	}

	int LayerOutSize(int layer) const {
		assert (layer < _numLayers);
		return (layer < _numLayers - 1) ? _hSize : _oSize;
	}
	
	int LayerInSize(int layer) const {
		assert (layer < _numLayers);
		return (layer == 0) ? _iSize : _hSize;
	}

	int InputSize() const { return _iSize; }
	
	void SetReadParams(const Vector & params, int numLayers = -1) {
		if (numLayers == -1 || numLayers > _numLayers) numLayers = _numLayers;

		int bSize = _iSize;
		int start = 0;
		for (int i = 0; i < numLayers; ++i) {
			int tSize = (i == _numLayers - 1) ? _oSize : _hSize;
			_readW[i] = params.SubVector(start, start + bSize * tSize).AsMatrix(bSize, tSize);
			start += bSize * tSize;
			_readB[i] = params.SubVector(start, start + tSize);
			start += tSize;
			bSize = tSize;
		}

		assert (start == params.Len());
	}

	void SetWriteParams(MutableVector params, int numLayers = -1) {
		if (numLayers == -1 || numLayers > _numLayers) numLayers = _numLayers;

		int bSize = _iSize;
		int start = 0;
		for (int i = 0; i < numLayers; ++i) {
			int tSize = (i == _numLayers - 1) ? _oSize : _hSize;
			_writeW[i] = params.SubVector(start, start + bSize * tSize).AsMatrix(bSize, tSize);
			start += bSize * tSize;
			_writeB[i] = params.SubVector(start, start + tSize);
			start += tSize;
			bSize = tSize;
		}

		assert (start == params.Len());
	}

	void ClearWriteParams() {
		for (int i = 0; i < _numLayers; ++i) {
			_writeW[i] = _emptyMatrix;
			_writeB[i] = _emptyMatrix;
		}
	}

	void SetReadParamsForLayer(const Vector & params, int layer) {
		int bSize = layer == 0 ? _iSize : _hSize;
		int tSize = (layer == _numLayers - 1) ? _oSize : _hSize;
		assert (params.Len() == (bSize + 1) * tSize);
		_readW[layer] = params.SubVector(0, bSize * tSize).AsMatrix(bSize, tSize);
		_readB[layer] = params.SubVector(bSize * tSize, (bSize + 1) * tSize);
	}

	void SetWriteParamsForLayer(MutableVector & params, int layer) {
		int bSize = layer == 0 ? _iSize : _hSize;
		int tSize = (layer == _numLayers - 1) ? _oSize : _hSize;
		assert (params.Len() == (bSize + 1) * tSize);
		_writeW[layer] = params.SubVector(0, bSize * tSize).AsMatrix(bSize, tSize);
		_writeB[layer] = params.SubVector(bSize * tSize, (bSize + 1) * tSize);
	}

	void ClearWriteParamsForLayer(int layer) {
		_writeW[layer] = _emptyMatrix;
	}

	void BackProp(const Matrix & input, const Matrix & errors, int numLayers = -1) {
		if (numLayers == -1 || numLayers > _numLayers) numLayers = _numLayers;
		if (numLayers == 0) return;

		int n = input.NumC();
		Matrix mappedError = _layers[numLayers - 1].ComputeErrors(errors);
		for (int l = numLayers - 1; ; --l) {
			for (int i = 0; i < n; ++i) _writeB[l] += mappedError.GetCol(i);

			const Matrix & lowerProbs = l > 0 ? _layers[l - 1].Activations() : input;

			_writeW[l].AddProd(lowerProbs, false, mappedError, true);

			if (l == 0) break;

			Layer::BackProp(_readW[l], mappedError, _tempMat);
			mappedError = _layers[l-1].ComputeErrors(_tempMat);
		}
	}

	void BackPropLayer(const Matrix & layerInput, const Matrix & errors, int layer) {
		ASSERT (layer >= 0 && layer < _numLayers);

		Matrix mappedError = _layers[layer].ComputeErrors(errors);
		for (int i = 0; i < errors.NumC(); ++i) _writeB[layer] += mappedError.GetCol(i);
		_writeW[layer].AddProd(layerInput, false, mappedError, true);
	}
	
	double RegularizeLayer(int layer, double alpha) {
		Matrix weights = _readW[layer];
		if (_writeW[layer].Len() > 0) _writeW[layer].Add(weights, alpha);
		return 0.5 * alpha * Vector::DotProduct(weights, weights);
	}

	double Regularize(double alpha, int numLayers = -1) {
		if (numLayers == -1 || numLayers > _numLayers) numLayers = _numLayers;

		double val = 0;

		for (int layer = 0; layer < numLayers; ++layer) {
			Matrix weights = _readW[layer];
			val += Vector::DotProduct(weights, weights);
			if (_writeW[layer].Len() > 0) _writeW[layer].Add(weights, alpha);
		}

		return 0.5 * alpha * val;
	}

	Matrix MapLayer(const Matrix & input, int layer) const {
		Layer::ActivationType actType = (layer == _numLayers - 1) ? _oActType : _hActType;
		return _layers[layer].ActivateUp(_readW[layer], _readB[layer], input, actType);
	}

	Matrix MapUp(const Matrix & input, int numLayers = -1) const {
		if (numLayers == -1 || numLayers > _numLayers) numLayers = _numLayers;

		Matrix mappedInput = input;
		
		for (int l = 0; l < numLayers; ++l) {
			Layer::ActivationType actType = (l == numLayers - 1) ? _oActType : _hActType;
			mappedInput = _layers[l].ActivateUp(_readW[l], _readB[l], mappedInput, actType);
		}

		return mappedInput;
	}

	static void InitializeInputBiases(const Matrix & input, AllocatingVector & inputBiases, double eps) {
		int n = input.NumC();
		inputBiases.Resize(input.NumR());
		inputBiases.MultInto(input, false, AllocatingVector(n, 1.0), 1.0/n);
	}
	
	AllocatingMatrix & GetTempMat() {
		return _tempMat;
	}
	
	double UpdateAE(const Matrix & input, const Vector & inputBiases, const Matrix & targetInput, int layer) {		
		Matrix weights = _readW[layer];
		Vector topBiases = _readB[layer];

		Layer & topLayer = _layers[layer];
		Matrix topProbs = topLayer.ActivateUp(weights, topBiases, input, _hActType);

		double negll = 0;

		Matrix bottomScores = _tempBottomLayer.ActivateDown(weights, inputBiases, topProbs, Layer::LINEAR);
		auto func = [&](double s, double x) {
			double t = s - x;
			negll += t * t;
		};
		bottomScores.ApplyConstWith(targetInput, func);
		negll /= 2;

		if (_writeW[layer].Len() > 0) {
			// fill bottomProbs with errors
			MutableMatrix & bottomProbs = _tempBottomLayer.Activations();
			bottomProbs -= targetInput;

			_writeW[layer].AddProd(bottomProbs, false, topProbs, true, 1.0, 1.0);

			topLayer.ReverseBackProp(weights, bottomProbs, _tempMat);
			Matrix topError = topLayer.ComputeErrors(_tempMat);

			_writeW[layer].AddProd(input, false, topError, true);

			_writeB[layer].Clear();
			for (int i = 0; i < input.NumC(); ++i) {
				_writeB[layer] += topError.GetCol(i);
			}
		}

		return negll;
	}

	void Pretrain(const Matrix & input, MutableVector params, Random & rand, bool quiet, PretrainHyperParams hyperParams, TrainModifiers trainModifiers) {
		if (params.Len() == 0) return;

		params.Apply([&](double x) { return rand.Normal() * 0.01; });

		Matrix mappedTrain = input;

		int startLayerParams = 0;

		for (int layer = 0; layer < _layers.size(); ++layer) {
			int numLayerParams = NumParamsInLayer(layer);
			MutableVector layerParams = params.SubVector(startLayerParams, startLayerParams + numLayerParams);

			double gaussianStdDev = (layer == 0) ? hyperParams.gaussianStdDevI : hyperParams.gaussianStdDevH;
			double pretrainL2 = (layer == 0) ? hyperParams.pretrainL2i : (layer == _layers.size() - 1) ? hyperParams.pretrainL2o : hyperParams.pretrainL2h;

			PretrainLayerAE(layer, mappedTrain, layerParams, rand, pretrainL2, gaussianStdDev, quiet, trainModifiers);

			mappedTrain = MapLayer(mappedTrain, layer);
			if (layer > 0) _layers[layer - 1].Clear();
		
			if (!quiet) {
				cout << "Activation quintile ranges" << endl;
				DistRanges(mappedTrain, 5);
			}

			startLayerParams += numLayerParams;
		}

		_layers.back().Clear();
		_tempMat.Resize(0,0);
	}
};
