//
//  DBN_bottleneck.cpp
//  DCCA
//
//  Created by gflfof gflfof on 14-8-4.
//  Copyright (c) 2014年 hit. All rights reserved.
//

#include <iostream>
#include "DBN_bottleneck.h"
using namespace std;

static void DBN_bottleneck::DistRanges(const Vector & v, int numBins) {
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
    
	void DBN_bottleneck::PretrainLayerAE(int layer, const Matrix & train, MutableVector & layerParams, Random & rand, double pretrainL2, double noiseLevel, bool quiet, TrainModifiers trainModifiers) {
		if (!quiet) {
			cout << "AE Pretraining layer " << layer << " of size " << LayerInSize(layer) << "x" << LayerOutSize(layer) << endl;
		}
        
		Distorter distorter(noiseLevel, rand);
        
		AETrainingFunction<Distorter> aeFunc(*this, train, layer, pretrainL2, true, distorter);
        
		LBFGS opt(quiet);
		opt.Minimize(aeFunc, layerParams, layerParams, trainModifiers.LBFGS_tol, trainModifiers.LBFGS_M, trainModifiers.testGrad);
        
		SetReadParamsForLayer(layerParams, layer);
	}
    
	void DBN_bottleneck::Deserialize(istream & inStream) { //todo
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
    
	void DBN_bottleneck::Serialize(ostream & outStream) const { // todo
		outStream.write((const char *) &_iSize, sizeof(int));
		outStream.write((const char *) &_hSize, sizeof(int));
		outStream.write((const char *) &_oSize, sizeof(int));
		outStream.write((const char *) &_numLayers, sizeof(int));
		outStream.write((const char *) &_hActType, sizeof(Layer::ActivationType));
		outStream.write((const char *) &_oActType, sizeof(Layer::ActivationType));
	}
	
	int DBN_bottleneck::NumParamsInLayer(int layer) const {  // done
		assert (layer < _numLayers);
		if (_numLayers == 1) return (_iSize + 1) * _oSize;
		else if (layer == 0) return (_iSize + 1) * _hSizes[layer];
		else if (layer == _numLayers - 1) return (_hSizes[layer - 1] + 1) * _oSize;
		else return (_hSizes[layer - 1] + 1) * _hSizes[layer];
	}
    
	int DBN_bottleneck::LayerOutSize(int layer) const { // done
		assert (layer < _numLayers);
		return (layer < _numLayers - 1) ? _hSizes[layer] : _oSize;
	}
	
	int DBN_bottleneck::LayerInSize(int layer) const { // done
		assert (layer < _numLayers);
		return (layer == 0) ? _iSize : _hSizes[layer - 1];
	}
	
	void DBN_bottleneck::SetReadParams(const Vector & params, int numLayers = -1) { //done
		if (numLayers == -1 || numLayers > _numLayers) numLayers = _numLayers;
        
		int bSize = _iSize;
		int start = 0;
		for (int i = 0; i < numLayers; ++i) {
			int tSize = (i == _numLayers - 1) ? _oSize : _hSizes[layer];
			_readW[i] = params.SubVector(start, start + bSize * tSize).AsMatrix(bSize, tSize);
			start += bSize * tSize;
			_readB[i] = params.SubVector(start, start + tSize);
			start += tSize;
			bSize = tSize;
		}
        
		assert (start == params.Len());
	}
    
	void DBN_bottleneck::SetWriteParams(MutableVector params, int numLayers = -1) {
		if (numLayers == -1 || numLayers > _numLayers) numLayers = _numLayers;
        
		int bSize = _iSize;
		int start = 0;
		for (int i = 0; i < numLayers; ++i) {
			int tSize = (i == _numLayers - 1) ? _oSize : _hSizes[layer];
			_writeW[i] = params.SubVector(start, start + bSize * tSize).AsMatrix(bSize, tSize);
			start += bSize * tSize;
			_writeB[i] = params.SubVector(start, start + tSize);
			start += tSize;
			bSize = tSize;
		}
        
		assert (start == params.Len());
	}
    
	void DBN_bottleneck::SetReadParamsForLayer(const Vector & params, int layer) { //done
		int bSize = layer == 0 ? _iSize : _hSizes[layer - 1];
		int tSize = (layer == _numLayers - 1) ? _oSize : _hSizes[layer];
		assert (params.Len() == (bSize + 1) * tSize);
		_readW[layer] = params.SubVector(0, bSize * tSize).AsMatrix(bSize, tSize);
		_readB[layer] = params.SubVector(bSize * tSize, (bSize + 1) * tSize);
	}
    
	void DBN_bottleneck::SetWriteParamsForLayer(MutableVector & params, int layer) { //done
		int bSize = layer == 0 ? _iSize : _hSizes[layer - 1];
		int tSize = (layer == _numLayers - 1) ? _oSize : _hSizes[layer];
		assert (params.Len() == (bSize + 1) * tSize);
		_writeW[layer] = params.SubVector(0, bSize * tSize).AsMatrix(bSize, tSize);
		_writeB[layer] = params.SubVector(bSize * tSize, (bSize + 1) * tSize);
	}
    
	void DBN_bottleneck::BackProp(const Matrix & input, const Matrix & errors, int numLayers = -1) {
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
    
	void DBN_bottleneck::BackPropLayer(const Matrix & layerInput, const Matrix & errors, int layer) {
		ASSERT (layer >= 0 && layer < _numLayers);
        
		Matrix mappedError = _layers[layer].ComputeErrors(errors);
		for (int i = 0; i < errors.NumC(); ++i) _writeB[layer] += mappedError.GetCol(i);
		_writeW[layer].AddProd(layerInput, false, mappedError, true);
	}
	
	double DBN_bottleneck::RegularizeLayer(int layer, double alpha) {
		Matrix weights = _readW[layer];
		if (_writeW[layer].Len() > 0) _writeW[layer].Add(weights, alpha);
		return 0.5 * alpha * Vector::DotProduct(weights, weights);
	}
    
	double DBN_bottleneck::Regularize(double alpha, int numLayers = -1) {
		if (numLayers == -1 || numLayers > _numLayers) numLayers = _numLayers;
        
		double val = 0;
        
		for (int layer = 0; layer < numLayers; ++layer) {
			Matrix weights = _readW[layer];
			val += Vector::DotProduct(weights, weights);
			if (_writeW[layer].Len() > 0) _writeW[layer].Add(weights, alpha);
		}
        
		return 0.5 * alpha * val;
	}
    
	Matrix DBN_bottleneck::MapLayer(const Matrix & input, int layer) const {
		Layer::ActivationType actType = (layer == _numLayers - 1) ? _oActType : _hActType;
		return _layers[layer].ActivateUp(_readW[layer], _readB[layer], input, actType);
	}
    
	Matrix DBN_bottleneck::MapUp(const Matrix & input, int numLayers = -1) const {
		if (numLayers == -1 || numLayers > _numLayers) numLayers = _numLayers;
        
		Matrix mappedInput = input;
		
		for (int l = 0; l < numLayers; ++l) {
			Layer::ActivationType actType = (l == numLayers - 1) ? _oActType : _hActType;
			mappedInput = _layers[l].ActivateUp(_readW[l], _readB[l], mappedInput, actType);
		}
        
		return mappedInput;
	}
	
	double DBN_bottleneck::UpdateAE(const Matrix & input, const Vector & inputBiases, const Matrix & targetInput, int layer) {		
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
    
	void DBN_bottleneck::Pretrain(const Matrix & input, MutableVector params, Random & rand, bool quiet, PretrainHyperParams hyperParams, TrainModifiers trainModifiers) { //no need
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
