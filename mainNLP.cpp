//
//  mainNLP.cpp
//  DeepCCA
//
//  Created by Kai Liu on 8/28/14.
//  Copyright (c) 2014 Kai Liu. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <vector>

#include "DeepCCAModel.h"
#include "HyperParams.h"
#include "ProgramArgs.h"

using namespace std;

void ParseArgCCA(string featAndVal,int featNum, int lineNum, AllocatingMatrix & mat) {
    if (featAndVal.size() == 0) return;
    
    //cout << featAndVal << endl;
    
    istringstream iss(featAndVal);
    string tempFeatAndVal;
    
    // everytime, output a "feature:value" pair, put into the matrix
    while(iss >> tempFeatAndVal){
        
        int eq = tempFeatAndVal.find(':');
        if (eq == string::npos) {
            cerr << "Couldn't find value for line: " << tempFeatAndVal << endl;
            exit(1);
        }
        string feature = tempFeatAndVal.substr(0, eq);
        string val = tempFeatAndVal.substr(eq + 1);
        
        int feat;
        stringstream convert(feature);
        if ( !(convert >> feat) )
            feat = 0;
        
        int value;
        stringstream convert2(val);
        if(!(convert2 >> value))
            value = 0;
        
        *(mat.Start() + lineNum*featNum + feat-1) = (double) value;
    }
    
}


void LoadData(const ProgramArgs & args, vector<AllocatingMatrix> & trainData) {
    
    for (int k=0;k<2;k++){
        ifstream argsFile(args.inData[k]);
        if (!argsFile.is_open()) {
            cerr << "Couldn't open " << args.inData[k] << endl;
            exit(1);
        }
        int featNum = args.iSize[k]; // feature number
        int instNum = args.trainSize; //instances number
        trainData[k].Resize(featNum, instNum);
        for(int i=0;i<featNum*instNum;i++){
            *(trainData[k].Start()+i) = 0;
        }
        
        int lineNum = 0;
        while (argsFile.good()) {
            string line;
            getline(argsFile, line);
            
            ParseArgCCA(line, featNum, lineNum, trainData[k]);
            lineNum++;
        }
        
        if(lineNum == 1029) cout<<"data input finished!"<<endl;
        
        double* pdata = trainData[k].Start();
        
        for(int i = 0; i < instNum; i++) {
            for (int j = 0; j < featNum; j++)
                cout << pdata[i * featNum + j] << " ";
            cout << "one instance end" << endl;
        }
        
        argsFile.close();
    }
}



double Map(const DeepCCAModel & model, const vector<AllocatingMatrix> & data, const string outputData[]) {
	vector<AllocatingMatrix> mapped(2);
    
	for (int v = 0; v < 2; ++v) {
		if (data[v].Len() > 0) {
            //kl: MapUp for the AutoEncoder only the first view 
           
            model.MapUpAE(data[v], mapped[v], v);
             
			if (outputData[v].size() > 0) mapped[v].WriteToFile(outputData[v]);
		}
	}
	
	return CCA::TestCorr(mapped);
}

void Train(const ProgramArgs & args, DeepCCAModel & model, const vector<AllocatingMatrix> & trainData) {
	DCCAHyperParams hyperParams;
	Deserialize(hyperParams, args.inParams);
    
	// override params with command line if specified
	for (int v = 0; v < 2; ++v) {
		if (args.hSize[v] > 0) hyperParams.params[v].layerWidthH = args.hSize[v];
	}
    
	cout << endl << "Hyperparams: " << endl;
	hyperParams.Print();
	cout << endl;
    
	Random rand;
    
	TrainModifiers pretrainModifiers = TrainModifiers::LBFGSModifiers(1e-3, 15, false);
	TrainModifiers finetuneModifiers = TrainModifiers::LBFGSModifiers(1e-4, 15, false);
    
	double corr = model.Train(hyperParams, args.numLayers, args.inFeatSelect, args.outputSize, trainData, pretrainModifiers, finetuneModifiers, rand);
    
	cout << "Regularized train DCCA corr: " << corr << endl;
}

int main(int argc, char** argv) {
    // kl: argc is the string number in the excute line, while argv is the array for all the words
	ProgramArgs args(argc, argv);
	
	vector<AllocatingMatrix> trainData(2);
    
	DeepCCAModel model;
	if (args.inModel.size() > 0) { // kl: after training, the model has been already generated.
		cout << "Reading model from " << args.inModel << endl;
		Deserialize(model, args.inModel); //kl: readin the model which has been generated during training
        cout<<"modle is fine"<<endl;
		for (int v = 0; v < 2; ++v) args.iSize[v] = 5000; //model.InputSize(v);
		LoadData(args, trainData);
	} else { //kl: without model, have to use the training data to train a model firstly
		LoadData(args, trainData);
		Train(args, model, trainData);
        
		if (args.outModel.size() > 0) {
			Serialize(model, args.outModel); //kl: store the model in output Model for testing
		}
	}
    
	double corr = Map(model, trainData, args.outData);
	if (!IsNaN(corr)) cout << "Total DCCA corr: " << corr << endl;
    
	return 0;
}
