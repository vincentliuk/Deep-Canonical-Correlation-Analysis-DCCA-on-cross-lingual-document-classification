#include <iostream>
#include <fstream>
#include <vector>

#include "DeepCCAModel.h"
#include "HyperParams.h"
#include "ProgramArgs.h"

using namespace std;

void ReadBin(const string & filename, AllocatingMatrix & mat, int numR, int maxCols = -1) {
	if (filename.size() == 0) return;

	ifstream inStream(filename, ios::in|ios::binary);
	if (!inStream.is_open()) {
		cout << "Couldn't open feature file " << filename.c_str() << endl;
		exit(1);
	}

	inStream.seekg(0, ios::end); // kl: seekg is used to move the position to the end of the file
	int endPos = inStream.tellg(); // kl:tellg is used to get the position in the stream after it has been moved with seekg to the end of the stream, therefore determining the size of the file.
	inStream.seekg(0, ios::beg); // kl: and then back to the beginning.
	ASSERT(endPos / sizeof(double) % numR == 0);
	int numC = (int)(endPos / sizeof(double) / numR);

    //kl: numC: trainSize
	if (maxCols != -1) numC = min(numC, maxCols);

	mat.Resize(numR, numC);
	inStream.read((char*)mat.Start(), numR * numC * sizeof(double));//kl: Extracts numR*numC*sizeof(double) characters from the stream and stores them in the array pointed by the pointer: mat.Start().
	inStream.close();
    
	cout << "Read " << filename << " of size " << numR << "x" << numC << endl;
}



void LoadData(const ProgramArgs & args, vector<AllocatingMatrix> & trainData) {
	
    for (int v = 0; v < 2; ++v) {
		ReadBin(args.inData[v], trainData[v], args.iSize[v], args.trainSize);
	}
    
    //kl mod: for aligning the label with one view's input
    //AlignLabel(args.inData[2],args.inData[1],trainData[1],args.iSize[1], args.trainSize);
    

	double corr = CCA::TestCorr(trainData); // kl: Just correlation
	if (!IsNaN(corr)) cout << "trainset linear corr: " << corr << endl << endl;
}

double Map(const DeepCCAModel & model, const vector<AllocatingMatrix> & data, const string outputData[], const ProgramArgs & args) {
	vector<AllocatingMatrix> mapped(2);

	for (int v = 0; v < 2; ++v) {
		if (data[v].Len() > 0) {
            //kl: MapUp for aligned label case
			model.MapUp(data[v], mapped[v], v, args.inData[2]);
            //kl: MapUp for the AutoEncoder case
            /*if (v==0){
                model.MapUp(data[v], mapped[v]);
            }*/
            
			if (outputData[v].size() > 0) mapped[v].WriteToFile(outputData[v]);
		}
	}
	
	return CCA::TestCorr(mapped);
}

void Train(const ProgramArgs & args, DeepCCAModel & model, const vector<AllocatingMatrix> & trainData) {
	DCCAHyperParams hyperParams;
	Deserialize(hyperParams, args.inParams); // kl: input inParams

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
    
    cout<<"inFeatSelect1: "<<args.inFeatSelect[0]<<"\t"<<"inFeatSelect2: "<< args.inFeatSelect[1]<<endl;

	double corr = model.Train(hyperParams, args.numLayers, args.inFeatSelect, args.outputSize, trainData, pretrainModifiers, finetuneModifiers, rand, args.inData[2]);

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
		for (int v = 0; v < 2; ++v) args.iSize[v] = model.InputSize(v);
		LoadData(args, trainData);
	} else { //kl: without model, have to use the training data to train a model firstly
		LoadData(args, trainData);
		Train(args, model, trainData);

		if (args.outModel.size() > 0) {
			Serialize(model, args.outModel); //kl: store the model in output Model for testing
		}
	}

	double corr = Map(model, trainData, args.outData, args);
	if (!IsNaN(corr)) cout << "Total DCCA corr: " << corr << endl;

	return 0;
}
