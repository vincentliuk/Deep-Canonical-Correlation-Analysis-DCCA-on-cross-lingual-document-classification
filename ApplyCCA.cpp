//
//  ApplyCCA.cpp
//  DeepCCA
//
//  Created by Kai Liu on 8/24/14.
//  Copyright (c) 2014 Kai Liu. All rights reserved.
//

#include <iostream>
#include "CCA.h"
#include "Matrix.h"
#include "CCA.h"
#include "sstream"

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
        //if (feat<=500){
            *(mat.Start() + lineNum*featNum + feat-1) = (double) value; //}
    }    
}

//kl: parse the AutoEncoder BNx ,BNy of the NLP task
void ParseAEline(string featAndVal,int featNum, int lineNum, AllocatingMatrix & mat) {
    if (featAndVal.size() == 0) return;
    
    //cout << featAndVal << endl;
    
    istringstream iss(featAndVal);
    string tempFeatAndVal;
    
    int feat=1;
    // everytime, output a "feature:value" pair, put into the matrix
    while(iss >> tempFeatAndVal){
        //cout<< tempFeatAndVal <<" ";
        double value;
        stringstream convert(tempFeatAndVal);
        if(!(convert >> value))
            value = 0;
        
        *(mat.Start() + lineNum*featNum + feat-1) = (double) value;
        feat = feat + 1;
    }
}

void LoadFromFileCCA(char** argv, vector<AllocatingMatrix> & trainData, vector<AllocatingMatrix> & testData, vector<AllocatingMatrix> & labelData) {
   //kl: the argv will look like:
//./cca_train_test_noCentered.exe 5000 1029 train.en.tok.data train.es.tok.data 5000 124 test.en.tok.data test.es.tok.data 50 1029 50dlabelVector.data
    //kl: input the train input data
  for (int k=3;k<5;k++){
    ifstream argsFile(argv[k]);
    if (!argsFile.is_open()) {
        cerr << "Couldn't open " << argv[k] << endl;
        exit(1);
    }
    int featNum = atoi(argv[1]); // feature number
      //int featNum = 500;
    int instNum = atoi(argv[2]); //instances number
    trainData[k-3].Resize(featNum, instNum);
    for(int i=0;i<featNum*instNum;i++){
        *(trainData[k-3].Start()+i) = 0.0;
    }
    
    int lineNum = 0;
    while (argsFile.good()) {
        string line;
        getline(argsFile, line);
        //kl: train.en.tok.data et.al for CCA
        // ParseArgCCA(line, featNum, lineNum, trainData[k-3]);
        //kl: for AutoEncoder BNx, BNy input data parse for CCA
        ParseAEline(line, featNum, lineNum, trainData[k-3]);
        lineNum++;
    }
    
    //if(lineNum == 1029) cout<<"data input finished!"<<endl;
          
    double* pdata = trainData[k-3].Start();
    
    for(int i = 0; i < instNum; i++) {
        for (int j = 0; j < featNum; j++)
            cout << pdata[i * featNum + j] << " ";
        cout << "one instance end" << endl;
    }
    
    argsFile.close();
  }
    //kl: input the test input data
    for (int k=7;k<=8;k++){
        ifstream argsFile(argv[k]);
        if (!argsFile.is_open()) {
            cerr << "Couldn't open " << argv[k] << endl;
            exit(1);
        }
        int featNum = atoi(argv[5]); // feature number
        //int featNum = 500;
        int instNum = atoi(argv[6]); //instances number
        testData[k-7].Resize(featNum, instNum);
        for(int i=0;i<featNum*instNum;i++){
            *(testData[k-7].Start()+i) = 0;
        }
        
        int lineNum = 0;
        while (argsFile.good()) {
            string line;
            getline(argsFile, line);
            
            //ParseArgCCA(line, featNum, lineNum, testData[k-7]);
            ParseAEline(line, featNum, lineNum, testData[k-7]);
            lineNum++;
        }
        
        //if(lineNum == 1029) cout<<"data input finished!"<<endl;
        
        double* pdata = testData[k-7].Start();
        
        /*for(int i = 0; i < instNum; i++) {
         for (int j = 0; j < featNum; j++)
         cout << pdata[i * featNum + j] << " ";
         cout << "one instance end" << endl;
         }*/
        
        argsFile.close();
    }
    
    //kl: input the EN label data
    for (int k=7;k<=8;k++){
        ifstream argsFile(argv[k]);
        if (!argsFile.is_open()) {
            cerr << "Couldn't open " << argv[k] << endl;
            exit(1);
        }
        int featNum = atoi(argv[5]); // feature number
        //int featNum = 500;
        int instNum = atoi(argv[6]); //instances number
        testData[k-7].Resize(featNum, instNum);
        for(int i=0;i<featNum*instNum;i++){
            *(testData[k-7].Start()+i) = 0;
        }
        
        int lineNum = 0;
        while (argsFile.good()) {
            string line;
            getline(argsFile, line);
            
            //ParseArgCCA(line, featNum, lineNum, testData[k-7]);
            ParseAEline(line, featNum, lineNum, testData[k-7]);
            lineNum++;
        }
        
        
        double* pdata = testData[k-7].Start();
        
        /*for(int i = 0; i < instNum; i++) {
         for (int j = 0; j < featNum; j++)
         cout << pdata[i * featNum + j] << " ";
         cout << "one instance end" << endl;
         }*/
        
        argsFile.close();
    }
}


int main(int argc, char** argv) {
    // kl: argc is the string number in the excute line, while argv is the array for all the words
    
   
    vector<AllocatingMatrix> trainData(2);
    vector<AllocatingMatrix> testData(2);
    
    if (argc <= 1 || (((string)argv[1]).compare("help") == 0)) {
       
        exit(0);
    }
    
    LoadFromFileCCA(argv, trainData, testData);
    
    
    vector<string> inputTrainData(2);
    inputTrainData[0] = "outputData2/trainInput1.dat";
    inputTrainData[1] = "outputData2/trainInput2.dat";
    for(int i=0;i<2;i++){
        trainData[i].WriteToFile(inputTrainData[i]);
        cout<<"train Input "<<i<<" 's NumC: "<<trainData[i].NumC()<<endl;
        cout<<"train Input "<<i<<" 's NumR: "<<trainData[i].NumR()<<endl;
    }
    
    vector<string> inputTestData(2);
    inputTestData[0] = "outputData2/testInput1.dat";
    inputTestData[1] = "outputData2/testInput2.dat";
    for(int i=0;i<2;i++){
        testData[i].WriteToFile(inputTestData[i]);
        cout<<"test Input "<<i<<" 's NumC: "<<testData[i].NumC()<<endl;
        cout<<"test Input "<<i<<" 's NumR: "<<testData[i].NumR()<<endl;
    }
    
    CCA _cca;
    
    double ccaVal = _cca.InitWeightsDoc(trainData, testData, 0.0, 0.0);
    
    cout << "cca: " << ccaVal << endl;
    
        return 0;
}

