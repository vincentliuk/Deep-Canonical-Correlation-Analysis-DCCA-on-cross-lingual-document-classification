#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>

using namespace std;

class ProgramArgs {
	void LoadFromFile(const string & argsFilename) {
		ifstream argsFile (argsFilename);
		if (!argsFile.is_open()) {
			cerr << "Couldn't open " << argsFilename << endl;
			exit(1);
		}

		while (argsFile.good()) {
			string line;
			getline(argsFile, line);
			int beginComment = line.find("/*");
			while (beginComment != string::npos) {
				string prefix = line.substr(0, beginComment);
				int endComment = line.find("*/");
				while (endComment == string::npos) {
					getline(argsFile, line);
					endComment = line.find("*/");
				}
				line = prefix + line.substr(endComment + 2);
				beginComment = line.find("/*");
			}
			beginComment = line.find("//");
			line = line.substr(0, beginComment); // still works if beginComment == npos

			int begin = 0;
			while (begin < line.size() && isspace(line[begin])) begin++;
			int end = line.size();
			while (end > 0 && isspace(line[end-1])) end--;
			line = line.substr(begin, end-begin);

			ParseArg(line);
		} 

		argsFile.close();
	}

	void ParseArg(string argAndVal) {
		argAndVal.erase(remove_if(argAndVal.begin(), argAndVal.end(), ::isspace), argAndVal.end());

		if (argAndVal.size() == 0) return;

		cout << argAndVal << endl;

		int eq = argAndVal.find('=');
		if (eq == string::npos) {
			cerr << "Couldn't find value for line: " << argAndVal << endl;
			exit(1);
		}
		string arg = argAndVal.substr(0, eq);
		string val = argAndVal.substr(eq + 1);

		if (arg.compare("args") == 0) {
			LoadFromFile(val);
		} else if (arg.compare("inParams") == 0) inParams = val;
		else if (arg.compare("outputSize") == 0) outputSize = atoi(val.c_str());
		else if (arg.compare("numLayers1") == 0) numLayers[0] = atoi(val.c_str());
		else if (arg.compare("numLayers2") == 0) numLayers[1] = atoi(val.c_str());
		else if (arg.compare("hSize1") == 0) hSize[0] = atoi(val.c_str());
		else if (arg.compare("hSize2") == 0) hSize[1] = atoi(val.c_str());
		else if (arg.compare("inModel") == 0) inModel = val;
		else if (arg.compare("outModel") == 0) outModel = val;
		else if (arg.compare("inData1") == 0) inData[0] = val;
		else if (arg.compare("inData2") == 0) inData[1] = val;
                else if (arg.compare("inData3") == 0) inData[2] = val;
		else if (arg.compare("outData1") == 0) outData[0] = val;
		else if (arg.compare("outData2") == 0) outData[1] = val;
		else if (arg.compare("iSize1") == 0) iSize[0] = atoi(val.c_str());
		else if (arg.compare("iSize2") == 0) iSize[1] = atoi(val.c_str());
		else if (arg.compare("iFeatSel1") == 0) inFeatSelect[0] = atoi(val.c_str());
		else if (arg.compare("iFeatSel2") == 0) inFeatSelect[1] = atoi(val.c_str());
		else if (arg.compare("trainSize") == 0) trainSize = atoi(val.c_str());
		else {
			cerr << "Unrecognized arg: " << arg << endl;
			PrintArgs();
			exit(1);
		}
	}

	void CheckArgs() {
		bool training = (inModel.size() == 0);

		if (training) {
			for (int v = 0; v < 2; ++v) {
				if (numLayers[v] == -1) {
					cerr << "Number of layers not specified for view " << (v+1) << endl;
					exit(1);
				}

				if (numLayers[v] > 1 && hSize[v] == -1) {
					cerr << "Number of hidden units not specified for view " << (v+1) << endl;
					exit(1);
				}

				if (inData[v].size() == 0) {
					cerr << "Training data not specified for view " << (v + 1) << endl;
					exit(1);
				}
			}

			if (inParams.size() == 0) {
				cerr << "Training hyperparameter path not specified." << endl;
				exit(1);
			}

			if (outModel.size() == 0) {
				cerr << "Warning: model output path not specified" << endl;
			}

			if (outputSize == 0) {
				cerr << "Output size not specified" << endl;
				exit(1);
			}
		} else {
      // not training, read stored model
			if (outModel.size() > 0) {
				cerr << "Cannot specify both inModel and outModel" << endl;
				exit(1);
			}

      bool twoViews = (inData[0].size() > 0) && (inData[1].size() > 0);

			for (int v = 0; v < 2; ++v) {
        bool haveIn = (inData[v].size() > 0), haveOut = (outData[v].size() > 0);
				if (haveOut && !haveIn) {
					cerr << "Input data not specified for view " << (v + 1) << endl;
					exit(1);
				}

				if (haveIn && !haveOut) {
          if (!twoViews) {
            cerr << "Output data not specified for view " << (v + 1) << endl;
            exit(1);
          }
					cerr << "Warning: output data not specified for view " << (v + 1) << endl;
					cerr << "Computing correlation only." << endl;
				}
			}
		}

		for (int v = 0; v < 2; ++v) {
			if (inData[v].size() == 0 && iSize[v] == -1) {
				cerr << "Data size not specified for view " << (v + 1) << endl;
				exit(1);
			}
		}
	}

public:
	static void PrintArgs() {
		printf("Arguments are specified on the command line with <name>=<value>. They can also\n");
		printf("   be read one per line from a file, allowing c-style comments.\n");
		printf("args         (string)   path to text file containing additional arguments\n");
		printf("inParams     (string)   path to binary file containing hyperparameter values\n");
		printf("outputSize   (int)      dimensionality of output representations\n");
		printf("numLayers1   (int)      number of layers in network for view 1\n");
		printf("numLayers2   (int)      number of layers in network for view 2\n");
		printf("hSize1       (int)      number of units per hidden layer for view 1\n");
		printf("hSize2       (int)      number of units per hidden layer for view 2\n");
		printf("inModel      (string)   path to previously stored DCCA model to read in\n");
		printf("outModel     (string)   path in which to store trained DCCA model\n");
		printf("inData1      (string)   path to matrix of input data for view 1\n");
		printf("inData2      (string)   path to matrix of input data for view 2\n");
		printf("outData1     (string)   path in which to store mapped data for view 1\n");
		printf("outData2     (string)   path in which to store mapped data for view 2\n");
		printf("iSize1       (int)      dimensionality of input data for view 1\n");
		printf("iSize2       (int)      dimensionality of input data for view 2\n");
		printf("iFeatSel1    (int)      reduced input dim of view 1 after PCA whitening\n");
		printf("iFeatSel2    (int)      reduced input dim of view 2 after PCA whitening\n");
		printf("trainSize    (int)      if specified, read only first n columns of all input\n");
	}

	string inParams;
	string inModel, outModel;
	string inData[3];
	string outData[2];
	int iSize[2];
	int numLayers[2];
	int hSize[2];
	int inFeatSelect[2];
	int outputSize;
	int trainSize;

	ProgramArgs(int argc, char** argv)
		:
		outputSize(-1),
		trainSize(-1)
	{
		if (argc <= 1 || (((string)argv[1]).compare("help") == 0)) {
			PrintArgs();
			exit(0);
		}

		for (int v = 0; v < 2; ++v) {
			iSize[v] = numLayers[v] = hSize[v] = inFeatSelect[v] = -1;
		}

    if (argc == 2 && ((string)argv[1]).find('=') == string::npos) {
      LoadFromFile(argv[1]);
    } else {
      for (int i = 1; i < argc; ++i) {
        ParseArg(argv[i]);
      }
    }
		cout << endl;

		CheckArgs();
	}

};
