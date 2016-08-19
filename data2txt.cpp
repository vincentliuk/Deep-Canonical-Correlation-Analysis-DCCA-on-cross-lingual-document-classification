#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void ReadBin(const string& filename, int numR, int maxCols = -1) {
	if (filename.size() == 0) return;

	ifstream inStream(filename.c_str(), ios::in|ios::binary);
	if (!inStream.is_open()) {
		cout << "Couldn't open feature file " << filename.c_str() << endl;
		exit(1);
	}

	inStream.seekg(0, ios::end);
	int endPos = inStream.tellg();
	inStream.seekg(0, ios::beg);
	assert(endPos / sizeof(double) % numR == 0);
	int numC = (int)(endPos / sizeof(double) / numR);

	if (maxCols != -1) numC = min(numC, maxCols);
    double* pdata = (double*) malloc(numR * numC * sizeof(double));
	inStream.read((char*)pdata, numR * numC * sizeof(double));
    //cout << maxCols << "\t" << numR << endl;
    for(int i = 0; i < maxCols; i++) {
        for (int j = 0; j < numR; j++)
            cout << pdata[i * numR + j] << " ";
        cout << endl;
    }
	inStream.close();

	cout << "Read " << filename << " of size " << numR << "x" << numC << endl;
}

int main(int argc, char** argv) {
	ReadBin(argv[1], atoi(argv[2]), atoi(argv[3]));

	return 0;
}
