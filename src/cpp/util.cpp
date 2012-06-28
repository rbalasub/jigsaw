#include <fstream>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <vector>
#include "util.h"

using namespace std;
double Random() {
  return rand() * 1.0 / RAND_MAX;
}

// Get a sample from Unif([0...max-1])
int UniformSample(int max) {
  return static_cast<int>((rand() * 1.0 / RAND_MAX) * max);
}

void MakePathAbsolute(string &path) {
  static char cwd[2048];
  static bool initialized = false;
  if (!initialized) {
    getcwd(cwd, 2048);
    initialized = true;
  }

  if (path.size() && path[0] != '/') {
    path = cwd + ("/" + path); 
  }
}

int SampleTopicFromMultinomial(vector<long double> &cdf) {
  long double unif_sample = rand() * 1.0 / RAND_MAX;
  int id = -1;
  for (int i = 0; i < cdf.size(); ++i) {
    if (unif_sample < (cdf[i] / cdf[cdf.size() - 1])) {
      id = i;
      break;
    } // end if
  } // end for
  return id;
}

int SampleTopicFromMultinomial(long double *cdf, int n) {
  long double unif_sample = rand() * 1.0 / RAND_MAX;
  int id = -1;
  for (int i = 0; i < n; ++i) {
    if (unif_sample < (cdf[i] / cdf[n - 1])) {
      id = i;
      break;
    } // end if
  } // end for
  return id;
}

void InitRandomizer(int rand_initializer) {
  if (!rand_initializer)
    return;

  if (rand_initializer == -1)  {
    srand(time(NULL));
    std::cout << "Seeding randomizer with time\n";
  }
  else {
    srand(rand_initializer); // preset initializer
    std::cout << "Seeding randomizer with specified value: " << rand_initializer << "\n";
  }
}

int GetNumLinesInFile(const string &file_name) {
  int cnt = 0;
  ifstream ifs(file_name.c_str());
  if (!ifs) {
    return 0;
  }
  string dummy;
  while (getline(ifs, dummy)) {
    if (dummy[0] != '#')
      cnt++;
  }
  ifs.close();
  return cnt;
}
