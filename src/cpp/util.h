#ifndef __UTIL_H__
#define __UTIL_H__

#include <cmath>
#include <map>
#include <numeric>
#include <string>
#include <sstream>

using namespace std;

void InitRandomizer(int rand_initializer);
double Random(); // return random number [0 .. 1]
int  UniformSample(int max);
int  SampleTopicFromMultinomial(vector<long double> &cdf);
int  SampleTopicFromMultinomial(long double *cdf, int n);
int GetNumLinesInFile(const string &file_name);
void MakePathAbsolute(string &path);

template <class T>
void DisplayMatrix (T **matrix, int m, int n,
                    char *label, bool sum, ostream &cout) {
  cout << "============== " << label << "===========\n";
  double *tot = new double[m];
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      cout << matrix[i][j] << ' ';
    }
    tot[i] = accumulate(matrix[i], matrix[i] + n, 0.0);
    if (sum)
      cout << "  Sum: " << tot[i] << '\n';
    else
      cout << '\n';
  }
  if (sum)
    cout << "Total = " << accumulate(tot, tot + m, 0.0) << "\n\n";
  else
    cout << "\n";

  delete[] tot;
}

template <class T>
void DisplayMatrix (T **matrix, int m, int *n,
                    char *label, bool sum, ostream &cout) {
  cout << "============== " << label << "===========\n";
  double *tot = new double[m];
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n[i]; ++j) {
      cout << matrix[i][j] << ' ';
    }
    tot[i] = accumulate(matrix[i], matrix[i] + n[i], 0.0);
    if (sum)
      cout << "  Sum: " << tot[i] << '\n';
    else
      cout << '\n';
  }
  if (sum)
    cout << "Total = " << accumulate(tot, tot + m, 0.0) << "\n\n";
  else
    cout << "\n";

  delete[] tot;
}

template <class T>
void DisplayColumn (T *column, int m, char *label, ostream &cout) {
  cout << "============== " << label << "===========\n";
  for (int i = 0; i < m; ++i)
    cout << column[i] << '\n';
  cout << "Total " << accumulate(column, column + m, 0.0) << "\n\n";
}

template<class T>
void Copy(T ***src, T***dest, int d1, int d2, int *d3) {
  for (int i = 0; i < d1; ++i)
    for (int j = 0; j < d2; ++j)
      for (int k = 0; k < d3[i]; ++k)
        dest[i][j][k] = src[i][j][k];
}

template<class T>
void Copy(T ***src, T***dest, int d1, int d2, int d3) {
  for (int i = 0; i < d1; ++i)
    for (int j = 0; j < d2; ++j)
      for (int k = 0; k < d3; ++k)
        dest[i][j][k] = src[i][j][k];
}

template<class T>
void Copy(T **src, T**dest, int d1, int d2) {
  for (int i = 0; i < d1; ++i)
    for (int j = 0; j < d2; ++j)
      dest[i][j] = src[i][j];
}

template<class T>
int SplitIntoArray(const string &s, char delim, T *arr, string t) {
  istringstream ss(s);
  string item;
  int i = 0;
  while(getline(ss, item, delim)) {
    if (t == "int")
      arr[i++] = atoi(item.c_str());
    else
      arr[i++] = atof(item.c_str());
  }
  return i;
}

#endif
