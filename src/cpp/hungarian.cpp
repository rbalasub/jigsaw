/*
   hungarian.cpp

Input: Predicted topics for words in vocab - usually generated by link_lda by picking topic with max prob for the word. usually will be [0, nTopics-1]
       True classes for words - external knowledge. this program requires it to be in the range [1, nClasses]

*/

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <iomanip>
#include <algorithm>

using namespace std;

void Read(const string &file, vector<int> &vec, int offset = 0) {
  ifstream ifs(file.c_str());
  while (!ifs.eof()) {
    int tmp;
    ifs >> tmp;
    vec.push_back(tmp);
    vec[vec.size() - 1] += offset;
  } // end while
  vec.pop_back();
}

double Accuracy(vector<int> &pred, vector<int> &orig) {
  int correct = 0;
  int total = 0;
  if (pred.size() != orig.size())
    cerr << "WTF\n";

  for (int i = 0; i < pred.size(); ++i) {
    if (orig[i] == 0)
      continue;
    total++;
    if (pred[i] == orig[i])
      correct++;
  }
  // cout << total << endl;

  return correct * 1.0 / total;
}

int main(int argc, char **argv) {
  vector<int> tm_pred, true_classes;

  Read(argv[1], tm_pred); 
  Read(argv[2], true_classes);

  vector<int> tmp = true_classes;
  sort(tmp.begin(), tmp.end());
  int n_classes = unique(tmp.begin(), tmp.end()) - tmp.begin();
  if (find(tmp.begin(), tmp.end(), 0) != tmp.end())
    n_classes--;

  vector<int> row(n_classes+1);
  vector<vector<int> >matrix;
  for (int k = 1; k <= n_classes; ++k) {
    matrix.push_back(row);
  }


  for (int i = 0; i < tm_pred.size(); ++i) {
    for (int k = 0; k < n_classes; ++k) {
      // class k is cluster
      //if (tm_pred[i] != k)
      //  matrix[k][tm_pred[i]][k]++;
      if (k != tm_pred[i]) {
        matrix[k][true_classes[i]]++;
      }
    } // end for classes
    matrix[tm_pred[i]][0]++;
  } // end predictions

  for (int k = 0; k < n_classes; ++k) {
    //cout << matrix[k][0] << " : ";
    for (int l = 1; l <= n_classes; ++l) {
      cout << matrix[k][l] << ' ';
    }
    cout << endl;
  }
}