#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <cmath>
#include <map>

using namespace std;
typedef vector<double> FV;

int main(int argc, char **argv) {

  vector<FV> all_nodes_trans;

  ifstream ifs (argv[1]);
  string line;
  double dbl;
  while (getline(ifs, line)) {
    istringstream iss(line); 
    FV node;
    while (iss >> dbl) {
      node.push_back(dbl);
    } // end while
    node.pop_back();
    node.pop_back();
    node.pop_back();
    all_nodes_trans.push_back(node);
  } // end of model file

  ifs.close();

  FV topic_vec(all_nodes_trans.size());
  double avg_entropy = 0.0;
  for (int i = 0; i < all_nodes_trans[0].size(); ++i) {
    FV word_vec(all_nodes_trans.size());
    double sum = 0.0;
    for (int j = 0; j < all_nodes_trans.size(); ++j) {
      word_vec[j] = all_nodes_trans[j][i];
      sum += word_vec[j];
    }
    double entropy = 0.0;
    double max = word_vec[0];
    int maxid = 0;
    for (int j = 0; j < all_nodes_trans.size(); ++j) {
      double p = word_vec[j] / sum;
      if (p > 0) 
        entropy -= p * log(p);
      if (word_vec[j] > max) {
        max = word_vec[j];
        maxid = j;
      }
    }
    topic_vec[maxid]++;
    avg_entropy += entropy;
    //cerr << entropy/log(2.0) << endl;
  } // end nodes
  cout << "Entropy:" << avg_entropy /(log(2.0) * all_nodes_trans[0].size()) << '\t';

  double entropy = 0.0;
  double sum = 0;
  for (int j = 0; j < all_nodes_trans.size(); ++j) {
    double p = topic_vec[j];
    if (p > 0) 
      entropy -= p * log(p);
    sum += topic_vec[j];
  }
  entropy /= sum;
  entropy += log(sum);
  entropy /= log(2.0);
  cout << " BalanceEntropy:" << entropy << endl;
}
