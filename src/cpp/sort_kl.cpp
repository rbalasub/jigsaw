#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <cmath>
#include <map>
#include <set>

using namespace std;
typedef vector<double> FV;

double KL(FV &p, FV &q) {
  double kl = 0.0;
  double denom1 = 0.0;
  double denom2 = 0.0;
  for (int i = 0; i < p.size(); ++i) {
//    kl += p[i] * log(p[i] / q[i]);
    kl += abs(p[i] - q[i]);
    denom1 = p[i] * p[i];
    denom2 = q[i] * q[i];
  }
  return kl;
}

double Distance(FV &distr_1, FV &distr_2) { 
  FV m(distr_1.size());
  for (int i = 0; i < distr_1.size(); ++i) {
    m[i] = (distr_1[i] + distr_2[i]) / 2.0;
  }
  return (KL(distr_1, m) + KL(distr_2, m)) / 2;
}

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

  vector<FV> all_nodes(all_nodes_trans[0].size()); // as many words
  for (int i = 0; i < all_nodes_trans[0].size(); ++i) {
    FV word_vec(all_nodes_trans.size());
    double sum = 0;
    for (int j = 0; j < all_nodes_trans.size(); ++j) {
      word_vec[j] = all_nodes_trans[j][i];
      sum += word_vec[j];
    }
    for (int j = 0; j < all_nodes_trans.size(); ++j) {
      word_vec[j] /= sum;
    }
    all_nodes[i] = word_vec;
  }
  cerr << all_nodes.size() << " number of words" << endl;
  cerr << all_nodes[0].size() << " dimensions" << endl;

  vector<double> js_div(all_nodes.size());

  map<int, map<int, double> > memoize;

  set<int> not_done;
  int starter = 0;
  double super_max = -1000000;
  for (int i = 0; i < all_nodes.size(); ++i) {
    not_done.insert(i);
    for (int j = 0; j < all_nodes.size(); ++j) {
      if (j == i) {
        js_div[j] = 0;
        continue;
      }
      double this_js_div;
        
      memoize[i][j] = KL(all_nodes[i], all_nodes[j]);
      js_div[j] = this_js_div;
      cerr << memoize[i][j] << " ";
    } // end candidate nodes
    cerr << endl;

    double max_distance = *max_element(js_div.begin(), js_div.end());
    if (max_distance > super_max) {
      super_max = max_distance;
      starter = i;
    }
    
  } // end nodes

  do {
    cout << starter;
    not_done.erase(starter);

    double min_distance = 100000000;
    int closest = 1;
    for (set<int>::iterator i = not_done.begin(); i != not_done.end(); ++i) {
      if (memoize[starter][*i] < min_distance) {
        min_distance = memoize[starter][*i];
        closest = *i;
      }
    }
    starter = closest;
    cout <<  " " << min_distance << endl;

  } while(not_done.size());
  return 0;
}

