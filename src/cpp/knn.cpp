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

double KL(FV &p, FV &q) {
  double kl = 0.0;
  for (int i = 0; i < p.size(); ++i) {
    kl += p[i] * log(p[i] / q[i]);
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

class Comparer {
 public:
   vector<double> &ref_;
   Comparer(vector<double> &ref):ref_(ref) {}
   bool operator()(int a, int b) {
     return ref_[a] < ref_[b];
   }
};

bool operator>(pair<int,int> &a, pair<int, int> &b) {
  return a.second > b.second;
}

int main(int argc, char **argv) {

  // int K = atoi(argv[3]);
  int K[] = {1, 3, 5};
  int sz = 3;
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

  vector<int> labels;
  ifs.open(argv[2]);
  int cter = 0;
  while (getline(ifs, line)) {
    istringstream iss(line);
    int tmp;
    iss >> tmp;
    labels.push_back(tmp);
  } //end reading labels file
  cerr << labels.size() << " number of labels\n";

  bool mltopic = 0;
  ofstream mlt_ofs;
  if (argc >= 4) {
    mltopic = 1;
    mlt_ofs.open(argv[3]);
  }

  vector<FV> all_nodes(all_nodes_trans[0].size()); // as many words
  for (int i = 0; i < all_nodes_trans[0].size(); ++i) {
    FV word_vec(all_nodes_trans.size());
    double sum = 0;
    for (int j = 0; j < all_nodes_trans.size(); ++j) {
      word_vec[j] = all_nodes_trans[j][i];
      sum += word_vec[j];
    }
    double max = word_vec[0];
    double max_idx = 0;
    for (int j = 0; j < all_nodes_trans.size(); ++j) {
      if (word_vec[j] > max) {
        max = word_vec[j];
        max_idx = j;
      }
      word_vec[j] /= sum;
    }
    if (mltopic) {
      mlt_ofs << max_idx << '\n';
    }
    all_nodes[i] = word_vec;
  }
  cerr << all_nodes.size() << " number of words" << endl;
  cerr << all_nodes[0].size() << " dimensions" << endl;

  vector<double> js_div(all_nodes.size());

  vector<int> idx;
  for (int i = 0; i < all_nodes.size(); ++i) {
    idx.push_back(i);
  }

  int correct_cnt[3] = {0, 0, 0};
  int total[3] = {0, 0, 0};
  int skipped_real[3] = {0, 0, 0};
  map<int, map<int, double> > memoize;
  for (int i = 0; i < all_nodes.size(); ++i) {
    // copy(all_nodes[i].begin(), all_nodes[i].end(), ostream_iterator<double>(cout, " "));

    // cout << "Divergences ";
    for (int j = 0; j < all_nodes.size(); ++j) {
      if (j == i) {
        js_div[j] = 10000000;
        // cout << 100000 << ' ';
        continue;
      }
      double this_js_div;
      if (memoize.count(i) && memoize[i].count(j))
        this_js_div = memoize[i][j];
      else {
        this_js_div = Distance(all_nodes[i], all_nodes[j]);
        memoize[i][j] = this_js_div;
        memoize[j][i] = this_js_div;
      }  
      // cout << this_js_div << ' ';
      js_div[j] = this_js_div;
    } // end candidate nodes
    // cout << endl;

    vector<int> myidx = idx;
    sort(myidx.begin(), myidx.end(), Comparer(js_div));
    // cout << "distances ";
    for (int j = 0; j < all_nodes.size(); ++j) {
      // cout << myidx[j] << ' ' << js_div[myidx[j]] << ' ';
    }
    // cout << endl;
    
    for (int h = 0; h < sz; ++h) {
      map<int, int> counts;
      int k = 0;
      int ctr = 0;
      while (k < K[h]) {
        if (labels[myidx[ctr]] != 0) {
          counts[labels[myidx[ctr]]]++;
          ++k;
        }
        ctr++;
      }
      // cout << endl;
      
      int winning_label = (std::max_element(counts.begin(), counts.end()))->first;
      
      if (labels[i] != 0) { // 0 == UNK true label
        if (winning_label == 0) {
          // cerr << "Best match is with unknown points\n";
          skipped_real[h]++;
        } else {
          bool correct = (labels[i] == winning_label);
          correct_cnt[h] += correct;
          total[h]++;
        }
      } // end if
    } // end for
    // cout << "True " << labels[i] << "; Pred " << winning_label << '\n' << endl;
  } // end nodes

  for (int h = 0; h < 2; ++h) {
    double acc = correct_cnt[h] * 1.0 / total[h]; 
    cout << " KNN Accuracy " << acc <<  "-----" << all_nodes.size() << " = " << total[h] << " + " << skipped_real[h] << " + " << (all_nodes.size() - total[h] - skipped_real[h]) << ":";
  } // end for
  cout << endl;
  if (mltopic)
    mlt_ofs.close();
}

