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
  vector<FV> all_nodes;

  int match_for = atoi(argv[3]);
  int match_against = atoi(argv[4]);

  ifstream ifs (argv[1]);
  string line;
  double dbl;
  while (getline(ifs, line)) {
    istringstream iss(line); 
    FV node;
    while (iss >> dbl) {
      node.push_back(dbl);
    } // end while
    //node.pop_back();
    all_nodes.push_back(node);
  } // end of model file
  ifs.close();

  vector<pair<double, double> > latlongs;
  vector<string> names;
  ifs.open(argv[2]);
  int cter = 0;
  while (getline(ifs, line)) {
    istringstream iss(line);
    pair<double, double> latlong;
    string name;
    iss >> latlong.first >> latlong.second >> name;
    latlongs.push_back(latlong);
    names.push_back(name);
  } //end reading labels file
  cerr << latlongs.size() << " number of labels\n";

  cerr << all_nodes.size() << " number of docs" << endl;
  cerr << all_nodes[0].size() << " dimensions" << endl;

  vector<double> js_div(all_nodes.size());

  vector<int> idx;
  for (int i = match_against; i < all_nodes.size(); ++i) {
    idx.push_back(i);
  }

  int correct_cnt[3] = {0, 0, 0};
  int total[3] = {0, 0, 0};
  int skipped_real[3] = {0, 0, 0};
  map<int, map<int, double> > memoize;
  for (int i = 0; i < all_nodes.size(); ++i) {
//  for (int i = 0; i < match_for; ++i) {
    // copy(all_nodes[i].begin(), all_nodes[i].end(), ostream_iterator<double>(cout, " "));
    if (i > 100 && i < 78308) {
      cout << "PASS \n";
      continue;
    }

    // cout << "Divergences ";
    for (int j = match_against; j < all_nodes.size(); ++j) {
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
    //for (int j = 0; j < all_nodes.size(); ++j) {
      // cout << myidx[j] << ' ' << js_div[myidx[j]] << ' ';
    //}
    // cout << endl;
    
    for (int h = 0; h < sz; ++h) {
      map<int, int> counts;
      double weighted_lat = 0.0;
      double weighted_long = 0.0;
      double weight = 0.0;
      int k = 0;
      while (k < K[h]) {
        weighted_lat += memoize[i][myidx[k]] * latlongs[myidx[k] - match_against].first; 
        weighted_long += memoize[i][myidx[k]] * latlongs[myidx[k]- match_against].second; 
        weight += memoize[i][myidx[k]]; 
        ++k;
      }
      // cout << endl;
      
      cout << weighted_lat/weight << " " << weighted_long/weight << " " << names[myidx[0] - match_against] << endl;
    } // end for
    // cout << "True " << labels[i] << "; Pred " << winning_label << '\n' << endl;
  } // end nodes

  for (int h = 0; h < 2; ++h) {
//    double acc = correct_cnt[h] * 1.0 / total[h]; 
  //  cout << "Accuracy " << acc <<  "-----" << all_nodes.size() << " = " << total[h] << " + " << skipped_real[h] << " + " << (all_nodes.size() - total[h] - skipped_real[h]) << endl;
  } // end for
}

