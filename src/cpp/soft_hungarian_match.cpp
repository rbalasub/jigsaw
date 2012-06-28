// This program computes macro and micro average F1 scores for soft clustering using BlockLDA
// It can also output a histogram of num-labels vs. num-nodes to see how mixed the dataset both in terms of true labels and predicted labels.
#include "hungarian.h"
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>

using namespace std;

class Comparer {
 public:
   vector<double> &array;
   Comparer(vector<double> &a): array(a) {}
   bool operator()(int a, int b) {
     return array[a] > array[b];
   }
};

int main(int argc, char **argv) {
  // Read in predictions
  vector<vector<double> > node_distributions;
  ifstream ifs(argv[1]);
  string line;
  //double threshold = 0.035;
  int sz = -1;
  vector<double> all_prob_values;
  while (getline(ifs, line)) {
    vector<double> distr;
    istringstream iss(line);
    double d = 0.0;
    while (iss >> d) {
      distr.push_back(d);
      all_prob_values.push_back(d);
    } // end read in line
    if (sz != -1 && distr.size() != sz) {
      cout << "WTF?" << endl;
    }
    sz = distr.size();
    node_distributions.push_back(distr);
  } // end reading in distributions
  cout << sz << " topics in the model file " << endl;

  // Read in true labels
  int max = -1;
  int min = 100000;
  ifstream ifs2(argv[2]);
  vector<set<int> > true_labels;
  vector<set<int> > pred_labels;
  vector<set<int> > all_labels;
  map<int, int> true_histogram;
  map<int, int> pred_histogram;
  int total_true_labels = 0;
  while (getline(ifs2, line)) {
    set<int> labels;
    pred_labels.push_back(labels);
    istringstream iss(line);
    int d = 0;
    while (iss >> d) {
      labels.insert(d);
      total_true_labels++;
      if (d > max)
        max = d;
      if (d < min)
        min = d;
    } // end read in line
    true_histogram[labels.size()]++;
    true_labels.push_back(labels);
    all_labels.push_back(labels);
  } // end reading in distributions
  cout << "range of classes is " << min << " to " << max << endl;

  sort(all_prob_values.begin(), all_prob_values.end());
  double threshold = all_prob_values[all_prob_values.size() - total_true_labels];

  // Make cost matrix for alignment
  int t = node_distributions[0].size();
  int **cost = new int*[t];
  vector<int> cluster_ids;
  for (int i = 0; i < t; ++i) {
    cost[i] = new int[t];
    for (int j = 0; j < t; ++j)
      cost[i][j] = 0;
    cluster_ids.push_back(i);
  }

  for (int i = 0; i < true_labels.size(); ++i) {
    // make cost matrix
    //if (true_labels[i].count(max))
    //  continue; // hack - v. dirty
    sort(cluster_ids.begin(), cluster_ids.end(), Comparer(node_distributions[i]));
    for (int j = 0; j < true_labels[i].size(); ++j) {
      for (int k = 0; k < t; ++k) {
        if (!true_labels[i].count(k))
          cost[cluster_ids[j]][k]++;
      } // go over all classes
    } // all true labels of node i
  } // end nodes

  hungarian_problem_t problem;

  hungarian_init(&problem, cost, t, t, HUNGARIAN_MODE_MINIMIZE_COST);
  // hungarian_print_costmatrix(&problem);
  hungarian_solve(&problem);
  // hungarian_print_assignment(&problem);

  // get assignments
  cout << t << " classes " << endl;
  vector<int> matched_classes(t); //topic -> true class
  for (int i = 0; i < t; ++i) {
    for (int j = 0; j < t; ++j) {
      if (problem.assignment[i][j] == 1) {
        matched_classes[i] = j;
        break;
      } // end if
    } // look through all classes 
    cout << "Matched topic " << i << " to class " << matched_classes[i] << endl;
  } // look for all topics

  vector<vector<int> > contingency; // dim Nclasses x 3 (TP, FP, FN)
  vector<int> blah(3);
  for (int i = 0; i < t; ++i)
    contingency.push_back(blah);

  int pred_cnt = 0;
  double average_kl = 0.0;
  for (int i = 0; i < true_labels.size(); ++i) {
    //if (true_labels[i].count(max))
    //  continue; // hack - v. dirty
    // set up predicted labels and all labels
    sort(cluster_ids.begin(), cluster_ids.end(), Comparer(node_distributions[i]));

    int pred_label_cnt = 0;
    for (int k = 0; k < t; ++k) {
      if (node_distributions[i][k] >= threshold)
        pred_label_cnt++;
    } // for pred histogram
    if (pred_label_cnt == 0)
      pred_label_cnt = 1;
    pred_histogram[pred_label_cnt]++;

    double kl = 0.0;
    for (int k = 0; k < true_labels[i].size(); ++k) {
      pred_labels[i].insert(matched_classes[cluster_ids[k]]);
      pred_cnt++;
      all_labels[i].insert(matched_classes[cluster_ids[k]]);

      double p = 1.0 / true_labels[i].size();
      kl += p * log(p / node_distributions[i][cluster_ids[k]]) / log(2.0);
    }
    average_kl += kl;

    for (set<int>::iterator j = all_labels[i].begin(); j != all_labels[i].end(); ++j) {
      int in_pred = true_labels[i].count(*j);
      int in_true = pred_labels[i].count(*j);
      if (in_pred && in_true)
        contingency[*j][0]++; //TP
      if (in_pred && !in_true)
        contingency[*j][1]++; //FP
      if (!in_pred && in_true)
        contingency[*j][2]++; //FN
    } // all labels of node i
  } // end nodes
  cout << pred_cnt << " labels " << pred_cnt * 1.0 / true_labels.size() << " avg pred labels per node" << endl;

  double macro_r = 0.0;
  double macro_p = 0.0;
  double macro_f = 0.0;

  int tp = 0;
  int fp = 0;
  int fn = 0;
  for (int i = 0; i < t; ++i) {
    double precision = contingency[i][0] * 1.0 / (contingency[i][0] + contingency[i][1]); 
    double recall =    contingency[i][0] * 1.0 / (contingency[i][0] + contingency[i][2]); 
    double f = 2 * precision * recall / (precision + recall);
    tp += contingency[i][0];
    fp += contingency[i][1];
    fn += contingency[i][2];
    macro_p += precision;
    macro_r += recall;
    macro_f += f;
    //cout << contingency[i][0] << " " <<  contingency[i][1] << " " <<  contingency[i][2] << " " <<  endl;
    cout << "Class " << i << " precision: " << precision << " recall: " << recall << " f: " << f << endl;
  }
  macro_r /= t;
  macro_p /= t;
  macro_f /= t;

  double micro_p = tp * 1.0 / (tp + fp);
  double micro_r = tp * 1.0 / (tp + fn);
  double micro_f = 2 * micro_p * micro_r / (micro_p + micro_r);
  cout << "Macro precision: " << macro_p << " recall: " << macro_r << " f: " << macro_f << endl;
  cout << "Micro precision: " << micro_p << " recall: " << micro_r << " f: " << micro_f << endl;


  cout << "True label histogram\n";
  for (map<int, int>::iterator i = true_histogram.begin(); i != true_histogram.end(); ++i) {
    cout << i->first << "\t" << i->second << endl;
  }

  cout << "Pred label histogram\n";
  for (map<int, int>::iterator i = pred_histogram.begin(); i != pred_histogram.end(); ++i) {
    cout << i->first << "\t" << i->second << endl;
  }

  cout << "Average KL divergence = " << average_kl / true_labels.size() << endl;
  return 0;
}

