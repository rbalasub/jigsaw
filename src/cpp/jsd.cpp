#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <vector>

using namespace std;
double KL(vector<double> &v1, vector<double> &v2) {
  double kl = 0.0;
  for (int i = 0; i < v1.size(); ++i) {
    if (v1[i] <= 0 || v2[i] <=0 ) {
      //cerr << i << endl;
    }
    if (v1[i] > 0 && v2[i] > 0) {
      kl += v1[i] * (log(v1[i]) - log(v2[i])) / log(2.0);
    } else {
      cerr << i << " " << v1[i] << " " << v2[i] << endl;
    }
  }
  //cout << "^^" << kl << endl;
  return kl;
}

double JSD(vector<double> &v1, vector<double> &v2) {
  vector<double> avg(v1.size());
  for (int i = 0; i < v1.size(); ++i) {
    avg[i] = (v1[i] + v2[i]) / 2.0;
  }
  return 0.5 * (KL(v1, avg) + KL(v2, avg)); 
}

void ReadModel(ifstream &file, vector<vector<double> > &topics) {
  string line;

  while (getline(file, line)) {
    vector<double> topic;
    istringstream iss(line);
    double item;
    double tot = 0.0;
    while (iss >> item) {
      topic.push_back(item);
      //cerr << "BLAH " << item << endl;
      tot += item;
    }
    //cerr << tot << endl;
    topic.pop_back(); // remove weight
    topics.push_back(topic);
  }
}

int main(int argc, char **argv) {
  vector<vector<double> > topics_1;
  ifstream model_file_1(argv[1]);
  ReadModel(model_file_1, topics_1);
  model_file_1.close();

  //cerr << "Read model 1" << endl;
  vector<vector<double> > topics_2;
  ifstream model_file_2(argv[2]);
  //cerr << argv[2] << endl;
  ReadModel(model_file_2, topics_2);
  model_file_2.close();
  //cerr << "Read model 2" << endl;

  if (topics_1.size() != topics_2.size()) {
    cerr << "The two models have different number of topics" << endl;
    return 0;
  }

  for (int i = 0; i < topics_1.size(); ++i) {
    for (int j = 0; j < topics_2.size(); ++j) {
      //cerr << "Topic " << i << " " << j << endl;
      if (topics_1[i].size() != topics_2[j].size()) {
        cerr << i << " " << j << " topics have different #items in distr " << topics_1[i].size() << " " << topics_2[j].size() << endl;
        return 0;
      }
      cout << JSD(topics_1[i], topics_2[j]) << " ";
    }
    cout << endl;
  }
  return 0;
}
