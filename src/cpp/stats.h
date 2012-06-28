#ifndef __STATS_H__
#define __STATS_H__

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <time.h>

using namespace std;
class Stats {
 public:
  void Save(string, double);
  void Save(string, vector<double>&);
  void Save(string, double*, int);
  void Dump(ostream &);
  Stats() {
    time_t st    = time(NULL);
    start_time_  = asctime(localtime(&st));
  }
 private:
  string start_time_;
  map<string, vector<double> > real_stats_;
  map<string, vector<vector<double> > > vector_stats_;
};

#endif
