#include <sstream>
#include <cstdlib>
#include "stats.h"

void Stats::Dump(ostream &os) {
  time_t et = time(NULL);
  string end_time  = asctime(localtime(&et));
  os << "start-time=" << start_time_;
  os << "end-time="   << end_time;

  ostringstream exec_time;
  int seconds = clock() / CLOCKS_PER_SEC;
  os << "execution-time=" << clock() / CLOCKS_PER_SEC << " seconds\n";
  if (seconds > 60 * 60 * 24) {
    int days    = seconds / (60 * 60 * 24);
    exec_time << days << " days ";
    seconds = seconds % (60 * 60 * 24);
  }
  if (seconds > 60 * 60) {
    int hours = seconds / 3600;
    exec_time << hours << " hours ";
    seconds = seconds % 3600;
  }
  if (seconds > 60) {
    int minutes = seconds / 60;
    exec_time << minutes << " minutes ";
    seconds = seconds % 60;
  }
  exec_time << seconds << " seconds ";
  os << "execution-time-human=" << exec_time.str() << '\n';

  char hostname[2000];
  gethostname(hostname, 2000);
  os << "hostname=" << hostname << '\n';


  for (map<string, vector<double> >::iterator it = real_stats_.begin(); it != real_stats_.end(); ++it) {
    double avg = 0.0;
    for (int run_num = 0; run_num < (it->second).size(); ++run_num) {
      ostringstream key_str;
      key_str << "run-" << run_num << '.' << it->first;
      os << key_str.str() << '=' << (it->second)[run_num] << '\n';
      avg += (it->second)[run_num];
    }
    os << "avg." << it->first << '=' << avg / (it->second).size() << '\n';
  } // end metrics

  for (map<string, vector<vector<double> > >::iterator it = vector_stats_.begin(); it != vector_stats_.end(); ++it) {
    vector<double> avg((it->second)[0].size());
    for (int run_num = 0; run_num < (it->second).size(); ++run_num) {
      ostringstream key_str;
      key_str << "run." << run_num << '.' << it->first;
      os << key_str.str() << '=';
      for (int dim = 0; dim < (it->second)[run_num].size(); ++dim) {
        os << (it->second)[run_num][dim];
        if (dim == (it->second)[run_num].size() - 1)
          os << '\n';
        else 
          os << ',';
        avg[dim] += (it->second)[run_num][dim];
      } // end dim
    } // end run num

    os << "run.avg." << it->first << '=';
    for (int dim = 0; dim < avg.size(); ++dim) {
      os << avg[dim] / (it->second).size();
      if (dim == avg.size() - 1)
        os << '\n';
      else 
        os << ',';
    } // end dim

  } // end metrics
}

void Stats::Save(string metric, double val) {
  real_stats_[metric].push_back(val);
}

void Stats::Save(string metric, vector<double> &vector_val) {
  vector_stats_[metric].push_back(vector_val);
}

void Stats::Save(string metric, double *array, int n) {
  vector<double> vector_val;
  for (int i = 0; i < n; ++i)
    vector_val.push_back(array[i]);
  vector_stats_[metric].push_back(vector_val);
}
