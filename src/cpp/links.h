#ifndef __LINKS_H__
#define __LINKS_H__

#include <fstream>
class Model;
class Config;

class Links {
 public:

  const Config *config_; // a fully configured Config object.
  int n_links_;
  Links(const Config *c, int n, const string &file) : 
       config_(c),
       averager_count_(0),
       n_links_(n) {
    if (config_->model_links_ && n_links_ > 0)
      Setup(file);
  }

  int **links_; // ndocs x 2
  int **link_topic_assignments_; // ndocs x 2
  double **link_topic_pair_counts_; //K x K - distribution over pairs of topics.
  double **averager_link_topic_pair_counts_;
  int averager_count_;

  void Allocate();
  void Read(istream &);
  void Free();
  void RandomInit(int **node_labels_ = NULL);

  void AddToAverager();
  void Average();

  void SaveTopics(std::ostream &os);
  void SaveTopicDistributions(std::ostream &);
  void Save(const string &prefix);
  void Setup(const string &file);
  void CheckIntegrity();

  friend class Model;
};


#endif
