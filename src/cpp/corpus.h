#ifndef __CORPUS_H__
#define __CORPUS_H__

#include <fstream>
#include <iostream>
#include <vector>

#include "stats.h"
#include "component.h"

class Config;
class Model;

using namespace std;

class Corpus {
 public:

  const Config *config_; // a fully configured Config object.
  int n_docs_;

  Corpus(const Config *c, int n, const string &file);

  int **doc_num_words_; // ntypes x ndocs
  int ***corpus_words_; // ntypes x ndocs x nwords
  
  double **real_values_; // ndocs x number of real valued attributes.
  bool   **real_flags_;  // ndocs x number of real valued attrs - are they valid?

  double **real_targets_;      // ndocs x number of real targets
  double **pred_targets_;      // ndocs x number of real targets
  bool   **real_target_flags_; // ndocs x number of real targets

  int ***word_topic_assignments_; // ntypes x ndocs x nwords
  int **counts_docs_topics_;      // ndocs x ntopics

  double **theta_;                // ndocs x ntopics - normalized version of counts.
  double **theta_entropy_components_; // ndocs x ntopics 
  double *weight_;                // ndocs - weighted sum of doc
  int averager_count_;            // how many models is theta_ the average of

  Component<int> ***md_entropy_components_; // ndocs X 2(my domain, general domain)
  SplComponent<int> **md_senti_entropy_components_; // ndocs(sentiment constraints)
  int *domains_;    //ndocs -- indicates domains to which the document belongs
  vector<vector<long double> > md_domain_sampling_distr_;  // domains x topics

  void MakeDomainSampler();

  void AddToAverager(); 
  void AddWord(int doc, int cur_topic, int type, bool remove);

  double GetTopicProbability(int doc, int topic, int type);

  double GetAverageTopicEntropy();
  void Allocate();
  void Read(std::istream &);
  void RandomInit(int **node_labels_ = NULL);
  void Free();
  void DebugDisplay(std::ostream &stream = std::cout);

  void SaveTopicDistributions(std::ostream &);
  void SaveTopicLabels(std::ostream &os);
  void SavePredTargets(ostream &os);
  void Save(const string &prefix);
  void Setup(const string &file);

  void CheckIntegrity();
  void InitializeThetaEntropy();

  void CalculateRealTargetMSE(string prefix, Stats *stats);

};

#endif
