#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <stdlib.h>

class Options;

using namespace std;

enum RealDistr{NONE, GAUSSIAN, BETA};

class Config {
 public:

  int n_topics_;
  int n_entity_types_;      // num of types of entities
                            // like words, proteins, authors etc.

  int *vocab_size_;         // size of vocab for each type of entity.
  int *entity_weight_;      // relative weight for each entity.
  int n_real_valued_attrs_; // how many real valued attributes in the dataset.
  int n_real_targets_     ; // how many real valued targets to train slda on.

  RealDistr model_real_;    // are we fitting a normal/gaussian too? which distr?
  int model_links_;         // are we modelling a link corpus too?
  int model_targets_;       // slda style target modeling?

  double alpha_;           
  double *beta_;            // sym dir prior for each entity type.
  
  int link_attr_[2];        // which types of entities are being linked?
  double link_alpha_;       // alpha for topic pair distribution for links.
  int link_weight_;         // how important is each link compared to a doc.
  int lit_weight_;          // how important is each doc.

  double off_diagonal_discount_; // by how much do we penalized off diagonal blocks?

  double real_weight_;      // how important are real valued attrs 

  bool   mixedness_constraint_; 
  double mixedness_variance_;
  double mixedness_penalty_;
  bool   balance_constraint_; 
  double balance_variance_;
  double balance_penalty_;
  bool   theta_constraint_;
  double theta_variance_;
  double theta_penalty_;
  bool   md_theta_constraint_;
  double md_theta_variance_;
  double md_theta_penalty_;
  bool   md_mixed_constraint_; 
  double md_mixed_variance_;
  double md_mixed_penalty_;
  int    volume_constraint_;
  double volume_variance_;
  double volume_penalty_;

  int md_n_domains_;
  string md_splits_string_;
  int *md_splits_;
  int *md_split_start_indexes_;
  int TopicToDomain(int topic_id) const {
    for (int i = 1; i < md_n_domains_ + 1; ++i) {
      if (topic_id < md_split_start_indexes_[i])
        return i - 1;
    }
    return md_n_domains_;
  }
  string md_probs_string_;
  double md_probs_[3]; // within-domain, general, other-domain

  bool md_seeds_;

  void DebugDisplay(ostream &os = std::cout);
  int GetNumTrainingDocs() const {return n_docs_;}
  int GetNumTestDocs() const {return n_test_docs_;}
  int GetNumTrainingLinks() const {return n_train_links_;}
  int GetNumTestLinks() const {return n_test_links_;}

  void ReadConfigMap(ifstream &ifs);
  void SetConfigValues();
  string GetConfigValue(const string &key, const string &def, const string &desc = "");

  Config(string config_file, Options &opt);
  void CheckOptions();

  ~Config();
  int n_docs_;        // num of docs in test/main corpus.
  int n_test_docs_;   // num of docs in corpus.

  int n_train_links_; // num of links in train corpus.
  int n_test_links_;  // num of links in test corpus.

  map<string, string> file_options_;
  Options &options_;

  string train_file_;
  string test_file_;
  string link_train_file_;
  string link_test_file_;
  string output_prefix_;

  int n_iterations_;
  int n_runs_;
  int n_sample_iterations_;
  int n_avg_;

  double node_label_randomness_; // while initializing with given node labels
  double clamp_rigidity_;        // if node labels are given, how much clamping
                                 // are we to allow during Gibbs
  bool use_node_labels_;
  string node_label_file_;
  string input_topic_file_;
  bool use_input_topics_;
  bool use_fake_input_topics_;  // if fake_input_topics is set, pi and theta is
                                // sampled before MCMC with random topics to
                                // enable comparison with input topics without
                                // handicap of fewer sampling iterations
  string true_label_file_;                                

  bool hungarian_flag_;
  bool nmi_flag_;
  bool knn_flag_;

  bool check_integrity_;

  map<string, string> all_options_;
  map<string, string> option_desc_;
  void Help();
  void Dump(ostream &os = std::cout);

  int max_option_length_;
  vector<string> order_;

  bool fast_lda_;
};

#endif
