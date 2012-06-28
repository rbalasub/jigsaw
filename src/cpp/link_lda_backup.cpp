/*******************************************
link_lda.cpp - Implements a Gibbs sampler for a goulash of Link LDA, Topics over
Time and Stochastic block model

Ramnath Balasubramanyan (rbalasub@cs.cmu.edu)
Language Technologies Institute, Carnegie Mellon University
*******************************************/

#include <algorithm>
#include <iostream>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "gsl/gsl_randist.h"
#include <cmath>
#include <stdlib.h>
#include <time.h>

using namespace std;

// Modified for links

// We deal with the unseen tokens in test data in the following manner.  While
// training, add a new token to dictionary called UNK - id size_vocab.  During
// test, while converting test data, assign this id to new tokens.  
// TODO: 1.  Extend training code to estimate one extra parameter per type.  
// Note: 2. fix outside code to use this id for unseen tokens in test data.
// Note: Vocab size continues to refer to the size of the actual dictionary
// i.e. UNK is not counted as part of the vocab.

// For now, we use a suboptimal method. The dictionary is constructed from the
// union of train and test set. This has the effect of the frequency based
// token filtering having visibility into the test set which strictly is a
// no-no. However, this should not provide any signficant advantage to the
// training, so we'll continue until we fix it later, thereby earning a
// pedantically correct badge.

// NOTE: All word and entity indexes in input must be 0 based
// Nice to have:
// TODO: STYLE C++ - why is config ref in model and ptr in Corpus
// TODO: STYLE split into class files
// TODO: STYLE rewrite DebugDisplay without call to DisplayMatrix

// More imp:
// TODO: STEP read in previous counts for c_tv for future inference
// TODO: STEP xv

// todo: update gaussians after every word to "forget" stats about that word.
// Actually scratch that, we are generating one timestamp per doc using our
// weighting trick.  so it does not matter that we are not reestimating time
// parameters after every word.

int debug_time = 0;
enum RealDistr{NONE, GAUSSIAN, BETA};

class Options {
// TODO: STYLE check if all options are required 2. types of values entered 3,
 public:
  map<string, string> user_flags_;

  Options(int argc, char **argv) { ParseCommandLine(argc, argv); }
  void ParseCommandLine(int, char **);
  string GetStringValue(const string&, const string&);
  int GetIntValue(const string&, int);

  void DebugDisplay(ostream &os);
};

void Options::DebugDisplay(ostream &os = std::cout) {
  for (map<string, string>::iterator i = user_flags_.begin();
      i != user_flags_.end(); ++i) {
    os << i->first << " : " << i->second << endl;
  }
  os << endl;
}

void Options::ParseCommandLine(int argc, char **argv) {
  bool is_prev_token_a_directive = false;
  string prev_token;
  for (int i = 1; i < argc; ++i) {
    string cur_token = argv[i];
    bool is_cur_token_a_directive = (cur_token.substr(0,2) == "--");
    if (is_cur_token_a_directive)
      cur_token = cur_token.substr(2, cur_token.size() - 2);

    if (is_prev_token_a_directive && is_cur_token_a_directive) {
      user_flags_[prev_token] = "YES";
    }
    else if (is_prev_token_a_directive && !is_cur_token_a_directive) {
      user_flags_[prev_token] = cur_token;
    }
    else if (!is_prev_token_a_directive && is_cur_token_a_directive) {
      // do nothing for now
    }
    else if (!is_prev_token_a_directive && !is_cur_token_a_directive) {
      cerr << "Command line options are ill formatted\n";
      exit(1);
    }

    prev_token = cur_token;
    is_prev_token_a_directive = is_cur_token_a_directive;
  }
  if (is_prev_token_a_directive)
    user_flags_[prev_token] = "YES";
}

string Options::GetStringValue (const string &key,
                                const string &default_value = "NO") {
  string val;
  if ((val = user_flags_[key]) == "") {
    if (default_value != "NO") {
      return default_value;
    } else {
      cerr << "Please supply " << key << endl;
      exit(1);
    }
  } else {
    return val;
  }
}

int Options::GetIntValue(const string &key, int default_value = -1) {
  string val;
  if ((val = user_flags_[key]) == "") {
    if (default_value != -1) {
      return default_value;
    } else {
      cerr << "Please supply " << key << endl;
      exit(1);
    }
  } else {
    return atoi(val.c_str());
  }
}

// Get a sample from Unif([0...max-1])
int UniformSample(int max) {
  return static_cast<int>((rand() * 1.0 / RAND_MAX) * max);
}

template <class T>
void DisplayMatrix (T **matrix, int m, int n,
                    char *label, bool sum, ostream &cout) {
  cout << "============== " << label << "===========\n";
  double *tot = new double[m];
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      cout << matrix[i][j] << ' ';
    }
    tot[i] = accumulate(matrix[i], matrix[i] + n, 0.0);
    if (sum)
      cout << "  Sum: " << tot[i] << '\n';
    else
      cout << '\n';
  }
  if (sum)
    cout << "Total = " << accumulate(tot, tot + m, 0.0) << "\n\n";
  else
    cout << "\n";

  delete[] tot;
}

template <class T>
void DisplayColumn (T *column, int m, char *label, ostream &cout) {
  cout << "============== " << label << "===========\n";
  for (int i = 0; i < m; ++i)
    cout << column[i] << '\n';
  cout << "Total " << accumulate(column, column + m, 0.0) << "\n\n";
}

template <class T>
void DisplayMatrix (T **matrix, int m, int *n,
                    char *label, bool sum, ostream &cout) {
  cout << "============== " << label << "===========\n";
  double *tot = new double[m];
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n[i]; ++j) {
      cout << matrix[i][j] << ' ';
    }
    tot[i] = accumulate(matrix[i], matrix[i] + n[i], 0.0);
    if (sum)
      cout << "  Sum: " << tot[i] << '\n';
    else
      cout << '\n';
  }
  if (sum)
    cout << "Total = " << accumulate(tot, tot + m, 0.0) << "\n\n";
  else
    cout << "\n";
  delete[] tot;
}

class Config {
 public:

  void ReadConfig(istream &is);

  int n_topics_;
  int n_entity_types_;      // num of types of entities
                            // like words, proteins, authors etc.

  int *vocab_size_;         // size of vocab for each type of entity.
  int *entity_weight_;      // relative weight for each entity.
  int n_real_valued_attrs_; // how many real valued attributes in the dataset.
  RealDistr model_real_;    // are we fitting a normal/gaussian too? which distr?
  int model_links_;         // are we modelling a link corpus too?

  double *alpha_;           // sym dir prior for each entity type.
  double beta_;
  
  int link_attr_[2];        // which types of entities are being linked?
  double link_alpha_;       // alpha for topic pair distribution for links.
  int link_weight_;         // how important is each link compared to a doc.

  int diagonal_blocks_;     // do we model only diagonal blocks?
  double off_diagonal_discount_; // by how much do we penalized off diagonal blocks?

  double real_weight_;      // how important are real valued attrs 

  void DebugDisplay();
  int GetNumTrainingDocs() const {return n_docs_;}
  int GetNumTestDocs() const {return n_test_docs_;}
  int GetNumTrainingLinks() const {return n_train_links_;}
  int GetNumTestLinks() const {return n_test_links_;}

  Config(int n, RealDistr m, int l = 0):
      n_topics_(n),
      model_real_(m),
      model_links_(l) {
    link_attr_[0] = -1;
    link_attr_[1] = -1;
  }
  ~Config();
  int n_docs_;        // num of docs in test/main corpus.
  int n_test_docs_;   // num of docs in corpus.

  int n_train_links_; // num of links in train corpus.
  int n_test_links_;  // num of links in test corpus.
};

void Config::DebugDisplay() {
  cout << "Topics: " << n_topics_ << endl;
  cout << "Docs: " << n_docs_ << endl;
  cout << "Types: " << n_entity_types_ << endl;
  for (int i = 0; i < n_entity_types_; ++i) {
    cout << "Vocab = " << vocab_size_[i] << " ";
    cout << "Weight = " << entity_weight_[i] << " ";
    cout << endl;
  }
  cout << "Link Attr: " << link_attr_[0] << "->" << link_attr_[1] << endl;
}

Config::~Config() {
  delete[] alpha_;
  delete[] vocab_size_;
  delete[] entity_weight_;
}

int ReadLine(istream &ifs) {
  int val;
  string config_line;
  getline(ifs, config_line);
  istringstream iss(config_line);
  iss >> val;
  return val;
}

void Config::ReadConfig(istream &ifs) {
  // Format: nDocs nTestDocs nTypesOfEntities vocab_sizes weights time_in_data
  n_docs_         = ReadLine(ifs);
  n_test_docs_    = ReadLine(ifs);
  n_entity_types_ = ReadLine(ifs);

  vocab_size_     = new int[n_entity_types_];
  entity_weight_  = new int[n_entity_types_];
  
  string config_line;
  getline(ifs, config_line);
  istringstream iss_1(config_line);
  for (int i = 0; i < n_entity_types_; ++i)
    iss_1 >> vocab_size_[i];

  getline(ifs, config_line);
  istringstream iss_2(config_line);
  for (int i = 0; i < n_entity_types_; ++i)
    iss_2 >> entity_weight_[i];

  n_real_valued_attrs_ = ReadLine(ifs);

  //link_attr_ = ReadLine(ifs);
  getline(ifs, config_line);
  istringstream iss_3(config_line);
  iss_3 >> link_attr_[0] >> link_attr_[1]; // Watchout for old configs with 
                                           // only one link attr.

  n_train_links_ = ReadLine(ifs);
  n_test_links_ = ReadLine(ifs);
  link_alpha_ = ReadLine(ifs);
  link_weight_  = ReadLine(ifs);

  alpha_ = new double[n_entity_types_];
  for (int i = 0; i < n_entity_types_; ++i)
    alpha_[i] = 1.0;
  beta_ = 1.0;
}

class Model;
class Links {
 public:

  const Config *config_; // a fully configured Config object.
  int n_links_;
  Links(const Config *c, int n) : 
       config_(c),
       n_links_(n) {
  }

  int **links_; // ndocs x 2
  int **link_topic_assignments_; // ndocs x 2
  int **link_topic_pair_counts_; //K x K - distribution over pairs of topics.

  void Allocate();
  void Read(istream &, Model *model);
  void Free();
  void RandomInit();

  void SaveTopics(ostream &os);
  friend class Model;
  //void DebugDisplay(ostream &);
  //void SaveTopicDistributions(ostream &);
  //void SaveAsVector(ostream &);
};


void Links::RandomInit() {
  for (int i = 0; i < config_->n_topics_; ++i) {
    for (int j = 0; j < config_->n_topics_; ++j) {
      link_topic_pair_counts_[i][j] = 0;
    } // end topic 2
  } // end topic 1

  for (int i = 0; i < n_links_; ++i) {
    int t1 = UniformSample(config_->n_topics_);
    int t2 = UniformSample(config_->n_topics_);
    link_topic_assignments_[i][0] = t1;
    link_topic_assignments_[i][1] = t2;
    link_topic_pair_counts_[t1][t2]++;
  } // end all links
}

void Links::Allocate() {
  // allocate space for corpus wide topic pair distribution.
  link_topic_assignments_ = new int*[n_links_];
  links_ = new int*[n_links_];
  for (int i = 0; i < n_links_; ++i) {
    link_topic_assignments_[i] = new int[2];
    links_[i] = new int[2];
  }

  link_topic_pair_counts_ = new int*[config_->n_topics_];
  for (int i = 0; i < config_->n_topics_; ++i) {
    link_topic_pair_counts_[i] = new int[config_->n_topics_];
  }
}

void Links::Free() {
  for (int i = 0; i < n_links_; ++i) {
    delete[] link_topic_assignments_[i]; 
    delete[] links_[i]; 
  }
  for (int i = 0; i < config_->n_topics_; ++i) {
    delete[] link_topic_pair_counts_[i];
  }
  delete[] link_topic_assignments_;
  delete[] links_;
  delete[] link_topic_pair_counts_;
}

class Corpus {
 public:

  const Config *config_; // a fully configured Config object.
  int n_docs_;

  Corpus(const Config *c, int n);

  int **doc_num_words_; // ntypes x ndocs
  int ***corpus_words_; // ntypes x ndocs x nwords
  
  double **real_values_; // ndocs x number of real valued attributes.
  bool   **real_flags_;  // ndocs x number of real valued attrs - are they valid?

  int ***word_topic_assignments_; // ntypes x ndocs x nwords
  int **counts_docs_topics_;      // ndocs x ntopics
  void Allocate();
  void Read(istream &, Model *model);
  void RandomInit();
  void Free();
  void DebugDisplay(ostream &);

  void SaveTopicDistributions(ostream &);
  void SaveAsVector(ostream &);
};

void Corpus::RandomInit() {
  // Clean out 
  for (int doc = 0; doc < n_docs_; ++doc) {
    for (int topic = 0; topic < config_->n_topics_; ++topic) {
      counts_docs_topics_[doc][topic] = 0;
    }
  }

  // Assign random topics to words
  for (int type = 0; type < config_->n_entity_types_; ++type) {
    for (int doc = 0; doc < n_docs_; ++doc) {
      for (int word = 0; word < doc_num_words_[type][doc]; ++word) {
        int random_topic = UniformSample(config_->n_topics_);
        word_topic_assignments_[type][doc][word] = random_topic;
        counts_docs_topics_[doc][random_topic] += config_->entity_weight_[type];
      } // end words
    } // end docs
  } // end type
}

Corpus::Corpus(const Config *c, int n)
    : config_(c),
      n_docs_(n) {
}

/*void Corpus::SaveAsVector(ostream &os) {
  for (int doc = 0; doc < config->n_docs_; ++doc) {
    int sum = 0;
    for (int type = 0; type < config->n_entity_types_; ++type) 
      sum += doc_num_words_[type][doc] * config->entity_weight_[type];
    for (int type = 0; type < config->n_entity_types_; ++type) {
      for (int i = 0; i < doc_num_words_[type][doc]; ++i) {
        double normalized = corpus_words_[type][doc][i]
      }
    }
  }
}*/

void Links::SaveTopics(ostream &os) {
  for (int i = 0; i < n_links_; ++i) {
    os << link_topic_assignments_[i][0] << '\t' << link_topic_assignments_[i][1] << '\n';
  }
}

void Corpus::SaveTopicDistributions(ostream &os) {
  for (int i = 0; i < n_docs_; ++i) {
    double normalizing_constant =
        accumulate(counts_docs_topics_[i],
                   counts_docs_topics_[i] + config_->n_topics_,
                   0) * 1.0;
    for (int j = 0; j < config_->n_topics_; ++j) {
      double normalized_ratio = counts_docs_topics_[i][j] / normalizing_constant;
      os << normalized_ratio << '\t'; 
    } // end topics
    os << '\n';
  } // end docs
}

class Model {
 public:
  const Config &config_;
  Model(const Config &c):config_(c) {}
  Model(const Model &); // copy constructor.
  // store how many instances of each word/entity/etc in a topic
  double ***counts_topic_words_; // [nTypesOfEntities][nTopics][vocab_size]
  double **sum_counts_topic_words_;

  double ***beta_parameters_;        // nattrs X ntopics x 2
  double ***gaussian_parameters_;    // nattrs X ntopics x 2
  double ***real_stats_;             // nattrs X ntopics x 2
  double **topic_allocation_counts_; // ntopics x n_real_attrs 
                                     // how many words with valid attrs per topic

  void Allocate();
  void Add(const Model &);
  void Normalize();
  Model*               MCMC(Corpus &corpus,
                            Links *links,
                            int n_iterations,
                            int n_avg,
                            bool unseen,
                            bool debug,
                            bool silent);

  void InferLinkDistribution(Links &links, bool debug);
  void ComputePerplexity(Corpus &corpus, double *);
  double ComputeLinkPerplexity(Links &links);
  void EstimateBeta();
  void Save(const string&);
  void Free();
  void DebugDisplay(ostream &os);

  void AddDocument(Corpus &c, int doc);
  void RemoveDocument(Corpus &c, int doc);

  void AddLinks(Links &links, bool remove);
  void RemoveLinks(Links &links);
  void TestLinks(Links &links, int n_iterations, bool silent);

  Corpus *test_corpus_;
  Links  *test_link_corpus_;
  void SetTestCorpus(Corpus *c) {test_corpus_ = c;}
  void SetLinkTestCorpus(Links *l) {test_link_corpus_ = l;}

};

template<class T>
void Copy(T ***src, T***dest, int d1, int d2, int *d3) {
  for (int i = 0; i < d1; ++i)
    for (int j = 0; j < d2; ++j)
      for (int k = 0; k < d3[i]; ++k)
        dest[i][j][k] = src[i][j][k];
}

template<class T>
void Copy(T ***src, T***dest, int d1, int d2, int d3) {
  for (int i = 0; i < d1; ++i)
    for (int j = 0; j < d2; ++j)
      for (int k = 0; k < d3; ++k)
        dest[i][j][k] = src[i][j][k];
}

template<class T>
void Copy(T **src, T**dest, int d1, int d2) {
  for (int i = 0; i < d1; ++i)
    for (int j = 0; j < d2; ++j)
      dest[i][j] = src[i][j];
}

void Model::EstimateBeta() {
  if (debug_time) {
    cout << "Time Statistics\n";
    cout << "----------------" << endl;
  }

  for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
    double sum_allocated_words = 0; //  Basically number of words in the dataset.
                                    //  Used only for debugging time weirdness.
    double real_stats_sum_0 = 0.0;  //  also for debugging.
    double real_stats_sum_1 = 0.0;  //  also for debugging.

    for (int i = 0; i < config_.n_topics_; ++i) {
      if (topic_allocation_counts_[attr][i] == 0) {
        cerr << "No word in topic " << i << " for attr " << attr << endl;
      }
      double sample_mean     =
        real_stats_[attr][i][0] / topic_allocation_counts_[attr][i];
      double sample_variance =
        real_stats_[attr][i][1] / topic_allocation_counts_[attr][i] -
            (sample_mean * sample_mean);

      if (debug_time) {
        cout << "Attr " << attr << " Topic " << i << " : "
             << real_stats_[attr][i][0] << "->" << sample_mean << " "
             << real_stats_[attr][i][1] << "->" << sample_variance << " "
             << topic_allocation_counts_[attr][i] << endl;
        sum_allocated_words += topic_allocation_counts_[attr][i];
        real_stats_sum_0    += real_stats_[attr][i][0];
        real_stats_sum_1    += real_stats_[attr][i][1];
      }

      beta_parameters_[attr][i][0] = sample_mean *
          ((sample_mean * (1 - sample_mean))/(sample_variance + 1e-20) - 1);
      beta_parameters_[attr][i][1] = (1 - sample_mean) *
          ((sample_mean * (1 - sample_mean))/(sample_variance + 1e-20)- 1);

      gaussian_parameters_[attr][i][0] = sample_mean;
      gaussian_parameters_[attr][i][1] = sample_variance;
    } // end topics

    if (debug_time) {
      cout << "Total words is " << sum_allocated_words << endl;
      cout << "Sum of timestamp values " << real_stats_sum_0 << endl;
      cout << "Sum of timestamp^2 values " << real_stats_sum_1 << endl;
      for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
        DisplayMatrix(beta_parameters_[attr], config_.n_topics_, 2, "Beta", false, cout);
      }
    }
  } //end attr
  // TODO: adding weird smoothing ... FIX.


  // This block of code was written to validate the operation of the sufficient
  // stats collection for time related stuff. TODO: remove this later.
  /*
  if (debug_time >= 3) {
    double **tmp = new double*[config_.n_topics_];
    for (int i = 0; i < config_.n_topics_; ++i) {
      tmp[i] = new double[2];
      tmp[i][0] = 0.0;
      tmp[i][1] = 0.0;
    }
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    for (int doc = 0; doc < corpus.n_docs_; ++doc) {
      for (int topic = 0; topic < config_.n_topics_; ++topic) {
        tmp[topic][0] += corpus.counts_docs_topics_[doc][topic] * corpus.timestamps_[doc];
        tmp[topic][1] += corpus.counts_docs_topics_[doc][topic]
                         * corpus.timestamps_[doc] * corpus.timestamps_[doc];
        c += corpus.counts_docs_topics_[doc][topic];
      }
    }
    for (int topic = 0; topic < config_.n_topics_; ++topic) {
      if ((abs(tmp[topic][0] - real_stats_[topic][0]) >= 1.0e-04) ||
          (abs(tmp[topic][1] - real_stats_[topic][1]) >= 1.0e-04)) {
        cout << "Whoa!" << topic <<  endl;
        cout << tmp[topic][0] << " == " << real_stats_[topic][0]
             << " - " << (tmp[topic][0] - real_stats_[topic][0]) << endl;
        cout << tmp[topic][1] << " == " << real_stats_[topic][1]
             << " - " << (tmp[topic][1] - real_stats_[topic][1]) << endl;
      }
      a += tmp[topic][0];
      b += tmp[topic][1];
      delete[] tmp[topic];
    }
    cout << "Sums " << a << " " << b <<  " " << c << endl;
    delete[] tmp;
  }
  */
}

void Corpus::Allocate() {
  corpus_words_ = new int**[config_->n_entity_types_];
  //timestamps_ = new double[n_docs_];
  real_values_ = new double*[n_docs_];
  real_flags_  = new bool*[n_docs_];
  word_topic_assignments_ = new int**[config_->n_entity_types_];
  // effectively the sum of each row in corpus_words but we ask for it
  // redundantly for efficiency.
  doc_num_words_ = new int*[config_->n_entity_types_];
  for (int i = 0; i < config_->n_entity_types_; ++i) {
    corpus_words_[i] = new int*[n_docs_];
    word_topic_assignments_[i] = new int*[n_docs_];
    doc_num_words_[i] = new int[n_docs_];
  }

  counts_docs_topics_ = new int*[n_docs_]; //[nDocs][nTopics]
  for (int i = 0; i < n_docs_; ++i) {
    counts_docs_topics_[i] = new int[config_->n_topics_];
    for (int j = 0; j < config_->n_topics_; ++j)
      counts_docs_topics_[i][j] = 0;
    real_values_[i] = new double[config_->n_real_valued_attrs_];
    real_flags_[i] = new bool[config_->n_real_valued_attrs_];
  }
}

void Corpus::DebugDisplay(ostream &os = std::cout) {
  for (int type = 0; type < config_->n_entity_types_; ++type) {
    os << "=== Type " << type << " ===\n";
    DisplayMatrix(corpus_words_[type],
                  n_docs_,
                  doc_num_words_[type],
                  "corpus words",
                  false, os);
    DisplayMatrix(word_topic_assignments_[type],
                  n_docs_,
                  doc_num_words_[type],
                  "word topic assignments",
                  false, os);
  }
  DisplayMatrix(counts_docs_topics_, n_docs_, config_->n_topics_,
                "C_td", true, os);
}

// Hacky solution to initialize counts in model while reading in test set.
// Cleaner alternative is to go through corpus and initialize.

void Links::Read(istream &is, Model *model = NULL) {
  // TODO: use RandomInit and AddLinks instead of doing everything here in an
  //       ugly manner.

  string line;
  int link_ctr = 0;
  while (getline(is, line)) {
    istringstream iss(line);
    int id_1, id_2;
    int cnt_unnecessary; 
    // the links file includes a count in the first column although we know it
    // will always be 2. This maintains uniformity with the docs file to some
    // degree.
    iss >> cnt_unnecessary >> id_1 >> id_2;
    links_[link_ctr][0] = id_1;
    links_[link_ctr][1] = id_2;

    int random_topic_1 = UniformSample(config_->n_topics_);
    int random_topic_2 = UniformSample(config_->n_topics_);
    link_topic_assignments_[link_ctr][0] = random_topic_1;
    link_topic_assignments_[link_ctr][1] = random_topic_2;

    link_topic_pair_counts_[random_topic_1][random_topic_2]++;
    if (model) {
//      int link_weight =
//          config_->entity_weight_[config_->link_attr_] * config_->link_weight_;
      int link_weight = config_->link_weight_;
      int type_1 = config_->link_attr_[0];
      int type_2 = config_->link_attr_[1];
      (model->counts_topic_words_)[type_1][random_topic_1][id_1] += link_weight;
      (model->sum_counts_topic_words_)[type_1][random_topic_1] += link_weight;
      (model->counts_topic_words_)[type_2][random_topic_2][id_2] += link_weight;
      (model->sum_counts_topic_words_)[type_2][random_topic_2] += link_weight;
    }

    link_ctr++;
  } // end while - reading file.
}

void Corpus::Read(istream &ifs, Model *model = NULL) {
  // TODO: use RandomInit instead of initializing here.
  // TODO: write a function to add Corpus to Model instead of doing it here.
  string line;
  int doc_ctr = 0;
  while (getline(ifs, line)) {
    istringstream iss(line);

    for (int type = 0; type < config_->n_entity_types_; ++type) {
      int word_cnt; // is not words always, could be any other kind of entity.
      iss >> word_cnt;
      doc_num_words_[type][doc_ctr] = word_cnt;
      corpus_words_[type][doc_ctr] = new int[word_cnt];
      word_topic_assignments_[type][doc_ctr] = new int[word_cnt];
      for (int i = 0; i < word_cnt; ++i) {
        int word_id;
        iss >> word_id;
        corpus_words_[type][doc_ctr][i] = word_id;
        int random_topic = UniformSample(config_->n_topics_);
        word_topic_assignments_[type][doc_ctr][i] = random_topic;
        if (model) {
          (model->counts_topic_words_)[type][random_topic][word_id] +=
              config_->entity_weight_[type];
          (model->sum_counts_topic_words_)[type][random_topic] +=
              config_->entity_weight_[type];
        }
        counts_docs_topics_[doc_ctr][random_topic] +=
            config_->entity_weight_[type];
      } // end reading entities
    } // end types
    for (int real_attr = 0; real_attr < config_->n_real_valued_attrs_;
         ++real_attr) {
      string s;
      iss >> s;
      if (s == "NA" || s == "") {
        real_flags_[doc_ctr][real_attr] = 0;
        real_values_[doc_ctr][real_attr] = 0;
      } else {
        real_values_[doc_ctr][real_attr] = atof(s.c_str());
        real_flags_[doc_ctr][real_attr] = 1;
     //   cout << s.c_str() << " ";
      }
    } // read in real valued attributes
    //cout << "\n";

    if (model && config_->model_real_) {
      for (int type = 0; type < config_->n_entity_types_; ++type) {
        for (int i = 0; i < doc_num_words_[type][doc_ctr]; ++i) {
          int top = word_topic_assignments_[type][doc_ctr][i];
          for (int attr = 0; attr < config_->n_real_valued_attrs_; ++attr) {
            if (real_flags_[doc_ctr][attr]) {
              model->topic_allocation_counts_[attr][top] +=
                  config_->entity_weight_[type];
              model->real_stats_[attr][top][0] +=
                  config_->entity_weight_[type] * real_values_[doc_ctr][attr] ;
              model->real_stats_[attr][top][1] +=
                  (config_->entity_weight_[type]) * (config_->entity_weight_[type]) *
                  (real_values_[doc_ctr][attr] * real_values_[doc_ctr][attr]);
            }
          }
        } // end words
      } // end type 
    } // end init time ds

    doc_ctr++;
  } // end file
}

void Corpus::Free() {
  // Matrices - Ctd Cvt sCvt
  // Inference - wta Ctd
  // Corpus - corpus_words doc_num_words (CONSTANT)
  for (int type = 0; type < config_->n_entity_types_; ++type) {
    for (int i = 0; i < n_docs_; ++i) {
      delete[] word_topic_assignments_[type][i];
      delete[] corpus_words_[type][i];
      if (type == 0) { // this is not type specific.
        delete[] counts_docs_topics_[i];
      }
    }

    delete[] word_topic_assignments_[type];
    delete[] corpus_words_[type];
    delete[] doc_num_words_[type];
  }
  for (int doc = 0; doc < n_docs_; ++doc) {
    delete[] real_values_[doc];
  }
  delete[] counts_docs_topics_;
  delete[] doc_num_words_;
  delete[] word_topic_assignments_;
  delete[] corpus_words_;
  delete[] real_values_;
}

Model::Model(const Model &base) : config_(base.config_) {
  Allocate();
  Copy(base.counts_topic_words_, counts_topic_words_, config_.n_entity_types_, config_.n_topics_, config_.vocab_size_);
  Copy(base.sum_counts_topic_words_, sum_counts_topic_words_, config_.n_entity_types_, config_.n_topics_);

  if (config_.model_real_) {
    Copy(base.beta_parameters_, beta_parameters_, config_.n_real_valued_attrs_, config_.n_topics_, 2);
    Copy(base.gaussian_parameters_, gaussian_parameters_, config_.n_real_valued_attrs_, config_.n_topics_, 2);
    Copy(base.real_stats_, real_stats_, config_.n_real_valued_attrs_, config_.n_topics_, 2);

    Copy(base.topic_allocation_counts_, topic_allocation_counts_, config_.n_real_valued_attrs_, config_.n_topics_);
  }
}

void Model::Add(const Model &base) {
  if (config_.model_real_) {
    // TODO: Averaging of real params
    // TODO: Normalizing real params
  }

  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int i = 0; i < config_.n_topics_; ++i) {
      sum_counts_topic_words_[type][i] += base.sum_counts_topic_words_[type][i] ;
      for (int j = 0; j < config_.vocab_size_[type]; ++j) {
        counts_topic_words_[type][i][j] += base.counts_topic_words_[type][i][j];
      }
    } // end topics
  } // end types

}

void Model::Allocate() {
  // Initialize count matrices to 0
  counts_topic_words_ = new double**[config_.n_entity_types_];

  if (config_.model_real_) {
    // allocate space for time related data structures.

    beta_parameters_ = new double**[config_.n_real_valued_attrs_];
    gaussian_parameters_ = new double**[config_.n_real_valued_attrs_];
    real_stats_ = new double**[config_.n_real_valued_attrs_];

    topic_allocation_counts_ = new double*[config_.n_real_valued_attrs_];
    for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
      beta_parameters_[attr]         = new double*[config_.n_topics_];
      gaussian_parameters_[attr]     = new double*[config_.n_topics_];
      real_stats_[attr]              = new double*[config_.n_topics_];
      topic_allocation_counts_[attr] = new  double[config_.n_topics_];

      for (int i = 0; i < config_.n_topics_; ++i) {
        beta_parameters_[attr][i] = new double[2];
        gaussian_parameters_[attr][i] = new double[2];
        real_stats_[attr][i] = new double[2];

        topic_allocation_counts_[attr][i] = 0;
        real_stats_[attr][i][0]           = 0.0;
        real_stats_[attr][i][1]           = 0.0;
        beta_parameters_[attr][i][0]      = 0.0;
        beta_parameters_[attr][i][1]      = 0.0;
        gaussian_parameters_[attr][i][0]  = 0.0;
        gaussian_parameters_[attr][i][1]  = 0.0;
      } // end topics
    } // end attrs
  } // end initializing real structs

  // [nTypesOfEntities][nTopics]sum of matrix above for efficiency
  sum_counts_topic_words_ = new double*[config_.n_entity_types_];
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    counts_topic_words_[type] = new double*[config_.n_topics_];
    sum_counts_topic_words_[type] = new double[config_.n_topics_];
    for (int i = 0; i < config_.n_topics_; ++i) {
      counts_topic_words_[type][i] = new double[config_.vocab_size_[type]];
      sum_counts_topic_words_[type][i] = 0;
      for (int j = 0; j < config_.vocab_size_[type]; ++j)
        counts_topic_words_[type][i][j] = 0;
    } // end topics
  } // end types
}

// TODO: fix inefficieny of doing Add/Remove so many times ... 
  Model *               Model::MCMC(Corpus &corpus,      
                                 Links  *links = NULL,
                                 int n_iterations = 20,
                                 int n_avg = 10,
                                 bool unseen = false, 
                                 bool debug = false,
                                 bool silent = false) {
  double      *perplexity = new double[config_.n_entity_types_];
  long double *time_prob  = new long double[config_.n_topics_];

  vector<Model *> models; // for saving n_avg number of models for averaging
  for (int iteration = 0; iteration < n_iterations + n_avg; ++iteration) {
    if (!silent) {
      cout << "Iteration " << iteration << " ...  "; cout.flush();
    }
    /*ostringstream oss;
    oss << "dbg.time-" << "iter-" << iteration;
    Save(oss.str());*/
    
    if (config_.model_real_)
      EstimateBeta();

    for (int doc = 0; doc < corpus.n_docs_; ++doc) {
      if (unseen)
        AddDocument(corpus, doc);
      int zero_cnt = 0;
      if (config_.model_real_) {
        // If this is test-time, estimate beta after every doc because we
        // "forget" this document after sampling topics for all words in the
        // doc  and the effect of this document on time will be null.  While
        // this is expensive, this is the price we pay in the current setup.
        // In contrast during train time, we estimate once per iteration.
        // During test, we estimate once per doc per iteration.
        if (unseen) 
          EstimateBeta();
        double weight_of_doc = accumulate(corpus.counts_docs_topics_[doc],
                                          corpus.counts_docs_topics_[doc] + config_.n_topics_,
                                          0) * 1.0;
        if (weight_of_doc == 0)
          weight_of_doc = 1; // when a document has only real values and no words.

        for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
          double t = corpus.real_values_[doc][attr];
          for (int topic = 0; topic < config_.n_topics_; ++topic) {
            if (attr == 0) 
              time_prob[topic] = 1.0; // initialization
            if (!corpus.real_flags_[doc][attr]) { // timestamp not present in this document.
              time_prob[topic] *= 1.0;
              continue;
            }
            long double prob = 0.0;
            if (config_.model_real_ == BETA) {
              prob =
                  pow(1.0 - t, beta_parameters_[attr][topic][0] - 1) *
                  pow(t, beta_parameters_[attr][topic][1] - 1);
            } else { // Use Gaussian by default otherwise.
              // cout << "Estimating for t = " << t << endl;
              prob = pow(gaussian_parameters_[attr][topic][1], -0.5 * config_.real_weight_ / weight_of_doc) *
                     exp(static_cast<long double>(-1 * pow((t - gaussian_parameters_[attr][topic][0]), 2) /
                         (2 * gaussian_parameters_[attr][topic][1]) *
                         (config_.real_weight_ / weight_of_doc)));  // 100 is to remove inf bugs.
                          
              /*prob = gsl_ran_gaussian_pdf ((t - gaussian_parameters_[attr][topic][0]),
                                           gaussian_parameters_[attr][topic][1]);
              prob = pow(prob, 1.0 / weight_of_doc); */

              // 

              /* if (isinf(prob) || prob == 0) {
                cout << "Reaching infinity t  = " << t <<  " Prob = " << prob << endl;
                cout << gaussian_parameters_[attr][topic][0] << " " << gaussian_parameters_[attr][topic][1] << endl;
                cout << t - gaussian_parameters_[attr][topic][0] << endl;
                cout << weight_of_doc << endl;
                cout << "Exp " << (-1 * pow((t - gaussian_parameters_[attr][topic][0]), 2) /
                                     (2 * gaussian_parameters_[attr][topic][1]) *
                                     (1.0 / weight_of_doc)  // 100 is to remove inf bugs.
                                     ) << endl;
                cout << pow(gaussian_parameters_[attr][topic][1], -0.5 / weight_of_doc) << endl;
              } */
            }
            // Weight the time_prob to control the effect of time v/s generating
            // entities.
            // Updated: blended in above.
            // prob = pow(prob, 6.0 / weight_of_doc);
            time_prob[topic] *= prob;
          } // end topics
          if (debug_time >= 2) {
            cout << "Doc " << doc << " Time: " << t << endl;
            DisplayColumn(time_prob, config_.n_topics_, "Time prob", cout);
          }
        } // end attr

        for (int topic = 0; topic < config_.n_topics_; ++topic) {
          if (time_prob[topic] == 0)
            zero_cnt++;
        }
        
        /*if (zero_cnt == config_.n_topics_) {
          for (int topic = 0; topic < config_.n_topics_; ++topic) {
            time_prob[topic] = 1.0; // if every topic has 0 score from real values, dump them.
          }
        }*/
      } // end of setting up time specific ds

      for (int type = 0; type < config_.n_entity_types_; ++type) {
        for (int word_idx = 0;
             word_idx < corpus.doc_num_words_[type][doc];
             ++word_idx) {
          long double *cdf = new long double[config_.n_topics_];

          int cur_topic = corpus.word_topic_assignments_[type][doc][word_idx];
          int cur_wordid = corpus.corpus_words_[type][doc][word_idx];

          // remove effect of this word
          counts_topic_words_[type][cur_topic][cur_wordid]--;
          sum_counts_topic_words_[type][cur_topic]--;

          // remove effect of this word from time-related data structures
          if (config_.model_real_) {
            int wt = config_.entity_weight_[type];
            for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
              if (corpus.real_flags_[doc][attr]) {
                topic_allocation_counts_[attr][cur_topic] -= wt;
                real_stats_[attr][cur_topic][0] -= corpus.real_values_[doc][attr] * wt;
                real_stats_[attr][cur_topic][1] -=
                    (corpus.real_values_[doc][attr] * corpus.real_values_[doc][attr] * wt);
              }
            }
          }

          corpus.counts_docs_topics_[doc][cur_topic] -=
              config_.entity_weight_[type];

          // compute topic CDF.
          for (int topic = 0; topic < config_.n_topics_; ++topic) {
            long double topic_prob =
                (corpus.counts_docs_topics_[doc][topic] + config_.beta_) *
                (counts_topic_words_[type][topic][cur_wordid] + config_.alpha_[type]) /
                (sum_counts_topic_words_[type][topic] + config_.vocab_size_[type] * config_.alpha_[type]);
            if (config_.model_real_)
              topic_prob = topic_prob * time_prob[topic];

            if (topic == 0)
              cdf[topic] = topic_prob;
            else
              cdf[topic] = cdf[topic - 1] + topic_prob;
          }

          // sample topic using CDF
          long double unif_sample = rand() * 1.0 / RAND_MAX;
          int new_topic = -1;
          for (int topic = 0; topic < config_.n_topics_; ++topic) {
            if (unif_sample < (cdf[topic] / cdf[config_.n_topics_ - 1])) {
              new_topic = topic;
              break;
            }
          }

          if (new_topic == -1) {
            // if this error msg pops up, investigate, silently carrying on for
            // now to avoid exasperation if we have to quit because of this
            // after hours.
            cout << "Couldn't assign new topic - Iteration " << iteration
                 << " Doc " << doc << " Type " << type
                 << " Word " << word_idx << endl;
            cout << "Zero count " << zero_cnt << endl;
            for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
              cout  << "Attr " << attr << ": " << "Flag:" << corpus.real_flags_[doc][attr] << " " << corpus.real_values_[doc][attr] << endl;
              DisplayColumn(topic_allocation_counts_[attr], config_.n_topics_, "Topic counts", cout);
              DisplayMatrix(real_stats_[attr], config_.n_topics_, 2, "Time Stats", false, cout);
              // DisplayMatrix(beta_parameters_[attr], config_.n_topics_, 2, "Beta", false, cout);
              DisplayMatrix(gaussian_parameters_[attr], config_.n_topics_, 2, "Gaussian", false, cout);
            }
            DisplayColumn(time_prob, config_.n_topics_, "Time Prob", cout);
            DisplayColumn(cdf, config_.n_topics_, "CDF", cout);
            new_topic = cur_topic;
          }

          corpus.word_topic_assignments_[type][doc][word_idx] = new_topic;
          // add this word
          counts_topic_words_[type][new_topic][cur_wordid]++;
          sum_counts_topic_words_[type][new_topic]++;
          corpus.counts_docs_topics_[doc][new_topic] +=
              config_.entity_weight_[type];

          //if (config_.model_real_ && (corpus.timestamps_[doc] >= 0)) {
          if (config_.model_real_) {
            int wt = config_.entity_weight_[type];
            for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
              if (corpus.real_flags_[doc][attr]) {
                topic_allocation_counts_[attr][new_topic] += wt;
                real_stats_[attr][new_topic][0] += corpus.real_values_[doc][attr] * wt;
                real_stats_[attr][new_topic][1] +=
                    (corpus.real_values_[doc][attr] * corpus.real_values_[doc][attr]) * wt;
              }
            }
          }

          if (debug) {
            double prev = 0;
            double total = cdf[config_.n_topics_ - 1];
            for (int topic = 0; topic < config_.n_topics_; ++topic) {
              double old = cdf[topic];
              // only for display - normalizing to 1
              cdf[topic] = (cdf[topic] - prev) / total;
              prev = old;
            }
            cout << "Iteration " << iteration
                 << " Doc " << doc << " Word " << word_idx <<"\n\n";
            DisplayColumn(cdf, config_.n_topics_, "topic distribution", cout);
            cout << "Moved from " << cur_topic << " to " << new_topic
                 << " based on " << unif_sample << endl;
            corpus.DebugDisplay(cout);
            DebugDisplay(cout);
          }
          delete[] cdf;
        } // end words
      } // end types of words
      if (unseen)
        RemoveDocument(corpus, doc);
    } //  end docs

    if (config_.model_links_ && links) {
      InferLinkDistribution(*links, debug);
    }

    if (!silent) {
      cout << " : "; cout.flush();
    }

    ComputePerplexity(corpus, perplexity);
    if (!silent || iteration == n_iterations + n_avg - 1) {
      for (int j = 0; j < config_.n_entity_types_; ++j) {
        cout << perplexity[j] << " ";
      }
    }

    // During test inference, don't do anything related to links.
    if (!unseen && config_.model_links_ && links) {
      double link_perplexity = ComputeLinkPerplexity(*links);
      cout << "Link: " << link_perplexity << " ";
    }

    // run test inference at every k-th iteration on test_corpus.
    // Making sure we aren't in test phase, to avoid infinite loop
    // and the chaos that can ensue.
    if (!unseen && iteration >= n_iterations) {
      Model *photocopy = new Model(*this);
      photocopy->Normalize();
      models.push_back(photocopy);
    }

    if (!unseen && ((iteration % 5 == 0) || (iteration == n_iterations + n_avg - 1))) {
      cout << "Test perplexities: ";

      test_corpus_->RandomInit();
      if (config_.GetNumTestDocs() > 0) {
        int backup = debug_time;
        debug_time = 0;
        MCMC(*test_corpus_, NULL, 10, 10, true, false, true);
        debug_time = backup;
      }

      if (config_.model_links_ && config_.GetNumTestLinks() > 0) {
        test_link_corpus_->RandomInit();
        TestLinks(*test_link_corpus_, 100, true);
      }

      /*
      // Let's check to see if the DS have been restored to pristine condition
      // after testing.
      ComputePerplexity(corpus, perplexity);
      cout << "\nChecking after test ";
      for (int j = 0; j < config_.n_entity_types_; ++j) {
        cout << perplexity[j] << " ";
      } // end topics
      if (!unseen && config_.model_links_ && links) {
        double link_perplexity = ComputeLinkPerplexity(*links);
        cout << "Link: " << link_perplexity << " ";
      }
      // End of test
      */
    } // end if - test perlexity

    if (!silent)
      cout << endl;
  } // end iterations

  if (!silent)
    cout << "Done with MCMC\n";
  delete[] perplexity;
  delete[] time_prob;

  Model *average_model = new Model(config_);
  if (!unseen) {
    average_model->Allocate();
    cout << "Average over " << models.size() << " last iterations" << endl;
    for (int i = 0; i < models.size(); ++i) {
      average_model->Add(*models[i]);
      models[i]->Free();
      delete models[i];
    }
  } // end if
  return average_model;
}

void Model::AddLinks(Links &links, bool remove = false) {
  int multiplier = 1;
  if (remove)
    multiplier = -1;

  for (int i = 0; i < links.n_links_; ++i) {
    int t1 = links.link_topic_assignments_[i][0];
    int t2 = links.link_topic_assignments_[i][1];

    int e1 = links.links_[i][0];
    int e2 = links.links_[i][1];

    int type_1 = config_.link_attr_[0];
    int type_2 = config_.link_attr_[1];
//    int weight = multiplier *
//        config_.entity_weight_[config_.link_attr_] * config_.link_weight_;
    int weight = multiplier * config_.link_weight_;

    counts_topic_words_[type_1][t1][e1] += weight;
    sum_counts_topic_words_[type_1][t1] += weight;

    counts_topic_words_[type_2][t2][e2] += weight;
    sum_counts_topic_words_[type_2][t2] += weight;
  }
}

void Model::RemoveLinks(Links &links) {
  AddLinks(links, true); // remove weight instead of add and voila Add is Remove
}

void Model::TestLinks(Links &links, int n_iterations, bool silent = false) {
  AddLinks(links); // add links with random assignments to DS
  double link_perplexity;
  for (int iteration = 0; iteration < n_iterations; ++iteration) {
    InferLinkDistribution(links, 0); // Do MCMC voodoo
    link_perplexity = ComputeLinkPerplexity(links);
    if (!silent) {
      cout << "Iteration " << iteration << " ...   : "
           << "Link: " << link_perplexity << endl;
    }
  }
  if (silent)
    cout << "Link: " << link_perplexity;
  RemoveLinks(links); // clean getaway, no trace of test corpus.
}

void Model::InferLinkDistribution(Links &links, bool debug) {

  double **cdf = new double*[config_.n_topics_];
  for (int topic = 0; topic < config_.n_topics_; ++topic) {
    cdf[topic] = new double[config_.n_topics_];
  }

  int type_1 = config_.link_attr_[0]; // what entity type do these links refer
  int type_2 = config_.link_attr_[1]; // to. short var to avoid clutter in code.

  for (int i = 0; i < links.n_links_; ++i) {
    int cur_topic_1 = links.link_topic_assignments_[i][0];
    int cur_topic_2 = links.link_topic_assignments_[i][1];

    int id_1 = links.links_[i][0];
    int id_2 = links.links_[i][1];

    // forget this link - from distribution, from entity distr.
    // int link_weight = 
    //     config_.entity_weight_[type] * config_.link_weight_;
    int link_weight = config_.link_weight_;
    links.link_topic_pair_counts_[cur_topic_1][cur_topic_2]--;
    counts_topic_words_[type_1][cur_topic_1][id_1] -= link_weight;
    counts_topic_words_[type_2][cur_topic_2][id_2] -= link_weight;
    sum_counts_topic_words_[type_1][cur_topic_1] -= link_weight;
    sum_counts_topic_words_[type_2][cur_topic_2] -= link_weight;

    // compute CDF.
    double alpha_norm_1 = config_.vocab_size_[type_1] * config_.alpha_[type_1];
    double alpha_norm_2 = config_.vocab_size_[type_1] * config_.alpha_[type_1];
    double prev = 0.0;
    for (int topic_1 = 0; topic_1 < config_.n_topics_; ++topic_1) {
      for (int topic_2 = 0; topic_2 < config_.n_topics_; ++topic_2) {
        int delta_z = (topic_1 == topic_2);
        double alpha = config_.link_alpha_;
        if (topic_1 != topic_2) {
          alpha = config_.link_alpha_ / config_.off_diagonal_discount_;
        }  
        if (config_.diagonal_blocks_ && topic_1 != topic_2) {
          cdf[topic_1][topic_2] = prev;
        }
        else {
          cdf[topic_1][topic_2] = prev +
              (links.link_topic_pair_counts_[topic_1][topic_2] + alpha) *
              (counts_topic_words_[type_1][topic_1][id_1] + config_.alpha_[type_1]) /
              (sum_counts_topic_words_[type_1][topic_1] + alpha_norm_1) * 
              (counts_topic_words_[type_2][topic_2][id_2] + config_.alpha_[type_2]) /
              (sum_counts_topic_words_[type_2][topic_2] + alpha_norm_2);
        } // end if
        prev = cdf[topic_1][topic_2];
      }
    }

    // generate sample.
    double unif_sample = rand() * 1.0 / RAND_MAX;
    int new_topic_1 = -1;
    int new_topic_2 = -1;
    int n_t = config_.n_topics_;
    bool broken = false;
    for (int topic_1 = 0; topic_1 < config_.n_topics_; ++topic_1) {
      for (int topic_2 = 0; topic_2 < config_.n_topics_; ++topic_2) {
        if (unif_sample < (cdf[topic_1][topic_2] / cdf[n_t - 1][n_t - 1])) {
          new_topic_1 = topic_1;
          new_topic_2 = topic_2;
          broken = true;
          break;
        }
      }
      if (broken) // cheap trick to break out of nested loop.
        break;
    }
    links.link_topic_assignments_[i][0] = new_topic_1;
    links.link_topic_assignments_[i][1] = new_topic_2;
    
    // add this link back - to distribution, entity distr with new shiny topics
    links.link_topic_pair_counts_[new_topic_1][new_topic_2]++;
    counts_topic_words_[type_1][new_topic_1][id_1] += link_weight;
    counts_topic_words_[type_2][new_topic_2][id_2] += link_weight;
    sum_counts_topic_words_[type_1][new_topic_1] += link_weight;
    sum_counts_topic_words_[type_2][new_topic_2] += link_weight;
  }

  // Cleanup.
  for (int topic = 0; topic < config_.n_topics_; ++topic) {
    delete[] cdf[topic];
  }
  delete[] cdf;
}

double Model::ComputeLinkPerplexity(Links &links) {
  double perplexity = 0.0;
  int type_1 = config_.link_attr_[0]; 
  int type_2 = config_.link_attr_[1]; 
  double alpha_norm_1 = config_.vocab_size_[type_1] * config_.alpha_[type_1]; 
  double alpha_norm_2 = config_.vocab_size_[type_2] * config_.alpha_[type_2]; 

  for (int i = 0; i < links.n_links_; ++i) {
    double link_perplexity = 0.0;
    int e_1 = links.links_[i][0];
    int e_2 = links.links_[i][1];
    for (int t_1 = 0; t_1 < config_.n_topics_; ++t_1) {
      for (int t_2 = 0; t_2 < config_.n_topics_; ++t_2) {
        int delta_z = (t_1 == t_2);
        link_perplexity +=
            (links.link_topic_pair_counts_[t_1][t_2] + config_.link_alpha_) *
            (counts_topic_words_[type_1][t_1][e_1] + config_.alpha_[type_1]) / 
            (sum_counts_topic_words_[type_1][t_1] + alpha_norm_1) * 
            (counts_topic_words_[type_2][t_2][e_2] + config_.alpha_[type_2]) / 
            (sum_counts_topic_words_[type_2][t_2] + alpha_norm_2 + delta_z);
      } // end topic 1
    } // end topic 2
    // incorporating normalizing constant for pair probability in one swoop.
    perplexity += log(link_perplexity / (links.n_links_ +
         config_.link_alpha_ * config_.n_topics_ * config_.n_topics_));
  } // end links

  perplexity = pow(2.0, -1.0 * perplexity / (log(2.0) * links.n_links_));
  return perplexity;
}

void Model::AddDocument(Corpus &c, int doc){
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int word = 0; word < c.doc_num_words_[type][doc]; ++word) {
      int cur_topic = c.word_topic_assignments_[type][doc][word];
      int cur_wordid = c.corpus_words_[type][doc][word];
      counts_topic_words_[type][cur_topic][cur_wordid]++;
      sum_counts_topic_words_[type][cur_topic]++;
    } // end word
  } // end type

  // Make changes to time related structures.
  if (config_.model_real_) {
//    if (c.timestamps_[doc] < 0)
//      return;
    
    for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
      for (int topic = 0; topic < config_.n_topics_; ++topic) {
        if (c.real_flags_[doc][attr]) {
          real_stats_[attr][topic][0] +=
              c.counts_docs_topics_[doc][topic] *
                  c.real_values_[doc][attr];
          real_stats_[attr][topic][1] +=
              c.counts_docs_topics_[doc][topic] *
                  c.real_values_[doc][attr] * c.real_values_[doc][attr];
          topic_allocation_counts_[attr][topic] +=
              c.counts_docs_topics_[doc][topic];
        }
      } // end topic
    } // end attributes
  } // end time modeling if.
}

void Model::RemoveDocument(Corpus &c, int doc){
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int word = 0; word < c.doc_num_words_[type][doc]; ++word) {
      int cur_topic = c.word_topic_assignments_[type][doc][word];
      int cur_wordid = c.corpus_words_[type][doc][word];
      counts_topic_words_[type][cur_topic][cur_wordid]--;
      sum_counts_topic_words_[type][cur_topic]--;
    } // end word
  } // end type

  // Make changes to time related structures.
  if (config_.model_real_) {
//    if (c.timestamps_[doc] < 0)
//      return;

    for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
      for (int topic = 0; topic < config_.n_topics_; ++topic) {
        if (c.real_flags_[doc][attr]) {
          real_stats_[attr][topic][0] -=
              c.counts_docs_topics_[doc][topic] *
                  c.real_values_[doc][attr];
          real_stats_[attr][topic][1] -=
              c.counts_docs_topics_[doc][topic] *
                  c.real_values_[doc][attr] * c.real_values_[doc][attr];
          topic_allocation_counts_[attr][topic] -=
              c.counts_docs_topics_[doc][topic];
        } // did it have real attributes.
      } // end topic
    } // end attribute
  }
}

void Model::ComputePerplexity(Corpus &c, double *perplexity) {
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    perplexity[type] = 0.0;
    for (int doc = 0; doc < c.n_docs_; ++doc) {

      // computing denominator for doc-topic distribution.
      double doc_topic_normalizer = config_.beta_ * config_.n_topics_; // prior
      for (int topic = 0; topic < config_.n_topics_; ++topic) {
        doc_topic_normalizer += c.counts_docs_topics_[doc][topic];
      }

      double diagnostic = config_.beta_ * config_.n_topics_; // prior
      for (int t2 = 0; t2 < config_.n_entity_types_; ++t2) {
        diagnostic += c.doc_num_words_[t2][doc] * config_.entity_weight_[t2];
      }
      if (diagnostic != doc_topic_normalizer) {
        cerr << "What's happening here matey?" << diagnostic << " " << doc_topic_normalizer << endl;
      }

      for (int word_idx = 0;
           word_idx < c.doc_num_words_[type][doc];
           ++word_idx) {
        double word_perplexity = 0.0;
        int cur_wordid = c.corpus_words_[type][doc][word_idx];
        for (int topic = 0; topic < config_.n_topics_; ++topic) {
          word_perplexity = word_perplexity +
              (c.counts_docs_topics_[doc][topic] + config_.beta_) *
              (counts_topic_words_[type][topic][cur_wordid] + config_.alpha_[type]) /
              (sum_counts_topic_words_[type][topic] + config_.vocab_size_[type] * config_.alpha_[type]);
        } // end topics
        perplexity[type] = perplexity[type] +
            log(word_perplexity / doc_topic_normalizer);
      } // end word
    } // end doc
    perplexity[type] = perplexity[type] /
        accumulate(c.doc_num_words_[type], c.doc_num_words_[type] + c.n_docs_, 0);
    perplexity[type] = pow(2.0, -1.0 * perplexity[type] / log(2.0));
  } // end type
}

void Model::Normalize() {
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int i = 0; i < config_.n_topics_; ++i) {
      for (int j = 0; j < config_.vocab_size_[type]; ++j) {
        counts_topic_words_[type][i][j] = (counts_topic_words_[type][i][j] + config_.alpha_[type])/
            (sum_counts_topic_words_[type][i] + config_.vocab_size_[type] * config_.alpha_[type]);
      } // end word
      sum_counts_topic_words_[type][i] = 1.0;
    } // end topic
  } // end type
}

void Model::Save(const string &model_file_name_prefix) {

  for (int type = 0; type < config_.n_entity_types_; ++type) {
    ostringstream oss;
    ostringstream raw_oss;
    ostringstream oss_most_likely_topic;
    oss     << model_file_name_prefix << "." << type;
    raw_oss << model_file_name_prefix << "." << type << ".raw";
    oss_most_likely_topic << model_file_name_prefix << "." << type << ".mltopic";
    ofstream ofs(oss.str().c_str());
    ofstream raw_ofs(raw_oss.str().c_str());
    ofstream mlt_ofs(oss_most_likely_topic.str().c_str());
    double num_words_of_type =
        accumulate(sum_counts_topic_words_[type],
                   sum_counts_topic_words_[type] + config_.n_topics_, 0.0);
    vector<pair<int, double> > ml_topic(config_.vocab_size_[type]);
    for (int i = 0; i < config_.n_topics_; ++i) {
      for (int j = 0; j < config_.vocab_size_[type]; ++j) {
        double prob = (counts_topic_words_[type][i][j] + config_.alpha_[type])/
            (sum_counts_topic_words_[type][i] + config_.vocab_size_[type] * config_.alpha_[type]);
        ofs << prob << ' ';
        raw_ofs << counts_topic_words_[type][i][j] << ' ';

        if (prob >= ml_topic[j].second) {
          ml_topic[j].first = i;
          ml_topic[j].second = prob;
        } // record topic with highest prob for word j
      }
      // Output fraction of entities of type that were assinged to this topic.
      double topic_weight = sum_counts_topic_words_[type][i] / (num_words_of_type * 1.0);
      ofs << topic_weight << '\n';
      raw_ofs << "\t " << sum_counts_topic_words_[type][i]  << ' ' << num_words_of_type << ' ' << topic_weight << '\n';
    }
    ofs.close();
    raw_ofs.close();

    for (int j = 0; j < config_.vocab_size_[type]; ++j) {
      mlt_ofs << ml_topic[j].first << endl;
    } // end saving best topics for words
    mlt_ofs.close();

  } // end types


  if (config_.model_real_) {
    ostringstream oss;
    ostringstream raw_oss;
    oss << model_file_name_prefix << "." << "time";
    raw_oss << model_file_name_prefix << "." << "time" << ".raw";
    ofstream ofs(oss.str().c_str());
    ofstream raw_ofs(raw_oss.str().c_str());
    for (int i = 0; i < config_.n_topics_; ++i) {
      for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
        ofs << gaussian_parameters_[attr][i][0] << ' ' << gaussian_parameters_[attr][i][1] 
            << ' '
            << beta_parameters_[attr][i][0] << ' ' << beta_parameters_[attr][i][1] << '\t';
        raw_ofs << real_stats_[attr][i][0] << ' ' << real_stats_[attr][i][1] << ' '
                << topic_allocation_counts_[attr][i] << '\t';
      }
      ofs <<'\n';
      raw_ofs << '\n';
    }
    ofs.close();
    raw_ofs.close();
  }
}
// TODO: write code for reading in statistics

void Model::DebugDisplay(ostream &os = std::cout) {
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    os << "=== Type " << type << " ===\n";
    DisplayMatrix(counts_topic_words_[type],
                  config_.n_topics_,
                  config_.vocab_size_[type],
                  "C_vt",
                  true, os);
    DisplayColumn(sum_counts_topic_words_[type],
                  config_.n_topics_,
                  "row sums c_vt",
                  os);
  }
}

void Model::Free() {
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int i = 0; i < config_.n_topics_; ++i) {
      delete[] counts_topic_words_[type][i];
    }
    delete[] counts_topic_words_[type];
    delete[] sum_counts_topic_words_[type];
  }
  delete[] counts_topic_words_;
  delete[] sum_counts_topic_words_;

  if (config_.model_real_) {
    for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
      for (int i = 0; i < config_.n_topics_; ++i) {
        delete[] beta_parameters_[attr][i];
        delete[] gaussian_parameters_[attr][i];
        delete[] real_stats_[attr][i];
      } // end topics
      delete[] topic_allocation_counts_[attr];
      delete[] real_stats_[attr];
      delete[] beta_parameters_[attr];
      delete[] gaussian_parameters_[attr];
    } // end attr
    delete[] topic_allocation_counts_;
    delete[] real_stats_;
    delete[] beta_parameters_;
    delete[] gaussian_parameters_;
  } // end destruction of real structs
}

int main(int argc, char **argv) {
  Options opt(argc, argv);
  string train_file       = opt.GetStringValue("train_file");
  string test_file        = opt.GetStringValue("test_file");
  string config_file      = opt.GetStringValue("config_file");
  string output_prefix    = opt.GetStringValue("output_prefix");
  int debug               = opt.GetIntValue("debug", 0);
  int n_iterations        = opt.GetIntValue("iter", 4);
  int n_topics            = opt.GetIntValue("topics", 10);
  int rand_initializer    = opt.GetIntValue("randinit", 0);
  int diagonly            = opt.GetIntValue("diagonly", 0);
  int diagdiscount        = opt.GetIntValue("diagdiscount", 1);
  int realweight          = opt.GetIntValue("realweight", 1);
  RealDistr model_real    = static_cast<RealDistr>(opt.GetIntValue("model_real", 0));

  string avg_model_file   = output_prefix + ".avg.model";
  string model_file       = output_prefix + ".model";       
                                        // multinomials
  string topic_distr_file = output_prefix + ".topic_distr";
                                        // topic distribution for docs
  string link_topics_file = output_prefix + ".link_topic_samples"; 
                                        // topics chosen for links
  string word_topics_file = output_prefix + ".topic_samples";
                                        // topics chosen for words and entities

  // To use links - set Flag and corpus using cmd line.
  // Set the size of link corpus and which attr type is being linked in config.
  int model_links         = opt.GetIntValue("model_links", 0);
  string link_train_file  = opt.GetStringValue("link_train_file");
  string link_test_file   = opt.GetStringValue("link_test_file");
  
  //Setting global values.
      debug_time          = opt.GetIntValue("debug_time", 0);

  if (model_real == GAUSSIAN) {
    cout << "Using gaussian for reals with weight " << realweight << "\n";
  }
  else if (model_real == BETA)
    cout << "Using beta for reals\n";

  if (rand_initializer != 0) {
    if (rand_initializer == -1)  {
      srand(time(NULL));
      cout << "rand init\n";
    }
    else {
      cout << "initting to specified value " << rand_initializer << "\n";
      srand(rand_initializer); // preset initializer
    }
  }

  if (debug) {
    cout << "debuggin" << endl;
  }
  // Read in config.
  Config config(n_topics, model_real, model_links);
  ifstream ifs(config_file.c_str());
  config.ReadConfig(ifs);
  ifs.close();
  cout << "Read in config\n";
  if (debug)
    config.DebugDisplay();
  if (model_real && !config.n_real_valued_attrs_) {
    config.model_real_ = NONE;
    cout << "Cannot model time when real information not present in data"
         << endl;
    return 1;
  }

  // Create model object.
  config.diagonal_blocks_ = diagonly;
  config.off_diagonal_discount_ = diagdiscount;
  config.real_weight_ = realweight;
  config.link_alpha_ = config.n_train_links_ * 1.0 / (config.n_topics_ * 2);
  Model link_lda(config);
  link_lda.Allocate();

  // Create and initialize corpus object.
  Corpus corpus(&config, config.GetNumTrainingDocs());
  corpus.Allocate();

  ifstream ifs2(train_file.c_str());
  corpus.Read(ifs2, &link_lda);
  ifs2.close();

  // Create test set.
  Corpus test_corpus(&config, config.GetNumTestDocs());

  // Unseen documents corpus
  if (config.GetNumTestDocs() > 0) {
    test_corpus.Allocate();

    ifstream ifs3(test_file.c_str());
    test_corpus.Read(ifs3);
    ifs3.close();
    if (debug)
      test_corpus.DebugDisplay();
  }

  // Do optional link setup.
  Links train_link_corpus(&config, config.GetNumTrainingLinks());
  Links test_link_corpus(&config, config.GetNumTestLinks());
  if (model_links) {
    train_link_corpus.Allocate();
    ifstream ifs(link_train_file.c_str());
    train_link_corpus.Read(ifs, &link_lda);
    ifs.close();

    // Are we testing on unseen links?
    if (config.GetNumTestLinks() > 0) {
      test_link_corpus.Allocate();
      ifstream ifs2(link_test_file.c_str());
      test_link_corpus.Read(ifs2);
      ifs2.close();
    }
  } // end if model_links

  // Run MCMC. (4th arg indicates we are estimating).
  // TODO: Do this only if there is a valid test corpus
  link_lda.SetTestCorpus(&test_corpus);
  link_lda.SetLinkTestCorpus(&test_link_corpus);
  Model *average_model = link_lda.MCMC(corpus, &train_link_corpus, n_iterations, 10, false, debug);
  average_model->Save(avg_model_file);
  average_model->Free();
  delete average_model;

  link_lda.Save(model_file.c_str());
  if (topic_distr_file != "") {
    if (config.GetNumTestDocs() > 0 ) {
      ofstream os(topic_distr_file.c_str());
      test_corpus.SaveTopicDistributions(os);
    os.close();
    }

    ofstream os_train((topic_distr_file + ".train").c_str());
    corpus.SaveTopicDistributions(os_train);
    os_train.close();
  }

  if (model_links && link_topics_file != "") {
    ofstream os((link_topics_file + ".links.train").c_str());
    train_link_corpus.SaveTopics(os); 
    os.close();

    if (config.GetNumTestLinks() > 0 ) {
      ofstream os((link_topics_file + ".links").c_str());
      test_link_corpus.SaveTopics(os); 
      os.close();
    }
  }

/*  if (model_links && word_topics_file != "") {
    ofstream os((link_topics_file + ".links.train").c_str());
    train_link_corpus.SaveTopics(os); 
    os.close();

    if (config.GetNumTestLinks() > 0 ) {
      ofstream os((link_topics_file + ".links").c_str());
      test_link_corpus.SaveTopics(os); 
      os.close();
    }
  } */

  string log_file = model_file + ".log";
  ofstream ofs(log_file.c_str());
  link_lda.DebugDisplay(ofs);
  ofs.close();

  // Shut down shop.
  link_lda.Free();
  corpus.Free();
  if (config.GetNumTestDocs() > 0)
    test_corpus.Free();

  if (model_links) {
    train_link_corpus.Free();
    test_link_corpus.Free();
  }
  return 0;
}

// Summary of my data structures
// Matrices - Ctd Cvt sCvt
// Inference - wta
// Corpus - corpus_words doc_num_words (CONSTANT)
