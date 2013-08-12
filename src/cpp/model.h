#ifndef __MODEL_H__
#define __MODEL_H__

#include <fstream>
#include <string>
#include <vector>
#include <ext/hash_map>

#include <dlib/svm.h>

class Config;
class Links;
class Corpus;
class Stats;

struct Log2 {
  __gnu_cxx::hash_map<int, double> cache_;
  double log2;
  long int cache_hits_;
  long int total_hits_;
  __gnu_cxx::hash_map<int,double>::iterator it_;
  Log2() {
    log2 = log(2.0);
    cache_hits_ = 0;
    total_hits_ = 0;
  }
  double operator()(int i) {
    if ((it_ =cache_.find(i)) == cache_.end()) {
      double val = log(i) / log2;
      cache_.insert(make_pair(i, val));
      total_hits_++;
      return val;
    } else {
      cache_hits_++;
      total_hits_++;
      return it_->second;
    }
  }
  double CacheHitRate() {
    return cache_hits_ * 1.0 / total_hits_;
    cache_hits_ = 0;
    total_hits_ = 0;
  }
};

struct MHStats {
  int gt1_;
  int accept_;
  int reject_;
  MHStats(int g, int a, int r) : gt1_(g), accept_(a), reject_(r) {}
};

class Model {
 public:
  const Config &config_;
  Model(const Model &); // copy constructor.
  Model(const Config &c):config_(c) {norm_ = NULL; freed_ = 0; normalized_ = 0; input_labels_ = NULL; true_labels_ = NULL;}

  // store how many instances of each word/entity/etc in a topic
  double ***counts_topic_words_;     // [nTypesOfEntities][nTopics][vocab_size]
  double **sum_counts_topic_words_;  // [nTypesOfEntities][nTopics]

  // only used when using penalties to enforce mixed-ness constraints
  double **word_entropies_;          // [nTypesOfEntities][vocab_size]
  double **frequencies_;             // [nTypesOfEntities][vocab_size]
  double *topic_sizes_;              // [nTopics] -- this is a type-marginalized version of topic_weights_
                                     // Can also be treated as walking over the ds below and histogramming over topics.
  int **topic_;                      // [nTypesOfEntities][vocab_size] -- which topic does this word belong to

  // volume constraint datastructures
  double *volume_entropy_components_; // volume entropy components
  double total_volume_;               // total volume of graph + docs (which is same as weight of all docs)

  // data structures for real stats
  double ***beta_parameters_;        // nattrs X ntopics x 2
  double ***gaussian_parameters_;    // nattrs X ntopics x 2
  double ***real_stats_;             // nattrs X ntopics x 2
  double **topic_allocation_counts_; // ntopics x n_real_attrs 

  // data structures for slda
  typedef dlib::matrix<double, 100, 1> sample_type;
  typedef dlib::linear_kernel<sample_type> kernel_type;
  dlib::decision_function<kernel_type> **regressors_; 

  // for display only
  double **topic_weights_;           // [nTypes][ntopics]
  int normalized_;                   // has model been normalized? Values >1 indicate how many models it's an average of.

  double *perplexities_;             // recording final results
  double link_perplexity_;      
  Log2 log2_;

  void TrainRegression(Corpus &corpus);
  void PredictTargets(Corpus &corpus, int doc);
  void PredictTargetsFromTheta(Corpus &corpus);
  long double GetTargetProbability(Corpus &corpus, int doc, int topic, int weight);
  pair<double, pair<double, double> > MetropolisTest(Corpus &c, int doc, int type, int word_idx, int cur_topic, int new_topic);

  Model* MCMC(Corpus &corpus,      Links *links, 
              Corpus *test_corpus, Links *test_link_corpus,
              bool debug = false);

  void InferLinkDistribution(Links &links, bool debug);
  void FastLDA(Links &links, bool debug);
  void AddLink(Links &links, int i, bool remove = false);
  void AddLinkToPenalty(Links &links, int i, bool remove = false);

  void RemoveWord(Corpus &corpus, int doc, int word_idx, int type);
  void AddWord(Corpus &corpus, int doc, int word_idx, int type, bool remove = false);
  void AddWordToRealAttrs(Corpus &corpus, int type, int doc, int cur_topic, bool remove = false);
  void AddPenaltyWord(int type, int cur_topic, int cur_wordid, double weight, double vol_weight, bool remove = false);

  long double GetRoleEntropyProbability(Links &links, int i, int topic_1, int topic_2);
  long double GetRoleEntropyProbability(int type, int topic, int id, double weight, bool unnorm);
  long double GetBalanceEntropyProbability(Links &links, int i, int topic_1, int topic_2);
  long double GetBalanceEntropyProbability(int type, int topic, int cur_wordid);
  long double GetVolumeEntropyProbability(int type, int topic);
  long double GetVolumeEntropyProbabilityForLinks(int topic_1, int topic_2);
  void GetRealAttributeProbabilities(Corpus &corpus, int doc, long double *time_prob);
  long double GetWordTopicProbability(int topic, int type, int cur_wordid);
  long double GetAverageNodeRoleEntropy(int type);
  long double GetVolumeEntropy();
  void InitializeVolumeConstraint();

  void   SampleTopics(Corpus &corpus, bool useInputTopics = false);
  void   SampleTopicsForLinks(Links &links, bool useInputTopics = false);
  void   SampleTopicPair(double **cdf, int &new_topic_1, int &new_topic_2);
  double ComputeLinkPerplexity(Links &links, bool smooth = true);
  void   ComputePerplexity(Corpus &corpus, double *, bool smooth = true);

  void DebugRealAttrs(Corpus &corpus, int doc, long double *cdf, long double *time_prob);
  void DebugMCMC(Corpus &corpus, long double *cdf, int cur_topic, int new_topic, int iteration, int doc, int word_idx);

  void Add(const Model &);
  void Normalize();
  void CheckIntegrity(Corpus *, Links *);

  void EstimateBeta();
  void InitializePenaltyTerms(bool debug = false);
  int GetWinningTopic(int type, int wordid);
  long double GetClusterBalanceEntropy(bool allTypes);

  void AddDocument(Corpus &c, int doc, bool remove = false);
  void RemoveDocument(Corpus &c, int doc);
  void AddCorpus(Corpus &c, bool remove = false);

  void AddLinks(Links &links, bool remove = false);
  void RemoveLinks(Links &links);

  void Allocate();
  void Free();
  void Save(const string&);
  void DebugDisplay(std::ostream &os);

  void LoadBeta(const string &model_file_name_prefix);
  void LoadLabels(const string &label_file);
  int **ReadLabelFile(const string &label_file);
  void SampleFromInputTopics(const string &model_file_prefix, Corpus &corpus, Links *links);
  void SampleFromFakeInputTopics(Corpus &corpus, Links *links);
  void UnloadBeta();
  std::vector<double> GetAccuracyFromHungarian();
  void CalculateAccuracy(Stats *stats = NULL);
  std::vector<double> GetNMI();
  double GetEntropy(int *, int);
  int GetNumTrueClasses();
  std::vector<std::vector<double> > GetKNN();
  double JSD(int type, int p, int q);

  int **input_labels_;

 private:
  int freed_;
  double ***input_topics_;

  ofstream metro_log_stream_;
  std::vector<double> labels_;

  int **true_labels_;
  double **norm_;
  struct TopicPair {
    int t1_;
    int t2_;
    double count_;
    TopicPair(int t1, int t2, double c): t1_(t1), t2_(t2), count_(c) {}
    bool operator<(const TopicPair &a) const {
      return count_ > a.count_;
    }
  };
  multiset<TopicPair> order_;
  void InitNormsForFastLDA(Links &);
  void CheckFastLDAIntegrity(Links &);
  double norm_1_;
  void DebugFast(map<double, int> &norm_4);
};
#endif
