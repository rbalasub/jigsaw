#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>
#include <dlib/svm.h>

#include "config.h"
#include "corpus.h"
#include "hungarian.h"
#include "links.h"
#include "model.h"
#include "stats.h"
#include "util.h"

using namespace std;

int debug_time = 0;

// Pre-calculate cluster membership distribution entropies for each word for efficiency.
// During MCMC, we'll only need to modify it instead of recalculating.

void Model::InitializeVolumeConstraint() {
  total_volume_ = 0;
  for (int t = 0; t < config_.n_topics_; ++t) {
    for (int type = 0; type < config_.n_entity_types_; ++type) {
      total_volume_ += sum_counts_topic_words_[type][t] * config_.entity_weight_[type];
    } // end type
  } // end topic

  for (int t = 0; t < config_.n_topics_; ++t) {
    double topic_volume = 0.0;
    for (int type = 0; type < config_.n_entity_types_; ++type) {
      topic_volume += sum_counts_topic_words_[type][t] * config_.entity_weight_[type]; 
    } // end type
    if (topic_volume > 0)
      volume_entropy_components_[t] = topic_volume / total_volume_ * log2(topic_volume / total_volume_);
    else
      volume_entropy_components_[t] = 0.0;
  } // end topic
}

void Model::InitializePenaltyTerms(bool debug) {
  //if (config_.volume_constraint_)
    InitializeVolumeConstraint();

  if (config_.mixedness_constraint_) {
    for (int type = 0; type < config_.n_entity_types_; ++type) {
      for (int j = 0; j < config_.vocab_size_[type]; ++j) {
        double old_val = 0.0;
        if (debug)
          old_val = word_entropies_[type][j];
        word_entropies_[type][j] = 0.0;
        frequencies_[type][j] = 0.0;

        for (int t = 0; t < config_.n_topics_; ++t) {
          frequencies_[type][j] += counts_topic_words_[type][t][j];
          if (counts_topic_words_[type][t][j] == 0)
            continue;
          double p_t = counts_topic_words_[type][t][j];
          word_entropies_[type][j] -= p_t * log(p_t);
        } // end topics
        if (frequencies_[type][j] == 0.0) {
          continue;
        }
        word_entropies_[type][j] /= frequencies_[type][j];
        word_entropies_[type][j] += log(frequencies_[type][j]);
        word_entropies_[type][j] /= log(2.0);
        if (debug && abs(old_val - word_entropies_[type][j]) > 0.00001) {
          cout << "Entropy mismatch in type " << type << " word " << j << " should be " << word_entropies_[type][j] << " is " << old_val << endl;
        }
        //cout << word_entropies_[type][j] << " for " << j << " type " << type << endl;
        if (debug) {
          word_entropies_[type][j] = old_val;
        }
      } // end words
    } // end types
  } // end if

  // Now initializing data structures for cluster size distribution entropy.
  if (config_.balance_constraint_)
    GetClusterBalanceEntropy(true);
}

long double Model::GetClusterBalanceEntropy(bool allTypes = false) {
  long double log2 = log(2.0);

  // allTypes determines whether we are initializing for the very first time.
  if (allTypes) 
    for (int t = 0; t < config_.n_topics_; ++t)
      topic_sizes_[t] = 0.0;
  
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    if (!(allTypes || type == config_.link_attr_[0] || type == config_.link_attr_[1]))
      continue;
    for (int j = 0; j < config_.vocab_size_[type]; ++j) {
      if (!allTypes) // we are modifying an already initialized distribution.
        topic_sizes_[topic_[type][j]] -= config_.entity_weight_[type];
      int winning_topic = GetWinningTopic(type, j);
      topic_sizes_[winning_topic] += config_.entity_weight_[type];
      topic_[type][j] = winning_topic;
    } // end words
  } // end types

  long double entropy = 0.0;
  double tot = accumulate(topic_sizes_, topic_sizes_ + config_.n_topics_, 0.0);
  for (int t = 0; t < config_.n_topics_; ++t) {
    if (topic_sizes_[t]) {
      long double p = topic_sizes_[t] / tot;
      entropy -= (p * log(p) / log2);
    }
  }
  return entropy;
}

int Model::GetWinningTopic (int type, int wordid) {
  int winning_topic = 0;
  double best_topic_prob = 0.0;

  for (int t = 0; t < config_.n_topics_; ++t) {
    //if (t == 0 || (counts_topic_words_[type][t][wordid] + config_.beta_[type]) / (sum_counts_topic_words_[type][t] + config_.beta_[type] * config_.vocab_size_[type]) > best_topic_prob) {
    if (t == 0 || counts_topic_words_[type][t][wordid] > best_topic_prob) {
      //best_topic_prob = (counts_topic_words_[type][t][wordid] + config_.beta_[type]) / (sum_counts_topic_words_[type][t] + config_.beta_[type] * config_.vocab_size_[type]);
      best_topic_prob = counts_topic_words_[type][t][wordid];
      winning_topic = t;
    }
  } // end topics

  return winning_topic;
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
}

Model::Model(const Model &base) : config_(base.config_) {
  freed_ = 0;
  normalized_ = 0;
  norm_ = NULL;
  Allocate();
  Copy(base.counts_topic_words_, counts_topic_words_, config_.n_entity_types_, config_.n_topics_, config_.vocab_size_);
  Copy(base.sum_counts_topic_words_, sum_counts_topic_words_, config_.n_entity_types_, config_.n_topics_);

  if (config_.model_real_) {
    Copy(base.beta_parameters_, beta_parameters_, config_.n_real_valued_attrs_, config_.n_topics_, 2);
    Copy(base.gaussian_parameters_, gaussian_parameters_, config_.n_real_valued_attrs_, config_.n_topics_, 2);
    Copy(base.real_stats_, real_stats_, config_.n_real_valued_attrs_, config_.n_topics_, 2);
    Copy(base.topic_allocation_counts_, topic_allocation_counts_, config_.n_real_valued_attrs_, config_.n_topics_);
  }

  if (config_.model_targets_) {
    for (int target = 0; target < config_.n_real_targets_; ++target) {
      if (base.regressors_[target])
        regressors_[target] = new dlib::decision_function<kernel_type>(*base.regressors_[target]);
      else 
        cout << "Original regressor is null in copy constructor" << endl;
    } // end targets
  } // end if
  input_labels_ = NULL;
  true_labels_  = NULL;
}

void Model::Add(const Model &base) {
  if (!base.normalized_) {
    cerr << "Cannot add an unnormalized model to another model\n";
  }
  if (config_.model_real_) {
    for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
      for (int i = 0; i < config_.n_topics_; ++i) {
        for (int param = 0; param < 2; ++param) {
          gaussian_parameters_[attr][i][param] =
              (gaussian_parameters_[attr][i][param] * normalized_ + base.gaussian_parameters_[attr][i][param]) / (normalized_ + 1);
          beta_parameters_[attr][i][param] =
              (beta_parameters_[attr][i][param] * normalized_ + base.beta_parameters_[attr][i][param]) / (normalized_ + 1);
        } // end param
      } // end topic
    } // end attr
  }

  if (config_.model_targets_) {
    for (int target = 0; target < config_.n_real_targets_; ++target) {
      if (base.regressors_[target] != NULL) {
        if (!regressors_[target])
          regressors_[target] = new dlib::decision_function<kernel_type>(*base.regressors_[target]);
        else {
          for (int i = 0; i < config_.n_topics_; ++i) {
            regressors_[target]->basis_vectors(0)(i) = 
              (regressors_[target]->basis_vectors(0)(i) * normalized_ + base.regressors_[target]->basis_vectors(0)(i)) / (normalized_ + 1);
          } // end topic
          regressors_[target]->b = (regressors_[target]->b * normalized_ + base.regressors_[target]->b) / (normalized_ + 1);
        } // end if (adding to regressor)
      } else {
        cout << "Base model regressor is null while taking average of MCMC samples " << endl;
      } // end if no valid regressor
    } // end targets
  } // end slda targets

  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int i = 0; i < config_.n_topics_; ++i) {
      sum_counts_topic_words_[type][i] =
          (sum_counts_topic_words_[type][i] * normalized_ + base.sum_counts_topic_words_[type][i]) / (normalized_ + 1);
      for (int j = 0; j < config_.vocab_size_[type]; ++j) {
        counts_topic_words_[type][i][j] =
            (counts_topic_words_[type][i][j] * normalized_ + base.counts_topic_words_[type][i][j]) / (normalized_ + 1);
      }
      if (base.normalized_){
        topic_weights_[type][i] =
            (topic_weights_[type][i] * normalized_ + base.topic_weights_[type][i]) / (normalized_ + 1);
      }
    } // end topics
  } // end types
  normalized_++;
}

void Model::Allocate() {
  counts_topic_words_ = new double**[config_.n_entity_types_];
  perplexities_       = new double[config_.n_entity_types_];

  if (config_.model_targets_) {
    regressors_ = new dlib::decision_function<kernel_type>*[config_.n_real_targets_];
    dlib::decision_function<kernel_type> *ptr = NULL;
    fill(regressors_, regressors_ + config_.n_real_targets_, ptr);
  }

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

  word_entropies_            = new double*[config_.n_entity_types_];
  frequencies_               = new double*[config_.n_entity_types_];
  topic_                     = new int*[config_.n_entity_types_];
  topic_weights_             = new double*[config_.n_entity_types_];
  sum_counts_topic_words_    = new double*[config_.n_entity_types_];
  topic_sizes_               = new double[config_.n_topics_];
  volume_entropy_components_ = new double[config_.n_topics_];

  for (int type = 0; type < config_.n_entity_types_; ++type) {
    counts_topic_words_[type] = new double*[config_.n_topics_];
    sum_counts_topic_words_[type] = new double[config_.n_topics_];
    topic_weights_[type] = new double[config_.n_topics_];
    for (int i = 0; i < config_.n_topics_; ++i) {
      counts_topic_words_[type][i] = new double[config_.vocab_size_[type]];
      sum_counts_topic_words_[type][i] = 0;
      if (type == 0) {
        topic_sizes_[i] = 0;
      }

      if (i == 0) {
        word_entropies_[type] = new double[config_.vocab_size_[type]];
        frequencies_[type]    = new double[config_.vocab_size_[type]];
        topic_[type]          = new int[config_.vocab_size_[type]];
      }
      for (int j = 0; j < config_.vocab_size_[type]; ++j) {
        counts_topic_words_[type][i][j] = 0;
        if (i == 0) {
          word_entropies_[type][j] = 0;
        }
      } // end vocab
    } // end topics
  } // end types
}

void Model::GetRealAttributeProbabilities(Corpus &corpus, int doc, long double *time_prob) {
  // If this is test-time, estimate beta after every doc because we
  // "forget" this document after sampling topics for all words in the
  // doc  and the effect of this document on time will be null.  While
  // this is expensive, this is the price we pay in the current setup.
  // In contrast during train time, we estimate once per iteration.
  // During test, we estimate once per doc per iteration.
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
        prob = pow(gaussian_parameters_[attr][topic][1], -0.5 * config_.real_weight_ / weight_of_doc) *
               exp(static_cast<long double>(-1 * pow((t - gaussian_parameters_[attr][topic][0]), 2) /
                   (2 * gaussian_parameters_[attr][topic][1]) *
                   (config_.real_weight_ / weight_of_doc)));  // 100 is to remove inf bugs.
                    
      }
      time_prob[topic] *= prob;
    } // end topics
    if (debug_time >= 2) {
      cout << "Doc " << doc << " Time: " << t << endl;
      DisplayColumn(time_prob, config_.n_topics_, "Time prob", cout);
    }
  } // end attr
}

long double Model::GetWordTopicProbability(int topic, int type, int wordid) {
  if (normalized_)  
    return counts_topic_words_[type][topic][wordid];
  if (config_.flipped_model_)
    return counts_topic_words_[type][topic][wordid] + config_.beta_[type];
  return (counts_topic_words_[type][topic][wordid] + config_.beta_[type]) /
         (sum_counts_topic_words_[type][topic] + config_.vocab_size_[type] * config_.beta_[type]);
}

void Model::SampleTopics(Corpus &corpus, bool useInputTopics) {
  long double *time_prob                  = new long double[config_.n_topics_];
  long double *cdf                        = new long double[config_.n_topics_];

  //CheckIntegrity(&corpus, NULL);
  if (!useInputTopics && config_.model_real_)
    EstimateBeta();
  corpus.RandomInit(input_labels_);
  corpus.CheckIntegrity();

  int labeled_words_cnt = 0;
  int general_cnt = 0;
  int ood_cnt = 0;
  int slipped_by_cnt = 0;
  for (int iteration = 0; iteration < config_.n_sample_iterations_ + config_.n_avg_; ++iteration) {
    labeled_words_cnt = 0;
    slipped_by_cnt = 0;
    for (int doc = 0; doc < corpus.n_docs_; ++doc) {
      if (config_.model_real_)
        GetRealAttributeProbabilities(corpus, doc, time_prob);
      for (int type = 0; type < config_.n_entity_types_; ++type) {
        for (int word_idx = 0;
             word_idx < corpus.doc_num_words_[type][doc];
             ++word_idx) {

          int cur_wordid = corpus.corpus_words_[type][doc][word_idx];

          int cur_topic  = corpus.word_topic_assignments_[type][doc][word_idx];
          if (cur_topic < 0 || cur_topic >= config_.n_topics_) {
            cout << "Hmm ... weird topic assignment at doc " << doc << " word " << word_idx << " topic# " << cur_topic << endl;
            exit(0);
          }
          corpus.AddWord(doc, cur_topic, type, true);

          double p = (corpus.counts_docs_topics_[doc][cur_topic] + config_.alpha_) / corpus.weight_[doc];
          corpus.theta_entropy_components_[doc][cur_topic] = p * log2(p);

          int new_topic = -1;
          double random_number = 0.0;
          if (config_.use_node_labels_ &&
              config_.clamp_rigidity_ > 0.01 &&
              input_labels_[type][cur_wordid] != -1  && 
              (random_number = Random()) < config_.clamp_rigidity_) {
            labeled_words_cnt++;
            new_topic = input_labels_[type][cur_wordid];
            if (config_.md_n_domains_)
              new_topic += config_.md_split_start_indexes_[corpus.domains_[doc]];
            if (config_.md_seeds_) {
              double r = rand() * 1.0 / RAND_MAX;
              if (r > config_.md_probs_[0] && r <= (config_.md_probs_[0] + config_.md_probs_[1])) {
                // assign to general instead
                new_topic = input_labels_[type][cur_wordid] + config_.md_split_start_indexes_[config_.md_n_domains_];
                general_cnt++;
              } else if (r > (config_.md_probs_[0] + config_.md_probs_[1])) {
                // assign to out of domain instead
                int rand_domain = UniformSample(config_.md_n_domains_);
                new_topic = input_labels_[type][cur_wordid] + config_.md_split_start_indexes_[rand_domain];
                ood_cnt++;
              }
            } // end if multi domain
          } else {
            if (random_number > 0.0)
              slipped_by_cnt++;
            // compute topic CDF.
            for (int topic = 0; topic < config_.n_topics_; ++topic) {
              long double topic_prob = corpus.GetTopicProbability(doc, topic, type);
              if (useInputTopics) 
                topic_prob *= input_topics_[type][topic][cur_wordid];
              else
                topic_prob *= GetWordTopicProbability(topic, type, cur_wordid);
              if (config_.model_targets_)
                topic_prob *= GetTargetProbability(corpus, doc, topic, config_.entity_weight_[type]);
              if (config_.model_real_)
                topic_prob *= time_prob[topic];
              if (topic == 0)
                cdf[topic] = topic_prob;
              else
                cdf[topic] = cdf[topic - 1] + topic_prob;
            }
            // sample topic using CDF
            new_topic = SampleTopicFromMultinomial(cdf, config_.n_topics_);
          } // end if 

          corpus.word_topic_assignments_[type][doc][word_idx] = new_topic;
          corpus.AddWord(doc, new_topic, type, false);

          p = (corpus.counts_docs_topics_[doc][new_topic] + config_.alpha_) / corpus.weight_[doc];
          corpus.theta_entropy_components_[doc][new_topic] = p * log2(p);
        } // end words
      } // end types of words
    } //  end docs
    if (!useInputTopics && iteration >= config_.n_sample_iterations_)
      corpus.AddToAverager();
  } // end iteration
  //CheckIntegrity(&corpus, NULL);
  corpus.CheckIntegrity();
  if (labeled_words_cnt) {
    cout << "Out of " << labeled_words_cnt << " (" << general_cnt << "," << ood_cnt <<") " << slipped_by_cnt << " words slipped by" << endl;
  }
  cout << "Sampled over " <<  config_.n_sample_iterations_ + config_.n_avg_ << " iterations" << endl;
  delete[] time_prob;
  delete[] cdf;

  if (!useInputTopics) {
    if (config_.model_targets_)
      PredictTargetsFromTheta(corpus);
    double *perplexity = new double[config_.n_entity_types_];
    ComputePerplexity(corpus, perplexity, false); //Don't smooth with averaged model.
    for (int j = 0; j < config_.n_entity_types_; ++j) {
      perplexities_[j] = perplexity[j];
    }
    delete[] perplexity;
  } // end if
}

void Model::AddWord(Corpus &corpus, int doc, int word_idx, int type, bool remove) {
  int cur_topic = corpus.word_topic_assignments_[type][doc][word_idx];
  int cur_wordid = corpus.corpus_words_[type][doc][word_idx];

  int mult = 1;
  if (remove)
    mult = -1;
  // remove from penalty term ds and real attrs
  AddPenaltyWord(type, cur_topic, cur_wordid, config_.lit_weight_, 0.0, remove); //should vol be 0.0 - what does it mean?
  if (config_.model_real_)
    AddWordToRealAttrs(corpus, type, doc, cur_topic, remove);
  if (config_.theta_constraint_) {
    double p = (corpus.counts_docs_topics_[doc][cur_topic] + mult * config_.entity_weight_[type] + config_.alpha_) / corpus.weight_[doc];
    corpus.theta_entropy_components_[doc][cur_topic] = p * log2(p);
  }
  // remove from model core
  corpus.AddWord(doc, cur_topic, type, remove);
  counts_topic_words_[type][cur_topic][cur_wordid] += mult * config_.lit_weight_;
  sum_counts_topic_words_[type][cur_topic] += mult * config_.lit_weight_;
}

void Model::RemoveWord(Corpus &corpus, int doc, int cur_wordid, int type) {
  AddWord(corpus, doc, cur_wordid, type, true);
}

Model *  Model::MCMC(Corpus &corpus,      
                     Links  *links,
                     Corpus *test_corpus,
                     Links  *test_link_corpus,
                     bool debug) {

  cout << corpus.doc_labels_ << " is address of doc_labels" << endl;
  if (config_.fast_lda_)
    cout << "Using fast LDA for networks" << endl;
  if (corpus.n_docs_)
    AddCorpus(corpus);
  if (links && links->n_links_)
    AddLinks(*links);
  if (config_.metro_trace_ && !metro_log_stream_.is_open()) {
    metro_log_stream_.open((config_.output_prefix_ + ".metrotrace").c_str(), ios_base::out);
    cout << "Opening metro log file\n" << endl;
  }
  double      *perplexity                 = new      double[config_.n_entity_types_];
  long double *time_prob                  = new long double[config_.n_topics_];
  long double *cdf                        = new long double[config_.n_topics_];

  InitializePenaltyTerms();
  Model *average_model = new Model(config_);
  average_model->Allocate();
  if (config_.use_node_labels_)
    average_model->LoadLabels(config_.node_label_file_);
    
  Links *ptr = (config_.model_links_ && links && links->n_links_) ? links : NULL;
  CheckIntegrity(&corpus, ptr);

  if (config_.model_targets_ && config_.n_real_targets_) {
    cout << "Will train regressors" << endl;
    labels_.resize(corpus.n_docs_);
    TrainRegression(corpus);
  } else {
    cout << "Won't train regressors, Flag=" << config_.model_targets_ << " training " << config_.n_real_targets_ << " targets" << endl;
  }


  cout << "Start     ...    : ";
  // Report perplexities
  if (corpus.n_docs_) {
    ComputePerplexity(corpus, perplexity);
    for (int j = 0; j < config_.n_entity_types_; ++j) {
      cout << perplexity[j] << " ";
    }
  }

  if (config_.model_links_ && links) {
    double link_perplexity = ComputeLinkPerplexity(*links);
    cout << "Link: " << link_perplexity << " ";
  }
  cout << endl;

  vector<MHStats> mh_stats;

  for (int iteration = 0; iteration < config_.n_iterations_ + config_.n_avg_; ++iteration) {
    cout << "Iteration " << iteration << " ...  "; cout.flush();
    cout << "Avg Node Role Entropy: "; cout.flush();
    for (int type = 0; type < config_.n_entity_types_; ++type) {
      cout << GetAverageNodeRoleEntropy(type) << ' ';
    }
    cout.flush();

    if (config_.model_real_)
      EstimateBeta();


    int labeled_word_cnt = 0;
    int general_cnt = 0;
    int ood_cnt = 0;

    int labeled_docs_cnt = 0;

    int mh_accept_gt1 = 0;
    int mh_accept = 0;
    int mh_reject = 0;
    bool mh_dumped = 0;

    for (int doc = 0; doc < corpus.n_docs_; ++doc) {
      bool doc_already_labeled = false;
      /*
      if (config_.md_n_domains_) {
        set<int **> jazz;
        for (int d = 0; d < corpus.n_docs_; ++d) {
          jazz.insert(corpus.md_entropy_components_[d][0]->counts_);
          jazz.insert(corpus.md_entropy_components_[d][1]->counts_);
          jazz.insert(corpus.md_senti_entropy_components_[d]->counts_);
        } // end for
        for (set<int **>::iterator it = jazz.begin(); it != jazz.end(); ++it) {
          cout << "Iter " << iteration << " Doc " << doc << " ModelJazz " << *it << endl;
        } 
      }*/
      if (config_.model_real_)
        GetRealAttributeProbabilities(corpus, doc, time_prob);

      for (int type = 0; type < config_.n_entity_types_; ++type) {
        for (int word_idx = 0;
             word_idx < corpus.doc_num_words_[type][doc];
             ++word_idx) {

          if (doc == 0 && type == 0 && word_idx == 0)
            mh_dumped = 0;
          RemoveWord(corpus, doc, word_idx, type);

          int cur_wordid = corpus.corpus_words_[type][doc][word_idx];

          int new_topic = -1;

          // doc labels
          if (config_.use_doc_labels_ &&
              config_.clamp_rigidity_ > 0.01 &&
              corpus.doc_labels_[doc] != -1 && 
              Random() < config_.clamp_rigidity_) {
            if (!doc_already_labeled) {
              labeled_docs_cnt++;
              doc_already_labeled = true;
            }
            new_topic = corpus.doc_labels_[doc];
            if (new_topic > config_.n_topics_)
              cout << "Woah\n" << endl;
          }

          // word labels
          if (config_.use_node_labels_ &&
              config_.clamp_rigidity_ > 0.01 &&
              input_labels_[type][cur_wordid] != -1 && 
              Random() < config_.clamp_rigidity_) {
            labeled_word_cnt++;
            new_topic = input_labels_[type][cur_wordid];
            if (config_.md_n_domains_) 
              new_topic += config_.md_split_start_indexes_[corpus.domains_[doc]];
            if (config_.md_seeds_) {
              double r = rand() * 1.0 / RAND_MAX;
              if (r > config_.md_probs_[0] && r <= (config_.md_probs_[0] + config_.md_probs_[1])) {
                // assign to general instead
                new_topic = input_labels_[type][cur_wordid]; 
                if (config_.md_n_domains_) 
                  new_topic += config_.md_split_start_indexes_[config_.md_n_domains_];
                general_cnt++;
              } else if (r > (config_.md_probs_[0] + config_.md_probs_[1])) {
                // assign to out of domain instead
                int rand_domain = UniformSample(config_.md_n_domains_);
                new_topic = input_labels_[type][cur_wordid] + config_.md_split_start_indexes_[rand_domain];
                ood_cnt++;
              }
            } // end if multi domain
          } else {
            // compute topic CDF.
            for (int topic = 0; topic < config_.n_topics_; ++topic) {
              long double topic_prob = corpus.GetTopicProbability(doc, topic, type) * 
                                       GetWordTopicProbability(topic, type, cur_wordid);
              if (config_.model_real_)
                topic_prob *= time_prob[topic];
              if (config_.mixedness_constraint_) 
                topic_prob *= GetRoleEntropyProbability(type, topic, cur_wordid, config_.lit_weight_, false);
              if (config_.balance_constraint_)
                topic_prob /= GetBalanceEntropyProbability(type, topic, cur_wordid);
              if (config_.volume_constraint_)
                topic_prob /= GetVolumeEntropyProbability(type, topic);
              if (config_.model_targets_)
                topic_prob *= GetTargetProbability(corpus, doc, topic, config_.entity_weight_[type]);

              if (topic == 0)
                cdf[topic] = topic_prob;
              else
                cdf[topic] = cdf[topic - 1] + topic_prob;
            }
            // sample topic using CDF
            new_topic = SampleTopicFromMultinomial(cdf, config_.n_topics_);
          } // end if

          int cur_topic = corpus.word_topic_assignments_[type][doc][word_idx];
          if (new_topic == -1) {
            // if this error msg pops up, investigate, silently carrying on for
            // now to avoid exasperation if we have to quit because of this
            // after hours.
            //if (config_.model_targets_)
            //  cout << " Target prob is " << GetTargetProbability(corpus, doc, topic, config_.entity_weight_[type]) << endl;

            cout << "Couldn't assign new topic - Iteration " << iteration
                 << " Doc " << doc << " Type " << type
                 << " Word " << word_idx << endl;
            //DebugRealAttrs(corpus, doc, cdf, time_prob);
            DisplayColumn(cdf, config_.n_topics_, "CDF", cout);
            cout << "Assignment error " << endl;
            exit(0);
            new_topic = cur_topic;
          }

          if (config_.metropolis_hastings_ && new_topic ==  cur_topic) {
            mh_accept_gt1++;
            if (config_.metro_trace_)
              metro_log_stream_ << "Trace\t" << cur_wordid << "\t" << new_topic << "\t" << new_topic << "\t1\t1\t1\t1\t1\t1\t-2\t1" << endl;
          }
          if (config_.metropolis_hastings_ && new_topic !=  cur_topic) {
            pair<double, pair<double, double> > mh_prob_pair;

            mh_prob_pair = MetropolisTest(corpus, doc, type, cur_wordid, cur_topic, new_topic);
            double mh_prob = mh_prob_pair.first;
            double r = -1;
            bool accepted = true;
            if (mh_prob > 1.0) {
              mh_accept_gt1++;
            } else if ((r = Random()) < mh_prob) {
              mh_accept++;
            } else {
              accepted = false;
              mh_reject++;
              if (config_.metropolis_reject_) {
                new_topic = cur_topic;
              } // end if actual rejection
            } // end if rejected
            if (config_.metro_trace_)
              metro_log_stream_ << r << "\t" << accepted << endl;

            if (!mh_dumped) {
              cout << "Avg Beta Entropy: " << mh_prob_pair.second.first << ", " << mh_prob_pair.second.second << endl;
              mh_dumped = 1;
            }
          } // end metropolis hastings

          // add this word
          corpus.word_topic_assignments_[type][doc][word_idx] = new_topic;
          AddWord(corpus, doc, word_idx, type);

          
          if (debug)
            DebugMCMC(corpus, cdf, cur_topic, new_topic, iteration, doc, word_idx);

          //cout << "EBlah " << corpus.md_entropy_components_[doc][0]->counts_ << ' ';
          //cout << corpus.md_entropy_components_[doc][1]->counts_ << ' ';
          //cout << corpus.md_senti_entropy_components_[doc]->counts_ << endl;
        } // end words
      } // end types of words
      if (config_.metropolis_hastings_) {
        //if (doc % 10 == 0) {
          cout << '+'; cout.flush();
        //}
      }
    } //  end docs

    // Run Link MCMC
    if (config_.model_links_ && links) {
      if (config_.fast_lda_)
        FastLDA(*links, debug);
      else
        InferLinkDistribution(*links, debug);
    }

    if (config_.model_targets_ && config_.n_real_targets_) {
      TrainRegression(corpus);
    }

    cout << " : "; cout.flush();
    // Report perplexities
    if (corpus.n_docs_) {
      ComputePerplexity(corpus, perplexity);
      for (int j = 0; j < config_.n_entity_types_; ++j) {
        cout << perplexity[j] << " ";
      }
    }

    if (config_.model_links_ && links) {
      double link_perplexity = ComputeLinkPerplexity(*links);
      cout << "Link: " << link_perplexity << " ";
    }

    // we are mixed. Let's add model to averager.
    if (iteration >= config_.n_iterations_) {
      Model *photocopy = new Model(*this);
      photocopy->Normalize();
      average_model->Add(*photocopy);
      photocopy->Free();
      delete photocopy;

      // save link and corpu topic pair distribution for averaging
      if (config_.model_links_ && links) {
        links->AddToAverager();
      }
      corpus.AddToAverager();
    }

    // run test inference at every k-th iteration on test_corpus.
    // Making sure we aren't in test phase, to avoid infinite loop
    if ((iteration % 5 == 0) || (iteration == config_.n_iterations_ + config_.n_avg_ - 1)) {
      CheckIntegrity(&corpus, ptr); 
      if (config_.check_integrity_)
        InitializePenaltyTerms(true); // Unit test like tests
      if (test_corpus) {
        SampleTopics(*test_corpus);
        ComputePerplexity(*test_corpus, perplexity, true);
        cout << "Test perplexities: ";
        for (int k = 0; k < config_.n_entity_types_; ++k) {
          cout << perplexity[k] << " ";
        }
      }
      if (config_.model_links_ && test_link_corpus && test_link_corpus->n_links_) {
        if (!test_corpus)
          cout << "Test perplexities: ";
        SampleTopicsForLinks(*test_link_corpus);
        double link_perplexity = ComputeLinkPerplexity(*test_link_corpus, true);
        cout << " Link: " << link_perplexity;
      }
      CalculateAccuracy();
      corpus.CalculateAccuracy();
    } // end if - test perlexity
    if (config_.volume_constraint_)
      cout << " VolumeEntropy: " << GetVolumeEntropy();
    if (labeled_word_cnt) {
      cout << " Used supplied labels for " << labeled_word_cnt;
      cout << " Out of " << labeled_word_cnt << " (" << general_cnt << "," << ood_cnt <<") " << endl;
    }
    if (labeled_docs_cnt) {
      cout << " Used doc labels for " << labeled_docs_cnt << " docs " << endl;
    }

    if (config_.metropolis_hastings_) {
      MHStats m(mh_accept_gt1, mh_accept, mh_reject);
      mh_stats.push_back(m);
      cout << "MH accept: " << (mh_accept_gt1 + mh_accept) * 100.0 / (mh_accept_gt1 + mh_accept + mh_reject) << " ";
      cout << "above-1:" << mh_accept_gt1 * 100.0 / (mh_accept_gt1 + mh_accept + mh_reject) << " ";
      cout << "reject: " << mh_reject * 100.0 / (mh_accept_gt1 + mh_accept + mh_reject) << " ";
      cout << "dbg-total: " << (mh_accept_gt1 + mh_accept + mh_reject) << " ";
      cout << "CacheHitRate: " << log2_.CacheHitRate() * 100 << "% out of " << log2_.total_hits_ << " hits; " << log2_.total_hits_ - log2_.cache_hits_ << " misses;  size of cache: " << log2_.cache_.size();
    }
    cout << endl;
  } // end iterations
  delete[] cdf;

  if (config_.metropolis_hastings_) {
    int gt1 = 0;
    int accept = 0;
    int reject = 0;
    for (vector<MHStats>::iterator it = mh_stats.begin(); it != mh_stats.end(); ++it) {
      gt1 += it->gt1_;
      accept += it->accept_;
      reject += it->reject_;
    }
    cout << "CacheHitRate: " << log2_.CacheHitRate() << endl;
    cout << "Overall MH -- >1:" << gt1 * 100.0 / (gt1 + accept + reject) << " ";
    cout << "accept: " << (gt1 + accept) * 100.0 / (gt1 + accept + reject) << " ";
    cout << "reject: " << (reject) * 100.0 / (gt1 + accept + reject);
    cout << endl;
  }

  cout << "Average model " << "...  : ";
  if (corpus.n_docs_) {
    average_model->ComputePerplexity(corpus, perplexity, false); //Don't smooth with averaged model.
    for (int j = 0; j < config_.n_entity_types_; ++j) {
      cout << perplexity[j] << " ";
      average_model->perplexities_[j] = perplexity[j];
    }
  }
  if (config_.model_links_ && links) {
    links->Average();
    average_model->link_perplexity_ = average_model->ComputeLinkPerplexity(*links, false); // Don't smooth with averaged model.
    cout << "Link: " << average_model->link_perplexity_;
  }
  cout << endl;
  cout << "Saved average over " << config_.n_avg_ << " last iterations" << endl;
  corpus.Save(config_.output_prefix_ + ".debugger");

  CheckIntegrity(&corpus, ptr);
  delete[] time_prob;
  delete[] perplexity;
  cout << "Done with MCMC\n";
  return average_model;
}

pair<double, pair<double, double> > Model::MetropolisTest(Corpus &c, int doc, int type, int word_id, int cur_topic, int new_topic) {
  double term1 = 1.0;
  //    (c.counts_docs_topics_[doc][new_topic] + config_.alpha_) * 1.0 /
  //    (c.counts_docs_topics_[doc][cur_topic] + config_.lit_weight_ - 1 + config_.alpha_);
  double n_k_old = sum_counts_topic_words_[type][cur_topic] + config_.lit_weight_; 
  double n_k_new = sum_counts_topic_words_[type][new_topic]; 
 
  int gamma = static_cast<int>(ceil(config_.beta_[type] * 10));
  double gamma_real = config_.beta_[type];
  double V = config_.vocab_size_[type];
 


  double n_kv_old = counts_topic_words_[type][cur_topic][word_id] + config_.lit_weight_; 
  double n_kv_new = counts_topic_words_[type][new_topic][word_id]; 
 
  double pt1_top    = (n_kv_old - 1 + gamma_real) / (n_k_old - 1 + V * gamma_real);
  double pt1_bottom = (n_kv_old     + gamma_real) / (n_k_old     + V * gamma_real);
 
  double pt2_top    = (n_kv_new + 1 + gamma_real) / (n_k_new + 1 + V * gamma_real);
  double pt2_bottom = (n_kv_new     + gamma_real) / (n_k_new     + V * gamma_real);

  // double term2_pt1  = pow(pt1_top / pt1_bottom, n_kv_old - 1);
  // double term2_pt2  = pow(pt2_top/ pt2_bottom, n_kv_new); 
  // double term2 = term2_pt1 * term2_pt2 * pt2_top / pt1_bottom;
 
  double term2_pt1  = pow(pt1_top / pt1_bottom, (n_kv_old + gamma_real - 1));
  double term2_pt2  = pow(pt2_top / pt2_bottom, (n_kv_new + gamma_real    )); 
  double term2 = term2_pt1 * term2_pt2; 
 
  double numerator = 0.0;
  double dbg_avg_new_entropy = 0.0;
  double dbg_avg_old_entropy = 0.0;

  double max_entropy = -1;
 
  double term5 = 1.0;
  for (int v = 0; v < config_.vocab_size_[type]; ++v) {
    if (v != word_id) {
      double n_kv_old = counts_topic_words_[type][cur_topic][v]; 
      double n_kv_new = counts_topic_words_[type][new_topic][v]; 
     
      double pt1_top    = (n_kv_old + gamma_real) / (n_k_old - 1 + V * gamma_real);
      double pt1_bottom = (n_kv_old + gamma_real) / (n_k_old     + V * gamma_real);
     
      double pt2_top    = (n_kv_new + gamma_real) / (n_k_new + 1 + V * gamma_real);
      double pt2_bottom = (n_kv_new + gamma_real) / (n_k_new     + V * gamma_real);

      double term2_pt1  = pow(pt1_top / pt1_bottom, n_kv_old + gamma_real - 1);
      double term2_pt2  = pow(pt2_top / pt2_bottom, n_kv_new + gamma_real - 1); 
      term5 *= term2_pt1 * term2_pt2;
    }


    double old_entropy = 0.0;
    double new_entropy = 0.0;
    double new_sum = 0.0;
    double old_sum = 0.0;
    for (int k = 0; k < config_.n_topics_; ++k) {
      // these are un-row-normalized i.e. regular betas.
      int x = static_cast<int>(10 * counts_topic_words_[type][k][v] + gamma);
      int dx = static_cast<int>(10 * sum_counts_topic_words_[type][k] + V * gamma);
      
      int a = x;
      if (v == word_id && k == new_topic)
        a = x + 10;
      int da = dx;
      if (k == new_topic)
        da = da + 10;
      new_sum += a * 1.0 / da;
      new_entropy -= (a * 1.0 / da) * (log2_(a) - log2_(da));
 
      int b = x;
      if (v == word_id && k == cur_topic)
        b = b + 10;
      int db = dx;
      if (k == cur_topic)
        db = db + 10;
      old_sum += b * 1.0 / db;
      old_entropy -= (b * 1.0 / db) * (log2_(b) - log2_(db));
 
      //cout << a * 1.0 / da << " ";
    }
    //cout << new_sum << endl;
    new_entropy /= new_sum;
    new_entropy += log(new_sum) / log(2.0);
 
    old_entropy /= old_sum;
    old_entropy += log(old_sum) / log(2.0);
 
    dbg_avg_new_entropy += new_entropy;
    dbg_avg_old_entropy += old_entropy;
    if (old_entropy > max_entropy)
      max_entropy = old_entropy;
    numerator += (-1.0 * new_entropy * new_entropy + old_entropy * old_entropy);
  }
  double term3 = exp(numerator / (2 * config_.mixedness_variance_));
 
  ///   q
  // double q_old =
  //     GetWordTopicProbability(cur_topic, type, word_id) *
  //     c.GetTopicProbability(doc, cur_topic, type);

  // double q_new =
  //     GetWordTopicProbability(new_topic, type, word_id) *
  //     c.GetTopicProbability(doc, new_topic, type);

  // double term4 = q_old / q_new;

  // as - is  old | new
  // new      new | old    
  double h_old = 0.0;
  double h_new = 0.0;
  int n = 0;
  for (int k = 0; k < config_.n_topics_; ++k) {
    int p_old = counts_topic_words_[type][k][word_id];
    int p_new = counts_topic_words_[type][k][word_id];
    if (k == cur_topic)
      p_old += 1;
    if (k == new_topic)
      p_new += 1;
    if (p_old)
      h_old -= p_old * log2_(p_old);
    if (p_new)
      h_new -= p_new * log2_(p_new);

    n += p_old;
  }
  
  h_old /= n;
  h_new /= n;
  h_old += log2_(n);
  h_new += log2_(n);
  double term4 = 1.0; // exp((-1 * h_old * h_old + h_new * h_new) / (2 * config_.mixedness_variance_));

  //  GetRoleEntropyProbability(type, cur_topic, word_id, config_.lit_weight_, false) / 
  //  GetRoleEntropyProbability(type, new_topic, word_id, config_.lit_weight_, false);
  ///   q


  pair <double, pair<double, double> > ret;
  ret.first = term1 * term2 * term5 * term3 * term4;
  if (config_.metro_trace_)
    metro_log_stream_ << "Trace\t" << word_id << "\t" << cur_topic << "\t" << new_topic << "\t" 
                      << term1 << "\t" << term2 << "\t" << term3 << "\t" << term4 << "\t" << term5 << "\t"
                      << ret.first << "\t";
  ret.second.first = dbg_avg_new_entropy / V;
  ret.second.second = max_entropy; //dbg_avg_old_entropy / V;
  // cout << "Third term: " << term3 << " numerator: " << numerator << endl;
  // cout << "avg_new: " << dbg_avg_new_entropy / V << " avg_old :" << dbg_avg_old_entropy / V << endl;
  // cout << "Third-Numerator: " << numerator << endl;
  // cout << "First:" << term1 << "   Second:" << term2 << " = " << term2_pt1 << " * " << term2_pt2 << "     Third:" << term3 << endl;
  // cout << term1 * term2 * term3 << endl;
  return ret;
}

void Model::AddLinks(Links &links, bool remove) {
  int multiplier = 1;
  if (remove)
    multiplier = -1;
  int link_weight = config_.link_weight_;
  int type_1 = config_.link_attr_[0];
  int type_2 = config_.link_attr_[1];

  for (int i = 0; i < links.n_links_; ++i) {
    int t1 = links.link_topic_assignments_[i][0];
    int t2 = links.link_topic_assignments_[i][1];

    int e1 = links.links_[i][0];
    int e2 = links.links_[i][1];

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

void Model::SampleTopicsForLinks(Links &links, bool useInputTopics) {
  //cout << "From sampling in model " << this << endl;
  if (useInputTopics) 
    cout << "Using input topics " << endl;
  links.RandomInit(input_labels_);

  double **cdf = new double*[config_.n_topics_];
  for (int topic = 0; topic < config_.n_topics_; ++topic) {
    cdf[topic] = new double[config_.n_topics_];
  }

  int type_1 = config_.link_attr_[0]; // what entity type do these links refer
  int type_2 = config_.link_attr_[1]; // to. short var to avoid clutter in code.

  for (int iteration = 0; iteration < config_.n_sample_iterations_ + config_.n_avg_; ++iteration) {
    for (int i = 0; i < links.n_links_; ++i) {
      int id_1 = links.links_[i][0];
      int id_2 = links.links_[i][1];

      int new_topic_1 = -1;
      int new_topic_2 = -1;
      double rand_num = 0;


      int cur_topic_1 = links.link_topic_assignments_[i][0];
      int cur_topic_2 = links.link_topic_assignments_[i][1];
      links.link_topic_pair_counts_[cur_topic_1][cur_topic_2]--;

      if (config_.use_node_labels_          && config_.clamp_rigidity_ > 0.01    &&
          input_labels_[type_1][id_1] != -1 && input_labels_[type_2][id_2] != -1 &&
          (rand_num = Random()) < config_.clamp_rigidity_) {
        new_topic_1 = input_labels_[type_1][id_1];
        new_topic_2 = input_labels_[type_2][id_2];
      } else {
        // compute CDF.
        double prev = 0.0;
        for (int topic_1 = 0; topic_1 < config_.n_topics_; ++topic_1) {
          for (int topic_2 = 0; topic_2 < config_.n_topics_; ++topic_2) {

            double alpha = config_.link_alpha_;
            if (topic_1 != topic_2)
              alpha = config_.link_alpha_ / config_.off_diagonal_discount_;
            double this_pair_prob = (links.link_topic_pair_counts_[topic_1][topic_2] + alpha);

            if (!useInputTopics) {
              this_pair_prob        *= GetWordTopicProbability(topic_1, type_1, id_1) *
                                       GetWordTopicProbability(topic_2, type_2, id_2);
            } else {
              this_pair_prob        *= input_topics_[type_1][topic_1][id_1] * input_topics_[type_2][topic_2][id_2]; 
            }
            cdf[topic_1][topic_2] = prev + this_pair_prob;
            prev = cdf[topic_1][topic_2];
          } // end topic 1
        } // end topic 2

        // generate sample.
        SampleTopicPair(cdf, new_topic_1, new_topic_2);
        if (config_.use_node_labels_ && config_.clamp_rigidity_ > 0.01 && rand_num < config_.clamp_rigidity_) {
          if (input_labels_[type_1][id_1] != -1) 
            new_topic_1 = input_labels_[type_1][id_1];
          if (input_labels_[type_2][id_2] != -1) 
            new_topic_2 = input_labels_[type_2][id_2];
        } // end if replace one of the nodes
      } // end if

      links.link_topic_assignments_[i][0] = new_topic_1;
      links.link_topic_assignments_[i][1] = new_topic_2;
      links.link_topic_pair_counts_[new_topic_1][new_topic_2]++;
    } // end links
    if (!useInputTopics && iteration >= config_.n_sample_iterations_)
      links.AddToAverager();
  } // end iterations
  if (!useInputTopics)
    links.Average();

  // Cleanup.
  for (int topic = 0; topic < config_.n_topics_; ++topic)
    delete[] cdf[topic];
  delete[] cdf;
  if (!useInputTopics)
    link_perplexity_ = ComputeLinkPerplexity(links, false);
}

void Model::AddLink(Links &links, int i, bool remove) {
  AddLinkToPenalty(links, i, remove);
  int delta = 1;
  if (remove)
    delta = -1;

  int cur_topic_1 = links.link_topic_assignments_[i][0];
  int cur_topic_2 = links.link_topic_assignments_[i][1];
  int id_1 = links.links_[i][0];
  int id_2 = links.links_[i][1];
  int type_1 = config_.link_attr_[0]; // what entity type do these links refer
  int type_2 = config_.link_attr_[1]; // to. short var to avoid clutter in code.
  int link_weight = config_.link_weight_;

  links.link_topic_pair_counts_[cur_topic_1][cur_topic_2] += delta;
  counts_topic_words_[type_1][cur_topic_1][id_1] += delta * link_weight;
  counts_topic_words_[type_2][cur_topic_2][id_2] += delta * link_weight;
  sum_counts_topic_words_[type_1][cur_topic_1] += delta * link_weight;
  sum_counts_topic_words_[type_2][cur_topic_2] += delta * link_weight;
}

void Model::DebugMCMC(Corpus &corpus, long double *cdf, int cur_topic, int new_topic, int iteration, int doc, int word_idx) {
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
  cout << "Moved from " << cur_topic << " to " << new_topic << endl;
  corpus.DebugDisplay(cout);
  DebugDisplay(cout);
}

void Model::DebugRealAttrs(Corpus &corpus, int doc, long double *cdf, long double *time_prob) {
  for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
    cout  << "Attr " << attr << ": " << "Flag:" << corpus.real_flags_[doc][attr] << " " << corpus.real_values_[doc][attr] << endl;
    DisplayColumn(topic_allocation_counts_[attr], config_.n_topics_, "Topic counts", cout);
    DisplayMatrix(real_stats_[attr], config_.n_topics_, 2, "Time Stats", false, cout);
    DisplayMatrix(beta_parameters_[attr], config_.n_topics_, 2, "Beta", false, cout);
    DisplayMatrix(gaussian_parameters_[attr], config_.n_topics_, 2, "Gaussian", false, cout);
  }
  DisplayColumn(time_prob, config_.n_topics_, "Time Prob", cout);
  DisplayColumn(cdf, config_.n_topics_, "CDF", cout);
}

void Model::AddWordToRealAttrs(Corpus &corpus, int type, int doc, int cur_topic, bool remove) {
  // remove effect of this word from time-related data structures
  int mult = 1;
  if (remove)
     mult = -1;
  int wt = config_.entity_weight_[type];
  for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
    if (corpus.real_flags_[doc][attr]) {
      topic_allocation_counts_[attr][cur_topic] += mult * wt;
      real_stats_[attr][cur_topic][0] += mult * corpus.real_values_[doc][attr] * wt;
      real_stats_[attr][cur_topic][1] +=
          mult * (corpus.real_values_[doc][attr] * corpus.real_values_[doc][attr] * wt);
    } // end if
  } // end real attrs
}

void Model::AddPenaltyWord(int type, int cur_topic, int cur_wordid, double weight, double vol_weight, bool remove) {
  int mult = 1;
  if (remove)
    mult *= -1;

  if (config_.mixedness_constraint_) {
    double num_occurences_of_cur_word = frequencies_[type][cur_wordid];
    double p_cur_topic = counts_topic_words_[type][cur_topic][cur_wordid] / num_occurences_of_cur_word;
    double p_cur_topic_rem_cur_word = (counts_topic_words_[type][cur_topic][cur_wordid] + mult * weight) / num_occurences_of_cur_word;
    //cout << "Remove: " << remove << " Changing word entropy of " << cur_wordid << " type " << type << " from " << word_entropies_[type][cur_wordid];
    if (counts_topic_words_[type][cur_topic][cur_wordid] > 0) 
      word_entropies_[type][cur_wordid] += (p_cur_topic * log (p_cur_topic) / log(2.0)); 
    if ((counts_topic_words_[type][cur_topic][cur_wordid] + mult * weight) > 0) 
      word_entropies_[type][cur_wordid] -= (p_cur_topic_rem_cur_word * log (p_cur_topic_rem_cur_word) / log(2.0)); 
    //cout << " to " << word_entropies_[type][cur_wordid] << endl;
    if (!remove && word_entropies_[type][cur_wordid] > log(config_.n_topics_) / log(2.0))
      cout << "faulty entropy " << endl;
  } // end if

  if (config_.volume_constraint_ && vol_weight > 0) {
    if ((sum_counts_topic_words_[type][cur_topic] + mult * vol_weight) > 0) {
      double p = (sum_counts_topic_words_[type][cur_topic] + mult * vol_weight) / total_volume_;
      volume_entropy_components_[cur_topic] = p * log2(p);
    } else if ((sum_counts_topic_words_[type][cur_topic] + mult * vol_weight) < 0) {
      cout << "What! sum_counts_topic_words_ being pushed below 0 " << endl;
    } else {
      volume_entropy_components_[cur_topic] = 0;
    }
  } // end if volume constraint
}

void Model::AddLinkToPenalty(Links &links, int i, bool remove) {
  int type_1 = config_.link_attr_[0]; // what entity type do these links refer
  int type_2 = config_.link_attr_[1]; // to. short var to avoid clutter in code.
  
  int cur_topic_1 = links.link_topic_assignments_[i][0];
  int cur_topic_2 = links.link_topic_assignments_[i][1];
  int id_1 = links.links_[i][0];
  int id_2 = links.links_[i][1];

  bool self_edge = (type_1 == type_2 && id_1 == id_2 && cur_topic_1 == cur_topic_2);
  int link_weight = config_.link_weight_;
  double discount = link_weight;
  if (self_edge)
    discount = 2 * link_weight;

  double vol_disc_1 = config_.entity_weight_[config_.link_attr_[0]];
  double vol_disc_2 = config_.entity_weight_[config_.link_attr_[1]];
  if (cur_topic_1 == cur_topic_2) {
    vol_disc_1 = vol_disc_1 + vol_disc_2; 
    vol_disc_2 = 0.0;
  }

  AddPenaltyWord(type_1, cur_topic_1, id_1, discount, vol_disc_1, remove);
  if (!self_edge)
    AddPenaltyWord(type_2, cur_topic_2, id_2, link_weight, vol_disc_2, remove);
}

long double Model::GetAverageNodeRoleEntropy(int type) {
  long double tot = 0;
  for (int j = 0; j < config_.vocab_size_[type]; ++j) {
    tot += word_entropies_[type][j];
  }
  return tot / config_.vocab_size_[type];
}

long double Model::GetRoleEntropyProbability(int type, int topic, int id, double weight, bool unnorm = false) {
  if (!config_.mixedness_per_type_flags_[type])
    return 1.0;
  double num_occurences_of_cur_word = frequencies_[type][id];

  double p_candidate_topic_prev = counts_topic_words_[type][topic][id] / num_occurences_of_cur_word;
  double p_candidate_topic      = (counts_topic_words_[type][topic][id] + weight) / num_occurences_of_cur_word;
  double we = word_entropies_[type][id];
  double delta = 0.0;
  if (counts_topic_words_[type][topic][id] > 0) 
    delta += (p_candidate_topic_prev * log2 (p_candidate_topic_prev)); 
  delta -= (p_candidate_topic * log2(p_candidate_topic)); 
  if (unnorm)
    return delta;
  we = we + delta;

  double norm_prob = exp (-(we * we) / (2 * config_.mixedness_variance_per_type_[type]));
  return pow(norm_prob, config_.mixedness_penalty_);
}

long double Model::GetRoleEntropyProbability(Links &links, int i, int topic_1, int topic_2) {
  int type_1 = config_.link_attr_[0]; // what entity type do these links refer
  int type_2 = config_.link_attr_[1]; // to. short var to avoid clutter in code.

  int id_1 = links.links_[i][0];
  int id_2 = links.links_[i][1];

  // Special case when modifying two components of the word entrop distribution
  if (type_1 == type_2 && id_1 == id_2 && topic_1 != topic_2) {
    long double delta_1 = GetRoleEntropyProbability(type_1, topic_1, id_1, config_.link_weight_);
    long double delta_2 = GetRoleEntropyProbability(type_1, topic_2, id_1, config_.link_weight_);
    long double we = word_entropies_[type_1][id_1] + delta_1 + delta_2;

    double norm_prob = exp (-(we * we) / (2 * config_.mixedness_variance_));
    norm_prob = pow(norm_prob, config_.mixedness_penalty_);
    return norm_prob * norm_prob;
  }

  long double norm_prob_1 = 1.0;
  long double norm_prob_2 = 1.0;
  if (type_1 == type_2 && id_1 == id_2 && topic_1 == topic_2) {
    norm_prob_1 = GetRoleEntropyProbability(type_1, topic_1, id_1, 2 * config_.link_weight_);
    norm_prob_2 = norm_prob_1;
  } else {
    norm_prob_1 = GetRoleEntropyProbability(type_1, topic_1, id_1, config_.link_weight_);
    norm_prob_2 = GetRoleEntropyProbability(type_2, topic_2, id_2, config_.link_weight_);
  }

  return norm_prob_1 * norm_prob_2;
}

long double Model::GetTargetProbability(Corpus &corpus, int doc, int topic, int weight) {
  corpus.counts_docs_topics_[doc][topic] += weight;
  PredictTargets(corpus, doc);
  long double target_prob = 1.0;
  for (int i = 0; i < config_.n_real_targets_; ++i) {
    if (corpus.real_target_flags_[doc][i]) {
      long double diff = corpus.pred_targets_[doc][i] - corpus.real_targets_[doc][i];
      target_prob *= exp(-pow(diff, static_cast<long double>(2.0)) / (2 * 4));
    }
  }
  corpus.counts_docs_topics_[doc][topic] -= weight;
  return target_prob;
}

long double Model::GetVolumeEntropy() {
  return -1 * accumulate(volume_entropy_components_, volume_entropy_components_ + config_.n_topics_, 0.0);
}

long double Model::GetVolumeEntropyProbability(int type, int topic) {
  long double entropy = 0.0;
  for (int t = 0; t < config_.n_topics_; ++t) {
    if (t != topic)
      entropy -= volume_entropy_components_[t];
    else {
      double topic_volume = config_.entity_weight_[type];
      for (int iter_type = 0; iter_type < config_.n_entity_types_; ++iter_type) {
        topic_volume += sum_counts_topic_words_[iter_type][t] * config_.entity_weight_[iter_type];
      }
      double p = topic_volume / total_volume_;
      entropy -= p * log2(p);
    } // end if
  } // end topics

  double norm_prob = exp (-(entropy * entropy) / (2 * config_.volume_variance_));
  long double ret = pow(norm_prob, config_.volume_penalty_);
  return ret;
}

long double Model::GetVolumeEntropyProbabilityForLinks(int topic_1, int topic_2) {
  long double entropy = 0.0;
  for (int t = 0; t < config_.n_topics_; ++t) {
    if (t == topic_1 && t == topic_2) {
      double topic_volume = config_.entity_weight_[config_.link_attr_[0]] + config_.entity_weight_[config_.link_attr_[1]];
      for (int iter_type = 0; iter_type < config_.n_entity_types_; ++iter_type) {
        topic_volume += sum_counts_topic_words_[iter_type][t] * config_.entity_weight_[iter_type];
      }
      if (topic_volume > 0) {
        double p = topic_volume / total_volume_;
        entropy -= p * log2(p);
      }
    } else if (t == topic_1) {
      int type = config_.link_attr_[0];
      double topic_volume = config_.entity_weight_[type];
      for (int iter_type = 0; iter_type < config_.n_entity_types_; ++iter_type) {
        topic_volume += sum_counts_topic_words_[iter_type][t] * config_.entity_weight_[iter_type];
      }
      if (topic_volume > 0) {
        double p = topic_volume / total_volume_;
        entropy -= p * log2(p);
      }
    }
    else if (t == topic_2) {
      int type = config_.link_attr_[1];
      double topic_volume = config_.entity_weight_[type];
      for (int iter_type = 0; iter_type < config_.n_entity_types_; ++iter_type) {
        topic_volume += sum_counts_topic_words_[iter_type][t] * config_.entity_weight_[iter_type];
      }
      if (topic_volume > 0) {
        double p = topic_volume / total_volume_;
        entropy -= p * log2(p);
      }
    }
    else {
      entropy -= volume_entropy_components_[t];
    } // end if
  } // end topics

  long double norm_prob = exp (-(entropy * entropy) / (2 * config_.volume_variance_));
  long double ret = pow(norm_prob, static_cast<long double>(config_.volume_penalty_));
  //cout << "Topics - " << topic_1 << ", " << topic_2 << " = " << entropy << " " << ret << endl ;
  return ret;
}

long double Model::GetBalanceEntropyProbability(int type, int topic, int cur_wordid) {
  counts_topic_words_[type][topic][cur_wordid] += config_.lit_weight_;
  sum_counts_topic_words_[type][topic]         += config_.lit_weight_;

  long double updated_balance_entropy = GetClusterBalanceEntropy();
  double norm_prob = exp (-(updated_balance_entropy * updated_balance_entropy) / (2 * config_.balance_variance_));
  long double ret = pow(norm_prob, config_.balance_penalty_);

  counts_topic_words_[type][topic][cur_wordid] -= config_.lit_weight_;
  sum_counts_topic_words_[type][topic]         -= config_.lit_weight_;

  return ret;
}

long double Model::GetBalanceEntropyProbability(Links &links, int i, int topic_1, int topic_2) {
  int type_1 = config_.link_attr_[0]; // what entity type do these links refer
  int type_2 = config_.link_attr_[1]; // to. short var to avoid clutter in code.

  int id_1 = links.links_[i][0];
  int id_2 = links.links_[i][1];

  double link_weight = config_.link_weight_;
  counts_topic_words_[type_1][topic_1][id_1] += link_weight;
  sum_counts_topic_words_[type_1][topic_1]   += link_weight;
  counts_topic_words_[type_2][topic_2][id_2] += link_weight;
  sum_counts_topic_words_[type_2][topic_2]   += link_weight;

  long double updated_balance_entropy = GetClusterBalanceEntropy();
  double norm_prob = exp (-(updated_balance_entropy * updated_balance_entropy) / (2 * config_.balance_variance_));
  long double ret = pow(norm_prob, config_.balance_penalty_);

  counts_topic_words_[type_1][topic_1][id_1] -= link_weight;
  counts_topic_words_[type_2][topic_2][id_2] -= link_weight;
  sum_counts_topic_words_[type_1][topic_1]   -= link_weight;
  sum_counts_topic_words_[type_2][topic_2]   -= link_weight;

  return ret;
}

void Model::CheckFastLDAIntegrity(Links &links) {
  cout << "Checking Fast LDA ... "; cout.flush();
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int word = 0; word < config_.vocab_size_[type]; ++word) {
     double check = 0.0;
      for (int topic = 0; topic < config_.n_topics_; ++topic) {
        check += pow(counts_topic_words_[type][topic][word] + config_.beta_[type], 4);
      } // end topic
      if (abs(norm_[type][word] - check) > 0.000000001) {
        cout << "Word " << word << " of type " << type << " broken" << endl;
      }
    } // end word
  } // end type
  
  double check = 0;
  multiset<TopicPair> check_order;
  for (int topic_1 = 0; topic_1 < config_.n_topics_; ++topic_1) {
    for (int topic_2 = 0; topic_2 < config_.n_topics_; ++topic_2) {
      check_order.insert(TopicPair(topic_1, topic_2, links.link_topic_pair_counts_[topic_1][topic_2]));
      double alpha = 
        (topic_1 == topic_2) ? config_.link_alpha_ : config_.link_alpha_ / config_.off_diagonal_discount_;
      check += pow(links.link_topic_pair_counts_[topic_1][topic_2] + alpha, 2);
    } // end topic_2
  } // end topic 1
  if (abs(norm_1_ - check) > 0.000000001) {
    cout << "Norm 1 broken " << norm_1_ << " vs. " << check << endl;
  }

  multiset<TopicPair>::iterator orig_it  = order_.begin();
  multiset<TopicPair>::iterator check_it = check_order.begin();

  while (orig_it != order_.end()) {
    if (orig_it->count_ != check_it->count_) {
      cout << "Order broken" << endl;
    }
    ++orig_it; ++check_it;
  }
}

void Model::InitNormsForFastLDA(Links &links) {
  cout << "Initializing Fast LDA ... "; cout.flush();
  norm_ = new double*[config_.n_entity_types_];
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    norm_[type] = new double[config_.vocab_size_[type]];
    for (int word = 0; word < config_.vocab_size_[type]; ++word) {
      norm_[type][word] = 0.0;
      for (int topic = 0; topic < config_.n_topics_; ++topic) {
        norm_[type][word] += pow(counts_topic_words_[type][topic][word] + config_.beta_[type], 4);
      } // end topic
    } // end word
  } // end type
  
  norm_1_ = 0;
  for (int topic_1 = 0; topic_1 < config_.n_topics_; ++topic_1) {
    for (int topic_2 = 0; topic_2 < config_.n_topics_; ++topic_2) {
      cout  <<links.link_topic_pair_counts_[topic_1][topic_2] << " ";
      order_.insert(TopicPair(topic_1, topic_2, links.link_topic_pair_counts_[topic_1][topic_2]));
      double alpha = 
        (topic_1 == topic_2) ? config_.link_alpha_ : config_.link_alpha_ / config_.off_diagonal_discount_;
      norm_1_ += pow(links.link_topic_pair_counts_[topic_1][topic_2] + alpha, 2);
    } // end topic_2
  } // end topic 1
  cout << "done " << endl;
}

void Model::DebugFast(map<double, int> &norm_4) {
  double tot  = 0;
  int    tot2 = 0;
  double    tot3 = 0;
  for (map<double, int>::iterator dbg_it = norm_4.begin(); dbg_it != norm_4.end(); ++dbg_it) {
    cout << dbg_it->first << "-" << dbg_it->second << ' ' ;
    tot += dbg_it->first * (dbg_it->second);
    tot3 += dbg_it->first * (dbg_it->second * 1.0 / config_.n_topics_);
    tot2 += dbg_it->second;
  }
  cout << "TOTAL links w/smoothing and topic pairs: " << tot << " " << tot3 << " " << tot2 << '\n' << endl;
}

void Model::FastLDA(Links &links, bool debug) {
  if (!norm_)
    InitNormsForFastLDA(links);

  double *z    = new double[config_.n_topics_ * config_.n_topics_];
  double *sump = new double[config_.n_topics_ * config_.n_topics_];

  int type_1 = config_.link_attr_[0]; // what entity type do these links refer
  int type_2 = config_.link_attr_[1]; // to. short var to avoid clutter in code.
  int link_weight = config_.link_weight_;

  int new_topic_1 = -1;
  int new_topic_2 = -1;

  int pairs_seen = 0;
  int skipped = 0;
  map<int, double> modified;
  for (int i = 0; i < links.n_links_; ++i) {
    int id_1 = links.links_[i][0];
    int id_2 = links.links_[i][1];

    // Remove link
    int cur_topic_1 = links.link_topic_assignments_[i][0];
    int cur_topic_2 = links.link_topic_assignments_[i][1];
    
    // Saving topic pairs whose counts are modified to update the order in
    // which topic pairs are traversed
    int hash = cur_topic_1 * 10000 + cur_topic_2;
    if (modified.count(hash) == 0)
      modified[hash] = links.link_topic_pair_counts_[cur_topic_1][cur_topic_2];

    double alpha = 
      (cur_topic_1 == cur_topic_2) ? config_.link_alpha_ : config_.link_alpha_ / config_.off_diagonal_discount_;
    norm_1_ += (1 - 2 * (links.link_topic_pair_counts_[cur_topic_1][cur_topic_2] + alpha));
    links.link_topic_pair_counts_[cur_topic_1][cur_topic_2] -= 1 ;

    norm_[type_1][id_1] -= pow(counts_topic_words_[type_1][cur_topic_1][id_1] + config_.beta_[type_1], 4);
    counts_topic_words_[type_1][cur_topic_1][id_1] -= link_weight;
    norm_[type_1][id_1] += pow(counts_topic_words_[type_1][cur_topic_1][id_1] + config_.beta_[type_1], 4);

    norm_[type_2][id_2] -= pow(counts_topic_words_[type_2][cur_topic_2][id_2] + config_.beta_[type_2], 4);
    counts_topic_words_[type_2][cur_topic_2][id_2] -= link_weight;
    norm_[type_2][id_2] += pow(counts_topic_words_[type_2][cur_topic_2][id_2] + config_.beta_[type_2], 4);

    sum_counts_topic_words_[type_1][cur_topic_1]   -= link_weight;
    sum_counts_topic_words_[type_2][cur_topic_2]   -= link_weight;

    map<double, int> norm_4;
    map<double, int> norm_5;
    for (int topic = 0; topic < config_.n_topics_; ++topic) {
      norm_4[sum_counts_topic_words_[type_1][topic] + config_.vocab_size_[type_1] * config_.beta_[type_1]] += config_.n_topics_;
      norm_5[sum_counts_topic_words_[type_2][topic] + config_.vocab_size_[type_2] * config_.beta_[type_2]] += config_.n_topics_;
    }
    // DEBUG START
    //DebugFast(norm_4);
    //DebugFast(norm_5);
    // DEBUG END

    double u = Random();
    double n1     = norm_1_;
    double norm_2 = norm_[type_1][id_1] * config_.n_topics_;
    double norm_3 = norm_[type_2][id_2] * config_.n_topics_;
    int pair_idx = 0;


    for (set<TopicPair>::iterator pair_it = order_.begin(); pair_it != order_.end(); ++pair_it, ++pair_idx) {
      ++pairs_seen;
      int t1 = pair_it->t1_;
      int t2 = pair_it->t2_;

      double alpha = 
        (t1 == t2) ? config_.link_alpha_ : config_.link_alpha_ / config_.off_diagonal_discount_;
      n1 -= pow(links.link_topic_pair_counts_[t1][t2] + alpha, 2);
      norm_2 -= pow(counts_topic_words_[type_1][t1][id_1] + config_.beta_[type_1], 4);
      norm_3 -= pow(counts_topic_words_[type_2][t2][id_1] + config_.beta_[type_2], 4);
      double key_1 = sum_counts_topic_words_[type_1][t1] + config_.vocab_size_[type_1] * config_.beta_[type_1];
      if (norm_4.count(key_1) == 0) {
        cout << "After " << pair_idx << " in link " << i << "(" << id_1 << "," << id_2 << ")" << " in " << cur_topic_1 << "," << cur_topic_2 <<  " Inf norm element missing " << key_1 << " for topic " << t1 << endl;
        DebugFast(norm_4);
      }
      if (--norm_4[key_1] == 0)
        norm_4.erase(key_1);
      double key_2 = sum_counts_topic_words_[type_2][t2] + config_.vocab_size_[type_2] * config_.beta_[type_2];
      if (norm_5.count(key_2) == 0) {
        cout << "After " << pair_idx << " in link " << i <<"(" << id_1 << "," << id_2 << ")" <<  " in " << cur_topic_1 << "," << cur_topic_2 <<  " Inf norm element missing " << key_2 << " for topic " << t2 << endl;
        DebugFast(norm_5);
      }
      if (--norm_5[key_2] == 0)
        norm_5.erase(key_2);

      double prev = 0;
      if (pair_idx != 0)
        prev = sump[pair_idx - 1];
      sump[pair_idx] = prev + 
        (links.link_topic_pair_counts_[t1][t2] + alpha) *
        (counts_topic_words_[type_1][t1][id_1] + config_.beta_[type_1]) *
        (counts_topic_words_[type_2][t2][id_2] + config_.beta_[type_2]) /
        (sum_counts_topic_words_[type_1][t1] + config_.vocab_size_[type_1] * config_.beta_[type_1]) /
        (sum_counts_topic_words_[type_2][t2] + config_.vocab_size_[type_2] * config_.beta_[type_2]);
      long double norm = pow(n1, 1.0/2.0) * pow(norm_2, 1.0/4.0) * pow(norm_3, 1.0/4.0) / (norm_4.begin()->first) / (norm_5.begin()->first);
      z[pair_idx] = sump[pair_idx] + norm;

      if (u * z[pair_idx] > sump[pair_idx]) {
        continue;
      }

      if (pair_idx == 0 || u * z[pair_idx] > sump[pair_idx - 1]) {
        new_topic_1 = t1;
        new_topic_2 = t2;
        break;
      }
      u = (u * z[pair_idx - 1] - sump[pair_idx - 1]) * z[pair_idx] / (z[pair_idx - 1] * z[pair_idx]);
      int t = 0;
      for (set<TopicPair>::iterator pair_it2 = order_.begin(); pair_it2 != order_.end(); ++pair_it2, ++t) {
        int t1 = pair_it2->t1_;
        int t2 = pair_it2->t2_;
        if (sump[t] >= u) {
          new_topic_1 = t1;
          new_topic_2 = t2;
          break;
        }
      } // end t
      break;
    } // end pair


    if (new_topic_1 == -1 || new_topic_2 == -1) {
      new_topic_1 = cur_topic_1;
      new_topic_2 = cur_topic_2;
      skipped++;
    }
    links.link_topic_assignments_[i][0] = new_topic_1;
    links.link_topic_assignments_[i][1] = new_topic_2;

    hash = new_topic_1 * 10000 + new_topic_2;
    if (modified.count(hash) == 0)
      modified[hash] = links.link_topic_pair_counts_[new_topic_1][new_topic_2];

    alpha = 
      (new_topic_1 == new_topic_2) ? config_.link_alpha_ : config_.link_alpha_ / config_.off_diagonal_discount_;
    norm_1_ += (1 + 2 * (links.link_topic_pair_counts_[new_topic_1][new_topic_2] + alpha));
    links.link_topic_pair_counts_[new_topic_1][new_topic_2] += 1 ;
    norm_[type_1][id_1] -= pow(counts_topic_words_[type_1][new_topic_1][id_1] + config_.beta_[type_1], 4);
    counts_topic_words_[type_1][new_topic_1][id_1] += link_weight;
    norm_[type_1][id_1] += pow(counts_topic_words_[type_1][new_topic_1][id_1] + config_.beta_[type_1], 4);

    norm_[type_2][id_2] -= pow(counts_topic_words_[type_2][new_topic_2][id_2] + config_.beta_[type_2], 4);
    counts_topic_words_[type_2][new_topic_2][id_2] += link_weight;
    norm_[type_2][id_2] += pow(counts_topic_words_[type_2][new_topic_2][id_2] + config_.beta_[type_2], 4);

    sum_counts_topic_words_[type_1][new_topic_1]   += link_weight;
    sum_counts_topic_words_[type_2][new_topic_2]   += link_weight;

    // DEBUG START
//      cout << "Link "   << i << '(' << id_1        << "," << id_2       << ") "
 //          << "Topics "             << cur_topic_1 << "," << cur_topic_2 
  //         << " => "                << new_topic_1 << "," << new_topic_2  << endl;
    // DEBUG END
  } // end links
  for (map<int, double>::iterator it = modified.begin(); it != modified.end(); ++it) {
    int t1 = it->first / 10000;
    int t2 = it->first - (t1 * 10000);
    pair<multiset<TopicPair>::iterator, multiset<TopicPair>::iterator> range
        = order_.equal_range(TopicPair(t1, t2, it->second));
    multiset<TopicPair>::iterator is = range.first;
    while (is != range.second) {
      if (is->t1_ == t1 && is->t2_ == t2) {
        order_.erase(is);
        break;
      }
      ++is;
    } // end while
    if (is == range.second)
      cout << "Whoa! modified element not found " << endl;

    order_.insert(TopicPair(t1, t2, links.link_topic_pair_counts_[t1][t2]));
  } // end for
  cout << "\nModifed " << modified.size() << " elements in pi_L ";
  cout << "Skipped " << skipped << " links ";
  cout << "Speedup: " << pairs_seen * 1.0 / links.n_links_ << " instead of " << config_.n_topics_ * config_.n_topics_ << endl;

  delete[] z;
  delete[] sump;

  //CheckFastLDAIntegrity(links);
}

void Model::InferLinkDistribution(Links &links, bool debug) {
  double **cdf = new double*[config_.n_topics_];
  for (int topic = 0; topic < config_.n_topics_; ++topic) {
    cdf[topic] = new double[config_.n_topics_];
  }

  int type_1 = config_.link_attr_[0]; // what entity type do these links refer
  int type_2 = config_.link_attr_[1]; // to. short var to avoid clutter in code.
  int link_weight = config_.link_weight_;

  pair<double, double> *word_topic_probability =
      new pair<double, double>[config_.n_topics_];

  for (int i = 0; i < links.n_links_; ++i) {
    int id_1 = links.links_[i][0];
    int id_2 = links.links_[i][1];

    // forget this link - from distribution, from entity distr.
    AddLink(links, i, true);

    int new_topic_1 = -1;
    int new_topic_2 = -1;

    double rand_num = 0.0;
    if (config_.use_node_labels_          && config_.clamp_rigidity_ > 0.01    &&
        input_labels_[type_1][id_1] != -1 && input_labels_[type_2][id_2] != -1 &&
        (rand_num = Random()) < config_.clamp_rigidity_) {
      new_topic_1 = input_labels_[type_1][id_1];
      new_topic_2 = input_labels_[type_2][id_2];
    } else {
      // compute CDF.
      double prev = 0.0;
      bool short_circuit = false;

      for (int topic = 0; topic < config_.n_topics_; ++topic) {
        word_topic_probability[topic].first  = GetWordTopicProbability(topic, type_1, id_1);
        word_topic_probability[topic].second = GetWordTopicProbability(topic, type_2, id_2);
      } // end topic

      for (int topic_1 = 0; topic_1 < config_.n_topics_; ++topic_1) {
        for (int topic_2 = 0; topic_2 < config_.n_topics_; ++topic_2) {
          long double role_entropy_prob = 1.0; 
          long double balance_entropy_prob = 1.0; 
          long double volume_entropy_prob = 1.0; 

          if (config_.mixedness_constraint_) 
            role_entropy_prob = GetRoleEntropyProbability(links, i, topic_1, topic_2);

          if (config_.balance_constraint_)
            balance_entropy_prob = GetBalanceEntropyProbability(links, i, topic_1, topic_2);

          if (config_.volume_constraint_ == 4)
            volume_entropy_prob = GetVolumeEntropyProbabilityForLinks(topic_1, topic_2);

          long double topic_prob = role_entropy_prob;
          if (balance_entropy_prob < 1.0e-307 || volume_entropy_prob < 1.0e-307) {
            new_topic_1 = topic_1;
            new_topic_2 = topic_2;
            short_circuit = true;
            break;
          }

          if (balance_entropy_prob > 0) 
            topic_prob /= balance_entropy_prob;
          if (volume_entropy_prob > 0)
            topic_prob /= volume_entropy_prob;

          if (topic_prob <= 0 || isinf(topic_prob)) {
            cout << "Hmm " << topic_1 << " " << topic_2 << ';'; cout.flush();
          }

          double alpha = config_.link_alpha_;
          if (topic_1 != topic_2)
            alpha = config_.link_alpha_ / config_.off_diagonal_discount_;

          cdf[topic_1][topic_2] = prev +
              topic_prob *
              (links.link_topic_pair_counts_[topic_1][topic_2] + alpha) *
              word_topic_probability[topic_1].first *
              word_topic_probability[topic_2].second;

          prev = cdf[topic_1][topic_2];
        } // end topic 1
        if (short_circuit)
          break;
      } // end topic 2
      // generate sample.
      if (!short_circuit)
        SampleTopicPair(cdf, new_topic_1, new_topic_2);
      // check for the case if only one node has a given clamped label
      if (config_.use_node_labels_ && config_.clamp_rigidity_ > 0.01 && rand_num < config_.clamp_rigidity_) {
        if (input_labels_[type_1][id_1] != -1) 
          new_topic_1 = input_labels_[type_1][id_1];
        if (input_labels_[type_2][id_2] != -1) 
          new_topic_2 = input_labels_[type_2][id_2];
      } // end if replace one of the nodes
    } // end if

    links.link_topic_assignments_[i][0] = new_topic_1;
    links.link_topic_assignments_[i][1] = new_topic_2;
    
    // add this link back - to distribution, entity distr with new shiny topics
    AddLink(links, i, false);
  } // end links

  delete[] word_topic_probability;
  // Cleanup.
  for (int topic = 0; topic < config_.n_topics_; ++topic)
    delete[] cdf[topic];
  delete[] cdf;
}

void Model::SampleTopicPair(double **cdf, int &new_topic_1, int &new_topic_2) {
  double unif_sample = rand() * 1.0 / RAND_MAX;
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
}

double Model::ComputeLinkPerplexity(Links &links, bool smooth) {
  long double sum_perplexity = 0.0;
  int type_1 = config_.link_attr_[0]; 
  int type_2 = config_.link_attr_[1]; 
  double beta_norm_1 = config_.vocab_size_[type_1] * config_.beta_[type_1]; 
  double beta_norm_2 = config_.vocab_size_[type_2] * config_.beta_[type_2]; 

  for (int i = 0; i < links.n_links_; ++i) {
    long double link_probability = 0.0;
    int e_1 = links.links_[i][0];
    int e_2 = links.links_[i][1];
    double alpha_link_norm = 0.0;
    
    double debug_ctr = 0;
    for (int t_1 = 0; t_1 < config_.n_topics_; ++t_1) {
      for (int t_2 = 0; t_2 < config_.n_topics_; ++t_2) {
        double alpha = config_.link_alpha_;
        if (t_1 != t_2) {
          alpha = config_.link_alpha_ / config_.off_diagonal_discount_;
        }  
        alpha_link_norm += alpha;
        debug_ctr += links.link_topic_pair_counts_[t_1][t_2] + alpha; 
        
        link_probability +=
            (links.link_topic_pair_counts_[t_1][t_2] + alpha) *
            (counts_topic_words_[type_1][t_1][e_1] + config_.beta_[type_1] * smooth) / 
            (sum_counts_topic_words_[type_1][t_1] + beta_norm_1 * smooth) * 
            (counts_topic_words_[type_2][t_2][e_2] + config_.beta_[type_2] * smooth) / 
            (sum_counts_topic_words_[type_2][t_2] + beta_norm_2 * smooth);
      } // end topic 1
    } // end topic 2
    // incorporating normalizing constant for pair probability in one swoop.
    if (abs(debug_ctr - (links.n_links_ + alpha_link_norm)) > 0.001 ) 
      cout << "Foulness  in link perplexity " << debug_ctr << " " << (links.n_links_ + alpha_link_norm) << "\n";
    link_probability /= (links.n_links_ + alpha_link_norm);
    sum_perplexity += log2(link_probability);
  } // end links
  long double average_perplexity = sum_perplexity / links.n_links_;

  average_perplexity = pow(static_cast<long double>(2.0), -1.0 * average_perplexity);
  return average_perplexity;
}

void Model::AddCorpus(Corpus &c, bool remove) {
  for (int i = 0; i < c.n_docs_; ++i) {
    AddDocument(c, i);
  }
}

void Model::AddDocument(Corpus &c, int doc, bool remove) {
  int mult = 1;
  if (remove)
    mult = -1;

  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int word = 0; word < c.doc_num_words_[type][doc]; ++word) {
      int cur_topic = c.word_topic_assignments_[type][doc][word];
      int cur_wordid = c.corpus_words_[type][doc][word];
      counts_topic_words_[type][cur_topic][cur_wordid] += mult * config_.lit_weight_;
      sum_counts_topic_words_[type][cur_topic]         += mult * config_.lit_weight_;
    } // end word
  } // end type

  // Make changes to time related structures.
  if (config_.model_real_) {
    for (int attr = 0; attr < config_.n_real_valued_attrs_; ++attr) {
      for (int topic = 0; topic < config_.n_topics_; ++topic) {
        if (c.real_flags_[doc][attr]) {
          real_stats_[attr][topic][0] += mult *
              c.counts_docs_topics_[doc][topic] *
                  c.real_values_[doc][attr];
          real_stats_[attr][topic][1] += mult *
              c.counts_docs_topics_[doc][topic] *
                  c.real_values_[doc][attr] * c.real_values_[doc][attr];
          topic_allocation_counts_[attr][topic] += mult *
              c.counts_docs_topics_[doc][topic];
        }
      } // end topic
    } // end attributes
  } // end time modeling if.
}

void Model::RemoveDocument(Corpus &c, int doc){
  AddDocument(c, doc, true);
}

void Model::ComputePerplexity(Corpus &c, double *perplexity, bool smooth) {
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    perplexity[type] = 0.0;
    for (int doc = 0; doc < c.n_docs_; ++doc) {

      // computing denominator for doc-topic distribution.
      double doc_topic_normalizer = config_.alpha_ * config_.n_topics_; // prior
      for (int topic = 0; topic < config_.n_topics_; ++topic) {
        doc_topic_normalizer += c.counts_docs_topics_[doc][topic];
      }

      double diagnostic = config_.alpha_ * config_.n_topics_; // prior
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
              (c.counts_docs_topics_[doc][topic] + config_.alpha_) *
              (counts_topic_words_[type][topic][cur_wordid] + config_.beta_[type] * smooth) /
              (sum_counts_topic_words_[type][topic] + config_.vocab_size_[type] * config_.beta_[type] * smooth);
        } // end topics
        perplexity[type] = perplexity[type] +
            log2(word_perplexity / doc_topic_normalizer);
      } // end word
    } // end doc
    perplexity[type] = perplexity[type] /
        accumulate(c.doc_num_words_[type], c.doc_num_words_[type] + c.n_docs_, 0);
    perplexity[type] = pow(2.0, -1.0 * perplexity[type]);
  } // end type
}

void Model::Normalize() {
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    double num_words_of_type = accumulate(sum_counts_topic_words_[type],
                                          sum_counts_topic_words_[type] + config_.n_topics_, 0.0);
    for (int i = 0; i < config_.n_topics_; ++i) {
      topic_weights_[type][i] = sum_counts_topic_words_[type][i] / (num_words_of_type * 1.0);
      for (int j = 0; j < config_.vocab_size_[type]; ++j) {
        counts_topic_words_[type][i][j] = (counts_topic_words_[type][i][j] + config_.beta_[type])/
            (sum_counts_topic_words_[type][i] + config_.vocab_size_[type] * config_.beta_[type]);
      } // end word
      sum_counts_topic_words_[type][i] = 1.0;
    } // end topic
  } // end type
  normalized_ = 1;
}

void Model::Save(const string &model_file_name_prefix) {
  if (!normalized_) 
    Normalize();
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    // setup stream to save parameters.
    ostringstream oss;
    oss                   << model_file_name_prefix << "." << type;
    ofstream ofs(oss.str().c_str());
    if (!ofs) {
      cout << "Cannot save model with prefix " << model_file_name_prefix << endl;
      return;
    }

    double normalizer = accumulate(topic_weights_[type], topic_weights_[type] + config_.n_topics_, 0.0);
    for (int i = 0; i < config_.n_topics_; ++i) {
      double topic_weight = topic_weights_[type][i] / normalizer;
      for (int j = 0; j < config_.vocab_size_[type]; ++j) {
        ofs << counts_topic_words_[type][i][j] << ' ';
      } // end words.

      // Output fraction of entities of type that were assinged to this topic.
      ofs << topic_weight << '\n';
    }
    ofs.close();

    // Saving topics assigned to words.
    // setup stream to save most likely topic for word.
    ostringstream oss_most_likely_topic;
    oss_most_likely_topic << model_file_name_prefix << "." << type << ".mltopic";
    ofstream mlt_ofs(oss_most_likely_topic.str().c_str());

    for (int j = 0; j < config_.vocab_size_[type]; ++j) {
      mlt_ofs << GetWinningTopic(type, j) << endl;
    } // end words
    mlt_ofs.close();

    // Saving topics that have been assigned to words.
    ostringstream oss_topic_distribution;
    oss_topic_distribution << model_file_name_prefix << "." << type << ".word_topic_distr";
    ofstream wtd_ofs(oss_topic_distribution.str().c_str());

    for (int j = 0; j < config_.vocab_size_[type]; ++j) {
      double word_occurence_count = 0;
      for (int t = 0; t < config_.n_topics_; ++t)
        word_occurence_count += counts_topic_words_[type][t][j];
      for (int t = 0; t < config_.n_topics_; ++t) {
        wtd_ofs << counts_topic_words_[type][t][j] / word_occurence_count;
        if (t != config_.n_topics_ - 1)
          wtd_ofs << ' ';
      }
      wtd_ofs << '\n';
    } // end words
    wtd_ofs.close();
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

  if (config_.model_targets_) {
    ostringstream oss;
    oss << model_file_name_prefix << ".regression";
    ofstream ofs(oss.str().c_str(), ios::binary);
    for (int target = 0; target < config_.n_real_targets_; ++target) {
      if (regressors_[target])
        dlib::serialize(*regressors_[target], ofs);
      else 
        cout << "WTF: no regressor " << oss.str() << endl;

      ostringstream oss2;
      oss2 << model_file_name_prefix << ".coefficients.target." << target;
      ofstream ofs2(oss2.str().c_str());
      for (int topic = 0; topic < config_.n_topics_; ++topic)
        ofs2 << regressors_[target]->basis_vectors(0)(topic) << endl;
      ofs2 << regressors_[target]->b << endl;
      ofs2.close();
    }
    ofs.close();
  } // end if
}

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
  if (freed_) {
    cout << "Whoa! Already freed\n";
  }

  for (int type = 0; type < config_.n_entity_types_; ++type) {
    if (norm_)
      delete[] norm_[type];
    if (config_.use_node_labels_ && input_labels_) {
      delete[] input_labels_[type];
    }
    for (int i = 0; i < config_.n_topics_; ++i) {
      delete[] counts_topic_words_[type][i];
    }
    delete[] word_entropies_[type];
    delete[] frequencies_[type];
    delete[] topic_[type];
    if (true_labels_)
      delete[] true_labels_[type];
    delete[] counts_topic_words_[type];
    delete[] sum_counts_topic_words_[type];
    delete[] topic_weights_[type];
  }
  if (norm_)
    delete[] norm_;

  if (config_.use_node_labels_ && input_labels_)
    delete[] input_labels_;

  delete[] topic_;
  if (true_labels_)
    delete[] true_labels_;
  delete[] perplexities_;
  delete[] frequencies_;
  delete[] word_entropies_;
  delete[] topic_sizes_;
  delete[] counts_topic_words_;
  delete[] sum_counts_topic_words_;
  delete[] topic_weights_;
  delete[] volume_entropy_components_;

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

  if (config_.model_targets_) {
    for (int target = 0; target < config_.n_real_targets_; ++target) {
      if (regressors_[target] != NULL)
        delete regressors_[target];
    }
    delete[] regressors_;
  }
  freed_ = 1;
}

void Model::CheckIntegrity(Corpus *corpus, Links *links) {
  if (!config_.check_integrity_)
    return;
  cout << "Checking integrity\n";
  Model test_model(config_);
  test_model.Allocate();
  if (corpus) {
    corpus->CheckIntegrity();
    test_model.AddCorpus(*corpus);
  }
  if (links) {
    links->CheckIntegrity();
    test_model.AddLinks(*links);
  }

 for (int type = 0; type < config_.n_entity_types_; ++type) {
    vector<double> sums(config_.n_topics_);
    for (int word = 0; word < config_.vocab_size_[type]; ++word) {
      double freq = 0;
      long double entropy = 0.0;
      for (int topic = 0; topic < config_.n_topics_; ++topic) {
        if (counts_topic_words_[type][topic][word] != test_model.counts_topic_words_[type][topic][word]) {
          cout << "Model error in type " << type << " topic " << topic << " word " << word <<  " " << counts_topic_words_[type][topic][word] << " " << test_model.counts_topic_words_[type][topic][word] << endl;
        }
        freq += counts_topic_words_[type][topic][word];
        sums[topic] += counts_topic_words_[type][topic][word];
        entropy -= counts_topic_words_[type][topic][word] * log(counts_topic_words_[type][topic][word]);
      } // end topic
      entropy /= freq;
      entropy += log(freq);
      entropy /= log(2.0);

      if (config_.mixedness_constraint_ && abs(entropy - word_entropies_[type][word]) > 0.0001) {
        cout << "entropies does not match counts in type " << type << " word " << word
             << " is " << word_entropies_[type][word] << " should be " << entropy << endl;
      }

      if (config_.mixedness_constraint_ && freq != frequencies_[type][word]) {
        cout << "frequencies_ does not match counts in type " << type << " word " << word 
             << " is " << frequencies_[type][word] << " should be " << freq << endl;
      }
    } // end word

    for (int topic = 0; topic < config_.n_topics_; ++topic) {
      if (sums[topic] != sum_counts_topic_words_[type][topic]) {
        cout << "sum_counts_topic_words_ does not match counts in type " << type << " topic " << topic << endl;
      }
    } // end topic

  } // end type

  if (config_.volume_constraint_) {
    if (!corpus || corpus->n_docs_ == 0) {
      if (links->n_links_ * (config_.entity_weight_[config_.link_attr_[0]] + config_.entity_weight_[config_.link_attr_[1]]) != total_volume_) {
        cout << "Total volume is " << total_volume_ << " should be "  
             << links->n_links_ * (config_.entity_weight_[config_.link_attr_[0]] + config_.entity_weight_[config_.link_attr_[1]])  << endl;
      } // end check
    } // end if only links


    double total_volume = 0;
    for (int t = 0; t < config_.n_topics_; ++t) {
      for (int type = 0; type < config_.n_entity_types_; ++type) {
        total_volume += sum_counts_topic_words_[type][t] * config_.entity_weight_[type];
      } // end type
    } // end topic
    if (total_volume != total_volume_) {
      cout << "Total volume should be " << total_volume << " but is " << total_volume_;
    }

    for (int t = 0; t < config_.n_topics_; ++t) {
      double topic_volume = 0.0;
      for (int type = 0; type < config_.n_entity_types_; ++type) {
        topic_volume += sum_counts_topic_words_[type][t] * config_.entity_weight_[type]; 
      } // end type
      double debug = topic_volume / total_volume_ * log2(topic_volume / total_volume_);
      if (abs(debug - volume_entropy_components_[t]) > 0.00001)
        cout << "Volume entropy component for topic " << t << " broken" << endl;
    } // end topic

  } // end check vol constraint

  if (config_.balance_constraint_) {
    double size = 0;
    for (int topic = 0; topic < config_.n_topics_; ++topic) {
      size += topic_sizes_[topic];
    }
    int true_size = 0;
    for (int type = 0; type < config_.n_entity_types_; ++type) {
      true_size += config_.vocab_size_[type] * config_.entity_weight_[type];
    } 
        
    if (abs(size - true_size) > 0.0001) {
      cout << "topic_sizes_ does not match counts - should be " << true_size << " is " << size << endl;
    }
  } // end if balance constraint enforced i.e. topic_sizes_ is populated

  test_model.Free();
  cout << "Done checking\n" << endl;
}

// train regression
// use in sample topics and mcmc
// corpus integrity with theta entropy

void Model::TrainRegression(Corpus &corpus) {
  sample_type m;
  cout << " dlib ... "; cout.flush();
//  m.set_size(config_.n_topics_, 1);


  for (int target = 0; target < config_.n_real_targets_; ++target) {
    if (regressors_[target] != NULL)
      delete regressors_[target];

    std::vector<sample_type> samples;
    samples.reserve(corpus.n_docs_);
    for (int i = 0; i < corpus.n_docs_; ++i) {
      if (!corpus.real_target_flags_[i][target])
        continue;
      for (int j = 0; j < config_.n_topics_; ++j) {
        m(j) = (corpus.counts_docs_topics_[i][j] + config_.alpha_) / (corpus.weight_[i]);
      }
      for (int j = config_.n_topics_; j < 100; ++j)
        m(j) = 0;
      samples.push_back(m);
    }

    std::vector<double> labels;
    labels.reserve(corpus.n_docs_);
    for (int i = 0; i < corpus.n_docs_; ++i) {
      if (!corpus.real_target_flags_[i][target])
        continue;
      labels.push_back(corpus.real_targets_[i][target]);
    } // end of corpus

    dlib::rr_trainer<kernel_type> trainer;
    //trainer.be_verbose();
    regressors_[target] = new dlib::decision_function<kernel_type>(trainer.train(samples, labels));
  } // end target
}

void Model::PredictTargetsFromTheta(Corpus &corpus) {
  sample_type m;
  //m.set_size(config_.n_topics_, 1);

  for (int doc = 0; doc < corpus.n_docs_; ++doc) {
    for (int j = 0; j < config_.n_topics_; ++j)
      m(j) = corpus.theta_[doc][j];
    for (int j = config_.n_topics_; j < 100; ++j)
      m(j) = 0;

    for (int target = 0; target < config_.n_real_targets_; ++target) {
      if (!regressors_[target]) {
        cout << "Cannot predict target - Regressor null" << endl;
        exit(0);
      }
      corpus.pred_targets_[doc][target] = (*regressors_[target])(m);
    } // end target
  } // end doc
}

void Model::PredictTargets(Corpus &corpus, int doc) {
  sample_type m;
  //m.set_size(config_.n_topics_, 1);

  for (int j = 0; j < config_.n_topics_; ++j)
    m(j) = (corpus.counts_docs_topics_[doc][j] + config_.alpha_) / (corpus.weight_[doc]);
  for (int j = config_.n_topics_; j < 100; ++j)
    m(j) = 0;

  for (int target = 0; target < config_.n_real_targets_; ++target) {
    if (!regressors_[target]) {
      cout << "Cannot predict target - Regressor null" << endl;
      exit(0);
    }
    corpus.pred_targets_[doc][target] = (*regressors_[target])(m);
  } // end 
}

void Model::LoadBeta(const string &model_file_name_prefix) {
  input_topics_ = new double**[config_.n_entity_types_];
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    input_topics_[type] = new double*[config_.n_topics_];
    ostringstream oss;
    oss                   << model_file_name_prefix << "." << type;
    ifstream ifs(oss.str().c_str());
    if (!ifs) {
      cout << "Cannot read multinomials from " << oss.str() << endl;
      return;
    }
    for (int topic = 0; topic < config_.n_topics_; ++topic) {
      input_topics_[type][topic] = new double[config_.vocab_size_[type]];
      string line;
      if (!ifs) {
        cout << "Not enough topics in input model" << endl;
      }
      getline(ifs, line);
      istringstream iss(line);
      long double sum = 0.0;
      for (int id = 0; id < config_.vocab_size_[type]; ++id) {
        if (!iss) {
          cout << "Input model does not have sufficient entries id " << id << endl;
        }
        iss >> input_topics_[type][topic][id];
        sum += input_topics_[type][topic][id];
        //cout << "topic " << topic << " id " << id << " " << input_topics_[type][topic][id];
      } // end id
      for (int id = 0; id < config_.vocab_size_[type]; ++id) {
        input_topics_[type][topic][id] /= sum;
      }
    } // end topic
    ifs.close();
  } // end type
}

void Model::UnloadBeta() {
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int topic = 0; topic < config_.n_topics_; ++topic) {
      delete[] input_topics_[type][topic];
    }
    delete[] input_topics_[type];
  }
  delete[] input_topics_;
}

void Model::LoadLabels(const string &label_file) {
  input_labels_ = ReadLabelFile(label_file);
  if (!input_labels_) {
    cout << "Invalid input labels file" << endl;
    return;
  }
}

int** Model::ReadLabelFile(const string &label_file) {
  int **labels = new int*[config_.n_entity_types_];
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    labels[type] = new int[config_.vocab_size_[type]];
    for (int id = 0; id < config_.vocab_size_[type]; ++id) {
      labels[type][id] = -1;
    }
  } // end type

  ifstream ifs(label_file.c_str());
  if (!ifs) {
    cout << "Cannot open node label file " << label_file << endl;
    return NULL;
  } else {
    cout << "Reading node labels from " << label_file << endl;
  }

  string line;
  int ctr = 0;
  while (getline(ifs, line)) {
    istringstream iss(line);
    int type, id, label;
    iss >> type >> id >> label;
    if (type >= config_.n_entity_types_ || label >= config_.n_topics_ || id >= config_.vocab_size_[type]) {
      cout << ctr << ". Type " << type << " Label " << label << " id " << id << " inconsistent with config\n";
    } else {
      labels[type][id] = label;
    }
    ctr++;
  }
  ifs.close();
  cout << "Read " << ctr << " labels" << endl;
  return labels;
}

void Model::SampleFromInputTopics(const string &model_file_prefix, Corpus &corpus, Links *links) {
  LoadBeta(model_file_prefix);
  if (corpus.n_docs_)
    SampleTopics(corpus, true);
  if (links && links->n_links_)
    SampleTopicsForLinks(*links, true);
  UnloadBeta();
}

void Model::SampleFromFakeInputTopics(Corpus &corpus, Links *links) {
  cout << "Using fake input topics" << endl;
  Model link_lda(config_);
  link_lda.Allocate();
  if (corpus.n_docs_) {
    link_lda.AddCorpus(corpus);
    link_lda.SampleTopics(corpus);
  }
  if (links && links->n_links_) {
    link_lda.AddLinks(*links);
    link_lda.SampleTopicsForLinks(*links);
  }
  link_lda.Free();
}

void Model::CalculateAccuracy(Stats *stats) {
  if (config_.hungarian_flag_) {
    vector<double> hungarian_accuracies = GetAccuracyFromHungarian();
    ostringstream oss_p;
    oss_p << "BestAlign:";
    for (int i = 0; i < config_.n_entity_types_; ++i) {
      oss_p << setprecision(2) << hungarian_accuracies[i];
      if (i != config_.n_entity_types_ - 1)
        oss_p << ',';
      if (stats) {
        ostringstream oss;
        oss << "type_" << i << "_hungarian_accuracy";
        stats->Save(oss.str(), hungarian_accuracies[i]);
      }
    } // end for
    if (config_.n_entity_types_ > 1) {
      oss_p << setprecision(2) << ",Avg=" << hungarian_accuracies[config_.n_entity_types_];
      if (stats) {
        stats->Save("weighted_hungarian_accuracy", hungarian_accuracies[config_.n_entity_types_]);
      }
    }
    if (!stats)
     cout << "  " << oss_p.str();
  } // end if

  if (config_.nmi_flag_) {
    vector<double> nmi = GetNMI();
    ostringstream oss_p;
    oss_p << "NMI:";
    for (int i = 0; i < config_.n_entity_types_; ++i) {
      oss_p << setprecision(2) << nmi[i];
      if (i != config_.n_entity_types_ - 1)
        oss_p << ',';
      if (stats) {
        ostringstream oss;
        oss << "type_" << i << "_nmi";
        stats->Save(oss.str(), nmi[i]);
      }
    } // end for
    if (config_.n_entity_types_ > 1) {
      oss_p << setprecision(2) << ",Avg=" << nmi[config_.n_entity_types_];
      if (stats) {
        stats->Save("weighted_nmi", nmi[config_.n_entity_types_]);
      }
    }
    if (!stats)
      cout << "  " << oss_p.str();
  } // end if

  if (config_.knn_flag_) {
    vector<vector<double> > knn = GetKNN();
    int K[] = {1, 3, 5};
    for (int k = 0; k < 3; ++k) {
      ostringstream oss_p;
      oss_p << K[k] << "-NN:";
      for (int i = 0; i < config_.n_entity_types_; ++i) {
        oss_p << setprecision(2) << knn[i][k];
        if (i != config_.n_entity_types_ - 1)
          oss_p << ',';
        if (stats) {
          ostringstream oss;
          oss << "type_" << i << "_" << K[k]<< "-nn";
          stats->Save(oss.str(), knn[i][k]);
        }
      } // end types
      if (config_.n_entity_types_ > 1) {
        oss_p << setprecision(2) << ",Avg=" << knn[config_.n_entity_types_][k];
        if (stats) {
          ostringstream oss;
          oss << "weighted_" << K[k] << "-nn";
          stats->Save(oss.str(), knn[config_.n_entity_types_][k]);
        }
      }
      if (!stats)
        cout << "  " << oss_p.str();
    } // end K
  } // end if
}

vector<double> Model::GetAccuracyFromHungarian() {
  vector<double> accuracies;

  if (true_labels_ == NULL) {
    if (config_.true_label_file_ == "") {
      cout << "No true label file provided so exiting hungarian accuracy module" << endl;
      return accuracies;
    }
    true_labels_ = ReadLabelFile(config_.true_label_file_);
    if (!true_labels_) {
      cout << "Invalid true label file" << endl;
      return accuracies;
    }
  }

  int **cost_matrix = new int*[config_.n_topics_]; 
  hungarian_problem_t problem;

  int n_true_classes = GetNumTrueClasses();
  for (int t = 0; t < config_.n_topics_; ++t)  {
    cost_matrix[t] = new int[n_true_classes];
    for (int c = 0; c < n_true_classes; ++c)
      cost_matrix[t][c] = 0;
  } // end for

  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int i = 0; i < config_.vocab_size_[type]; ++i) {
      int pred_label = GetWinningTopic(type, i);
      int true_label = true_labels_[type][i];
      if (true_label == -1) 
          continue;

      for (int c = 0; c < n_true_classes; ++c) {
        // For the true class, add a penalty to match it with any topic that's not the pred label.
        if (c != true_label) 
          cost_matrix[pred_label][c] += config_.entity_weight_[type];
      }
    } // end word
  } // end type

  hungarian_init(&problem, cost_matrix, config_.n_topics_, n_true_classes, HUNGARIAN_MODE_MINIMIZE_COST);
  // hungarian_print_costmatrix(&problem);
  hungarian_solve(&problem);
  // hungarian_print_assignment(&problem);

  // get assignment
  vector<int> matched_class(config_.n_topics_);
  for (int i = 0; i < config_.n_topics_; ++i) {
    for (int j = 0; j < n_true_classes; ++j) {
      if (problem.assignment[i][j] == 1) {
        matched_class[i] = j;
        cout << "Node " << i << " == Class " << j << endl;
        break;
      } // end if
    } // end class
  } // end class

  hungarian_free(&problem);

  // get predicted labels and compare
  int big_total = 0;
  int correct_total = 0;
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    int total = 0;
    int correct = 0;
    for (int i = 0; i < config_.vocab_size_[type]; ++i) {
      int pred_topic = GetWinningTopic(type, i);
      int true_label = true_labels_[type][i];

      int pred_label = matched_class[pred_topic];
      if (pred_label == true_label)
        correct++;
      total++;
    } // end word
    big_total += total * config_.entity_weight_[type];
    correct_total += correct * config_.entity_weight_[type];
    accuracies.push_back(correct * 1.0 / total);
  } // end type
  accuracies.push_back(correct_total * 1.0 / big_total);

  for (int i = 0; i < config_.n_topics_; ++i)
    delete[] cost_matrix[i];
  delete[] cost_matrix; 
  return accuracies;
}

vector<double> Model::GetNMI() {
  vector<double> nmi;

  if (true_labels_ == NULL) {
    if (config_.true_label_file_ == "") {
      cout << "No true label file provided so exiting hungarian accuracy module" << endl;
      return nmi;
    }
    true_labels_ = ReadLabelFile(config_.true_label_file_);
    if (!true_labels_) {
      cout << "Invalide true labels file" << endl;
      return nmi;
    }
  }

  int n_true_classes    = GetNumTrueClasses();
  int *pred_distr       = new int[config_.n_topics_];
  int *pred_distr_big   = new int[config_.n_topics_];
  int *true_distr       = new int[n_true_classes];
  int *true_distr_big   = new int[n_true_classes];
  int **contingency     = new int*[config_.n_topics_]; 
  int **contingency_big = new int*[config_.n_topics_]; 
  for (int t = 0; t < config_.n_topics_; ++t) {
    contingency[t]     = new int[n_true_classes];
    contingency_big[t] = new int[n_true_classes];
  }

  for (int t = 0; t < config_.n_topics_; ++t)  {
    pred_distr_big[t] = 0;
    for (int c = 0; c < n_true_classes; ++c) {
      contingency_big[t][c] = 0;
      if (t == 0)
        true_distr_big[c] = 0;
    } // end classes
  } // end topics

  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int t = 0; t < config_.n_topics_; ++t)  {
      pred_distr[t] = 0;
      for (int c = 0; c < n_true_classes; ++c) {
        contingency[t][c] = 0;
        if (t == 0)
          true_distr[c] = 0;
      } // end classes
    } // end topics

    for (int i = 0; i < config_.vocab_size_[type]; ++i) {
      int pred_label = GetWinningTopic(type, i);
      int true_label = true_labels_[type][i];
      if (true_label == -1)
        continue;
      contingency[pred_label][true_label]++;
      contingency_big[pred_label][true_label] += config_.entity_weight_[type];
      pred_distr[pred_label]++;
      pred_distr_big[pred_label] += config_.entity_weight_[type];
      true_distr[true_label]++;
      true_distr_big[pred_label] += config_.entity_weight_[type];
    } // end word

    double h_cond = 0.0;
    for (int i = 0; i < config_.n_topics_; ++i) {
      h_cond += (pred_distr[i] * GetEntropy(contingency[i], n_true_classes));
      //cout << endl;
      //for (int c = 0; c < n_true_classes; ++c)
      //  cout << contingency[i][c] << ' ';
      //cout << endl;
    } // end class
    h_cond /= accumulate(pred_distr, pred_distr + n_true_classes, 0.0);
    double h_true = GetEntropy(true_distr, n_true_classes);
    double h_pred = GetEntropy(pred_distr, config_.n_topics_);
    double type_nmi = 2 * (h_true - h_cond) / (h_true + h_pred);
    nmi.push_back(type_nmi);
  } // end type

  double h_cond = 0.0;
  for (int i = 0; i < config_.n_topics_; ++i) {
    h_cond += (pred_distr_big[i] * GetEntropy(contingency_big[i], n_true_classes));
  } // end class
  h_cond /= accumulate(pred_distr_big, pred_distr_big + n_true_classes, 0.0);
  double h_true = GetEntropy(true_distr_big, n_true_classes);
  double h_pred = GetEntropy(pred_distr_big, config_.n_topics_);
  double big_nmi = 2 * (h_true - h_cond) / (h_true + h_pred);
  nmi.push_back(big_nmi);

  for (int i = 0; i < config_.n_topics_; ++i) {
    delete[] contingency[i];
    delete[] contingency_big[i];
  }
  delete[] contingency; 
  delete[] contingency_big; 
  delete[] pred_distr;
  delete[] pred_distr_big;
  delete[] true_distr;
  delete[] true_distr_big;
  return nmi;
}

double Model::GetEntropy(int *distr, int n) {
  double h = 0.0;
  double sum = accumulate(distr, distr + n, 0) * 1.0;
  for (int i = 0; i < n; ++i) {
    if (distr[i] > 0)
      h -= (distr[i] / sum) * log2(distr[i] / sum);
  } // end for
  return h;
}

int Model::GetNumTrueClasses() {
  int max = 0;
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int i = 0; i < config_.vocab_size_[type]; ++i) {
      if (true_labels_[type][i] > max)
        max = true_labels_[type][i];
      if (true_labels_[type][i] == -1) {
        // cout << "True label for type " << type << " word " << i << " is missing" << endl;
      } // end if
    } // end word
  } // end type
  return max + 1;
}

class Comparer {
 public:
   double *ref_;
   Comparer(double *ref):ref_(ref) {}
   bool operator()(int a, int b) {
     return ref_[a] < ref_[b];
   }
};

vector<vector<double> > Model::GetKNN() {
  vector<vector<double> > accuracies;
  int K[] = {1, 3, 5};
  int correct_cnt_big[] = {0, 0, 0};
  int total_big = 0;

  for (int type = 0; type < config_.n_entity_types_; ++type) {
    // if we're using mixedness constraint, frequencies are already computed.
    if (!config_.mixedness_constraint_) {
      for (int j = 0; j < config_.vocab_size_[type]; ++j) {
        frequencies_[type][j] = 0.0;
        for (int t = 0; t < config_.n_topics_; ++t) {
          frequencies_[type][j] += counts_topic_words_[type][t][j];
        } // end topic
      } // end word
    } // end if

    int correct_cnt[3] = {0, 0, 0};

    vector<int> idx(config_.vocab_size_[type]);
    for (int i = 0; i < config_.vocab_size_[type]; ++i)
      idx[i] = i;

    double **distances = new double*[config_.vocab_size_[type]];
    for (int i = 0; i < config_.vocab_size_[type]; ++i)
      distances[i] = new double[config_.vocab_size_[type]];

    int skipped = 0;
    for (int i = 0; i < config_.vocab_size_[type]; ++i) {
      distances[i][i] = 1000000000;
      for (int j = i + 1; j < config_.vocab_size_[type]; ++j) {
        distances[i][j] = JSD(type, i, j);
        distances[j][i] = distances[i][j];
      } // end word 2 
      if (true_labels_[type][i] == -1) {
        skipped++;
        continue;
      }

      vector<int> myidx = idx;
      sort(myidx.begin(), myidx.end(), Comparer(distances[i]));

      for (int h = 0; h < 3; ++h) {
        map<int, int> counts;
        int k = 0;
        int ctr = 0;
        while (k < K[h]) {
          if (true_labels_[type][myidx[ctr]] != -1) {
            counts[true_labels_[type][myidx[ctr]]]++;
            ++k;
          }
          ++ctr;
        }
        int winning_label = (std::max_element(counts.begin(), counts.end()))->first;
        
        bool correct = (true_labels_[type][i] == winning_label);
        correct_cnt[h] += correct;
      } // end for
    } // end word 1
    vector<double> type_acc;
    for (int h = 0; h < 3; ++h) {
      double acc = correct_cnt[h] * 1.0 / (config_.vocab_size_[type] - skipped);
      type_acc.push_back(acc);
      correct_cnt_big[h] += correct_cnt[h] * config_.entity_weight_[type];
    }
    accuracies.push_back(type_acc);
    total_big += (config_.vocab_size_[type] - skipped) * config_.entity_weight_[type];

    for (int i = 0; i < config_.vocab_size_[type]; ++i)
      delete[] distances[i];
    delete[] distances;
  } // end type
  vector<double> big_acc;
  for (int h = 0; h < 3; ++h) {
    double acc = correct_cnt_big[h] * 1.0 / total_big;
    big_acc.push_back(acc);
  }
  accuracies.push_back(big_acc);
  return accuracies;
}

// Get KL divergence between topic distributions of words p and q.
double Model::JSD(int type, int p, int q) {
  double distance  = 0.0;
  double f_p = 0;
  double f_q = 0;
  for (int i = 0; i < config_.n_topics_; ++i) {
    double c_pi = counts_topic_words_[type][i][p];
    double c_qi = counts_topic_words_[type][i][q];

    double p_pi = c_pi / frequencies_[type][p];
    double p_qi = c_qi / frequencies_[type][q];

    double component = 0;

    if (c_pi)
      component += p_pi * log(p_pi);
    if (c_qi)
      component += p_qi * log(p_qi);
    if (c_pi + c_qi)
      component -= (p_pi + p_qi) * log((p_pi + p_qi)/2);

    distance += component;
  }
  distance /= 2;
  return distance;
}

