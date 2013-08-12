#include <sstream>
#include <cmath>
#include <set>
#include <stdlib.h>

#include "config.h"
#include "corpus.h"
#include "hungarian.h"
#include "model.h"
#include "stats.h"
#include "util.h"

void Corpus::AddWord(int doc, int cur_topic, int type, bool remove) {
  int mult = (remove ? -1 : 1);
  counts_docs_topics_[doc][cur_topic] += mult * config_->entity_weight_[type];
  if (config_->md_n_domains_) {
    int cur_domain = config_->TopicToDomain(cur_topic);
    if (config_->md_seeds_ && cur_topic < config_->md_split_start_indexes_[cur_domain] + 2) {
      // do nothing if it's a senti topic because everything is freshly calculated there.
    } else {
      if (cur_domain == domains_[doc])
        md_entropy_components_[doc][0]->Adjust(cur_topic, mult * config_->entity_weight_[type]);
      if (cur_domain == config_->md_n_domains_)
        md_entropy_components_[doc][1]->Adjust(cur_topic, mult * config_->entity_weight_[type]);
    } // end if seeds
  } // end if multi-domain
}

void Corpus::MakeDomainSampler() {
//  cout << "Entering ... " << endl;
  for (int d = 0; d < config_->md_n_domains_; ++d) {
    int j = 0;
    double sum = 0.0;
    vector<long double> dom_sampler(config_->n_topics_);
    for (int i = 0; i < config_->md_n_domains_ + 1; ++i) {
      for (int k = 0; k < config_->md_splits_[i]; ++k, ++j) {
        if (i == d) {
          // end  in-domain
          dom_sampler[j] = sum + config_->md_probs_[0] / config_->md_splits_[i];
        } else if (i == config_->md_n_domains_) {
          // general domain
          dom_sampler[j] = sum + config_->md_probs_[1] / config_->md_splits_[i];
        } else {
          // out of domain
          dom_sampler[j] = sum + config_->md_probs_[2] / (config_->n_topics_ -  config_->md_splits_[d] - config_->md_splits_[config_->md_n_domains_]);
        } // end if
        sum = dom_sampler[j];
      } // end topics of domain
    } // domain-sub-div
    md_domain_sampling_distr_.push_back(dom_sampler);
  } // end domains
//  cout << md_domain_sampling_distr_.size() << " Exiting ... " << endl;
}

void Corpus::RandomInit(int **node_labels) {
  // Clean out 
  for (int doc = 0; doc < n_docs_; ++doc) {
    weight_[doc] = config_->alpha_ * config_->n_topics_;
    for (int topic = 0; topic < config_->n_topics_; ++topic) {
      counts_docs_topics_[doc][topic] = 0;
      theta_entropy_components_[doc][topic] = 0.0;
    }
  }

  int tot_cnt = 0;
  int assigned_cnt = 0;
  int assigned_to_general = 0;
  int assigned_to_ood = 0;
  int missed_cnt = 0;
  // Assign random topics to words
  for (int type = 0; type < config_->n_entity_types_; ++type) {
    for (int doc = 0; doc < n_docs_; ++doc) {
      for (int word = 0; word < doc_num_words_[type][doc]; ++word) {
        int random_topic = 0;
        if (config_->md_n_domains_) {
          //cout << md_domain_sampling_distr_.size() << md_domain_sampling_distr_[domains_[doc]].size() << endl;
          random_topic = SampleTopicFromMultinomial(md_domain_sampling_distr_[domains_[doc]]);
          if (config_->md_seeds_ && node_labels[type][corpus_words_[type][doc][word]] != -1) {
            if (Random() >= config_->node_label_randomness_) {
              random_topic = node_labels[type][corpus_words_[type][doc][word]] + config_->md_split_start_indexes_[domains_[doc]];
              double r = rand() * 1.0 / RAND_MAX;
              if (r > config_->md_probs_[0] && r <= (config_->md_probs_[0] + config_->md_probs_[1])) {
                // assign to general instead
                random_topic = node_labels[type][corpus_words_[type][doc][word]] + config_->md_split_start_indexes_[config_->md_n_domains_];
                assigned_to_general++;
              } else if (r > (config_->md_probs_[0] + config_->md_probs_[1])) {
                // assign to out of domain instead
                int rand_domain = UniformSample(config_->md_n_domains_);
                random_topic = node_labels[type][corpus_words_[type][doc][word]] + config_->md_split_start_indexes_[rand_domain]; 
                assigned_to_ood++;
              }
              assigned_cnt++;
            } else {
              missed_cnt++;
            }
          } // end if seed word
        } else {
          random_topic = UniformSample(config_->n_topics_);
          if (config_->use_node_labels_ && node_labels && node_labels[type][corpus_words_[type][doc][word]] != -1) {
            if (Random() >= config_->node_label_randomness_) {
              random_topic = node_labels[type][corpus_words_[type][doc][word]];
              assigned_cnt++;
            } else {
              missed_cnt++;
            } // end if using provided label
          } // end if word has provided label
        }
        tot_cnt++;
        word_topic_assignments_[type][doc][word] = random_topic;
        counts_docs_topics_[doc][random_topic] += config_->entity_weight_[type];
        weight_[doc] += config_->entity_weight_[type];
      } // end words
    } // end docs
  } // end type
  InitializeThetaEntropy();
  cout << "Of " << tot_cnt << " words " << assigned_cnt << " used supplied labels and " << missed_cnt << " words slipped by" << endl;
  if (config_->md_n_domains_) {
    cout << "Out of those assigned, " << assigned_to_general << " words were sent to general domain words " << endl;
    cout << "Out of those assigned, " << assigned_to_ood     << " words were sent to out of domain words " << endl;
  }

  if (config_->md_n_domains_) {
    set<int **> jazz;
    for (int d = 0; d < n_docs_; ++d) {
      md_entropy_components_[d][0]->InitEntropy();
      md_entropy_components_[d][1]->InitEntropy();
      //jazz.insert(md_entropy_components_[d][0]->counts_);
      //jazz.insert(md_entropy_components_[d][1]->counts_);
      //jazz.insert(md_senti_entropy_components_[d]->counts_);
    } // end for
    /*
    for (set<int **>::iterator it = jazz.begin(); it != jazz.end(); ++it) {
      cout << "Jazz " << *it << endl;
    } */
    CheckIntegrity();
  } // end if
}

int Corpus::GetMostLikelyTopic(int doc) {
  return std::max_element(counts_docs_topics_[doc], counts_docs_topics_[doc] + config_->n_topics_) - counts_docs_topics_[doc];
}

double Corpus::GetTopicProbability(int doc, int topic, int type) {
  double prob = counts_docs_topics_[doc][topic] + config_->alpha_;
  if (config_->theta_constraint_) {
    // base entropy
    double entropy =  -1.0 * accumulate(theta_entropy_components_[doc], theta_entropy_components_[doc] + config_->n_topics_, 0.0);
    //cout << "doc " << doc << " topic " << topic << " entropy " << entropy << endl;

    double p = (counts_docs_topics_[doc][topic] + config_->entity_weight_[type] + config_->alpha_) * 1.0 / weight_[doc];
    // cout << p * log2(p) << " replacing " << theta_entropy_components_[doc][topic];
    entropy += theta_entropy_components_[doc][topic];
    entropy -= (p * log2(p));

    double penalty_prob = exp (-(entropy * entropy) / (2 * config_->theta_variance_));
    prob *= pow(penalty_prob, config_->theta_penalty_);
    //cout << " new entropy " << entropy << " prob " << penalty_prob << endl;
  }
  if (config_->md_n_domains_) {
    //      cout << "A Blah " << md_entropy_components_[doc][0]->counts_ << ' ';
    //      cout << md_entropy_components_[doc][1]->counts_ << ' ';
    //      cout << md_senti_entropy_components_[doc]->counts_ << endl;

    // modify temporarily
    counts_docs_topics_[doc][topic] += config_->entity_weight_[type];

    int domain = config_->TopicToDomain(topic);
    if (domain == domains_[doc])
      prob *= config_->md_probs_[0] * 10;
    else if (domain == config_->md_n_domains_) 
      prob *= config_->md_probs_[1] * 10;
    else
      prob *= config_->md_probs_[2] * 10;

    if (config_->md_seeds_ && topic < config_->md_split_start_indexes_[domain] + 2) {
      // not checking if senti topic since everything is fresh there.
    } else if (domain == domains_[doc])
      md_entropy_components_[doc][0]->Adjust(topic, config_->entity_weight_[type]);
    else if (domain == config_->md_n_domains_)
      md_entropy_components_[doc][1]->Adjust(topic, config_->entity_weight_[type]);

    if (config_->md_theta_constraint_) {
      double domain_entropy  = md_entropy_components_[doc][0]->Entropy();
      double p1 = exp (-(domain_entropy * domain_entropy) / (2 * config_->md_theta_variance_));
      prob *= pow(p1, config_->md_theta_penalty_);

      double general_entropy = md_entropy_components_[doc][1]->Entropy();
      double p2 = exp (-(general_entropy * general_entropy) / (2 * config_->md_theta_variance_));
      prob *= pow(p2, config_->md_theta_penalty_);

      if (config_->md_seeds_) {
          //  cout << "Blah " << md_entropy_components_[doc][0]->counts_ << ' ';
          //  cout << md_entropy_components_[doc][1]->counts_ << ' ';
          //  cout << md_senti_entropy_components_[doc]->counts_ << endl;
        double senti_entropy   = md_senti_entropy_components_[doc]->Entropy();
        double p3 = exp (-(senti_entropy * senti_entropy) / (2 * config_->md_theta_variance_));
        prob *= pow(p3, config_->md_theta_penalty_);
      }
    }

    //restore 
    counts_docs_topics_[doc][topic] -= config_->entity_weight_[type];
    if (config_->md_seeds_ && topic < config_->md_split_start_indexes_[domain] + 2) {
      // not checking if senti topic since everything is fresh there.
    } else if (domain == domains_[doc])
      md_entropy_components_[doc][0]->Adjust(topic, -1 * config_->entity_weight_[type]);
    else if (domain == config_->md_n_domains_)
      md_entropy_components_[doc][1]->Adjust(topic, -1 * config_->entity_weight_[type]);
  }
  return prob;
}

void Corpus::InitializeThetaEntropy() {
  for (int doc = 0; doc < n_docs_; ++doc) {
    for (int topic = 0; topic < config_->n_topics_; ++topic) {
      double p = (counts_docs_topics_[doc][topic] + config_->alpha_) * 1.0 / weight_[doc];
      theta_entropy_components_[doc][topic] = p * log2(p);
    } // end topic
    //cout << "doc " << doc << accumulate(theta_entropy_components_[doc], theta_entropy_components_[doc] + config_->n_topics_, 0.0) << endl;
  } // end doc
}

Corpus::Corpus(const Config *c, int n, const string &file) : config_(c), n_docs_(n) {
  corpus_labels_ = NULL;
  doc_labels_   = NULL;
  averager_count_ = 0;
  if (n_docs_ <= 0)
      return;
  // cout << "Reading in " << n_docs_ << " test docs" << endl;
  Setup(file);
  if (config_->md_n_domains_) 
    MakeDomainSampler();
}

double Corpus::GetAverageTopicEntropy() {
  double tot = 0.0;
  for (int i = 0; i < n_docs_; ++i) {
    tot +=  -1.0 * accumulate(theta_entropy_components_[i], theta_entropy_components_[i] + config_->n_topics_, 0.0);
  }
  return tot / n_docs_;
}

void Corpus::AddToAverager() {
  for (int i = 0; i < n_docs_; ++i) {
    for (int j = 0; j < config_->n_topics_; ++j) {
      double normalized_ratio = (counts_docs_topics_[i][j] + config_->alpha_) / weight_[i];
      theta_[i][j] = (theta_[i][j] * averager_count_ + normalized_ratio) / (averager_count_ + 1);
    } // end topics
  } // end docs
  averager_count_++;
}

void Corpus::SaveTopicDistributions(ostream &os) {
  if (averager_count_ == 0) {
    cerr << "Cannot save topic distribution of unnormalized corpus" << endl;
    return;
  }
  for (int i = 0; i < n_docs_; ++i) {
    for (int j = 0; j < config_->n_topics_; ++j) {
      os << theta_[i][j] << '\t'; 
    } // end topics
    os << '\n';
  } // end docs
}

void Corpus::SaveTopicLabels(ostream &os) {
  for (int doc = 0; doc < n_docs_; ++doc) {
    for (int type = 0; type < config_->n_entity_types_; ++type) {
      os << doc_num_words_[type][doc] << " ";
      for (int word = 0; word < doc_num_words_[type][doc]; ++word) {
        os << word_topic_assignments_[type][doc][word] << " ";
      } // end words
      os << '\t';
    } // end type
    os << '\n';
  } // end docs
}

void Corpus::SavePredTargets(ostream &os) {
  for (int doc = 0; doc < n_docs_; ++doc) {
    for (int type = 0; type < config_->n_real_targets_; ++type) {
      os << pred_targets_[doc][type] << '\t';
    } // end type
    os << '\n';
  } // end docs
}

void Corpus::Allocate() {
  if (n_docs_ <= 0)
    return;
  corpus_words_ = new int**[config_->n_entity_types_];
  weight_       = new double[n_docs_];
  real_values_  = new double*[n_docs_];
  real_flags_   = new bool*[n_docs_];

  real_targets_      = new double*[n_docs_];
  pred_targets_      = new double*[n_docs_];
  real_target_flags_ = new bool*[n_docs_];

  word_topic_assignments_ = new int**[config_->n_entity_types_];
  // effectively the sum of each row in corpus_words but we ask for it
  // redundantly for efficiency.
  doc_num_words_ = new int*[config_->n_entity_types_];
  for (int i = 0; i < config_->n_entity_types_; ++i) {
    corpus_words_[i] = new int*[n_docs_];
    word_topic_assignments_[i] = new int*[n_docs_];
    doc_num_words_[i] = new int[n_docs_];
  }

  counts_docs_topics_ = new    int*[n_docs_]; //[nDocs][nTopics]
  theta_              = new double*[n_docs_]; //[nDocs][nTopics]
  theta_entropy_components_ = new double*[n_docs_]; //[nDocs][nTopics]
  for (int i = 0; i < n_docs_; ++i) {
    counts_docs_topics_[i] = new    int[config_->n_topics_];
    theta_[i]              = new double[config_->n_topics_];
    theta_entropy_components_[i] = new double[config_->n_topics_];
    for (int j = 0; j < config_->n_topics_; ++j)
      counts_docs_topics_[i][j] = 0;
    if (config_->n_real_valued_attrs_) {
      real_values_[i] = new double[config_->n_real_valued_attrs_];
      real_flags_[i] = new bool[config_->n_real_valued_attrs_];
    }
    if (config_->n_real_targets_) {
      real_targets_[i]      = new double[config_->n_real_targets_];
      pred_targets_[i]      = new double[config_->n_real_targets_];
      real_target_flags_[i] = new bool[config_->n_real_targets_];
    }
  }

  if (config_->md_n_domains_) {
    domains_ = new int[n_docs_];
    Component<int>::config_ = config_;
    SplComponent<int>::config_ = config_;
    md_entropy_components_ = new Component<int>**[n_docs_];
    if (config_->md_seeds_) 
      md_senti_entropy_components_ = new SplComponent<int>*[n_docs_];
    for (int i = 0; i < n_docs_; ++i) {
      md_entropy_components_[i] = new Component<int>*[2];
      md_entropy_components_[i][0] = new Component<int>(config_, counts_docs_topics_, i, 0);
      md_entropy_components_[i][1] = new Component<int>(config_, counts_docs_topics_, i, 0);
      bool debug = false;
      /*if (i == 1) {
        cout << config_ << endl;
        debug = true;
      }*/
      if (config_->md_seeds_) {
        md_senti_entropy_components_[i] = new SplComponent<int>(config_, counts_docs_topics_, i, 0, debug);
        //if (i == 1)
          //cout << "Senti: " << md_senti_entropy_components_[i] << endl;
      }
    } // end for
  }
}

void Corpus::DebugDisplay(ostream &os) {
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

void Corpus::Read(istream &ifs) {
  string line;
  int doc_ctr = 0;
  while (getline(ifs, line)) {
    istringstream iss(line);

    for (int type = 0; type < config_->n_entity_types_; ++type) {
      int word_cnt; // is not always words, could be any other kind of entity.
      iss >> word_cnt;
      doc_num_words_[type][doc_ctr] = word_cnt;
      corpus_words_[type][doc_ctr] = new int[word_cnt];
      word_topic_assignments_[type][doc_ctr] = new int[word_cnt];
      for (int i = 0; i < word_cnt; ++i) {
        int word_id = 0;
        iss >> word_id;
        if ((word_id + 1) > config_->vocab_size_[type])
          config_->vocab_size_[type] = word_id + 1;
        corpus_words_[type][doc_ctr][i] = word_id;
      } // end reading entities
    } // end types

    for (int real_attr = 0; real_attr < config_->n_real_valued_attrs_;
         ++real_attr) {
      string s;
      if (!iss) {
        cout << "Data file does not have the specified number of real attributes\n";
        exit(0);
      }
      iss >> s;
      if (s == "NA" || s == "") {
        real_flags_[doc_ctr][real_attr]  = 0;
        real_values_[doc_ctr][real_attr] = 0;
      } else {
        real_flags_[doc_ctr][real_attr]  = 1;
        real_values_[doc_ctr][real_attr] = atof(s.c_str());
      }
    } // read in real valued attributes

    for (int real_target = 0; real_target < config_->n_real_targets_;
         ++real_target) {
      string s;
      if (!iss) {
        cout << "Data file does not have the specified number of real targets\n";
        exit(0);
      }
      iss >> s;
      if (s == "NA" || s == "") {
        real_target_flags_[doc_ctr][real_target]  = 0;
        real_targets_[doc_ctr][real_target] = 0;
      } else {
        real_target_flags_[doc_ctr][real_target]  = 1;
        real_targets_[doc_ctr][real_target] = atof(s.c_str());
      }
    } // read in real valued targets

    if (config_->md_n_domains_) {
      if (!iss) {
        cout << "Data does not have domains specified"  << endl;
        exit(0);
      }
      iss >> domains_[doc_ctr];
      if (domains_[doc_ctr] < 0 || domains_[doc_ctr] >= config_->md_n_domains_) {
        cout << "Invalid domain " << domains_[doc_ctr] << " in doc " << doc_ctr << endl;
        exit(0);
      }
      md_entropy_components_[doc_ctr][0]->SetDomain(domains_[doc_ctr]);
      md_entropy_components_[doc_ctr][1]->SetDomain(config_->md_n_domains_);
    } // read in known domain of document

    doc_ctr++;
  } // end file
}

void Corpus::Free() {
  if (n_docs_ <= 0)
    return;
  for (int type = 0; type < config_->n_entity_types_; ++type) {
    for (int i = 0; i < n_docs_; ++i) {
      delete[] word_topic_assignments_[type][i];
      delete[] corpus_words_[type][i];
    }
    delete[] word_topic_assignments_[type];
    delete[] corpus_words_[type];
    delete[] doc_num_words_[type];
  }

  for (int i = 0; i < n_docs_; ++i) {
    delete[] counts_docs_topics_[i];
    delete[] theta_[i];
    delete[] theta_entropy_components_[i];
    if (config_->md_n_domains_) {
      delete md_entropy_components_[i][0];
      delete md_entropy_components_[i][1];
      if (config_->md_seeds_)
        delete md_senti_entropy_components_[i];
      delete[] md_entropy_components_[i];
    }

    if (config_->n_real_valued_attrs_) {
      delete[] real_values_[i];
      delete[] real_flags_[i];
    }

    if (config_->n_real_targets_) {
      delete[] real_targets_[i];
      delete[] pred_targets_[i];
      delete[] real_target_flags_[i];
    } 
  } // end for

  if (config_->md_n_domains_) {
    delete[] domains_;
    delete[] md_entropy_components_;
    if (config_->md_seeds_)
      delete[] md_senti_entropy_components_;
  }
  if (corpus_labels_)
    delete[] corpus_labels_;
  corpus_labels_ = NULL;

  if (doc_labels_)
    delete[] doc_labels_;
  doc_labels_ = NULL;

  delete[] weight_;
  delete[] counts_docs_topics_;
  delete[] theta_;
  delete[] theta_entropy_components_;
  delete[] doc_num_words_;
  delete[] word_topic_assignments_;
  delete[] corpus_words_;
  delete[] real_values_;
  delete[] real_flags_;
  delete[] real_targets_;
  delete[] pred_targets_;
  delete[] real_target_flags_;
}

void Corpus::Save(const string &output_prefix) {
  string topic_distr_file = output_prefix + ".doc_topic_distr";
  ofstream os(topic_distr_file.c_str());
  if (!os) {
    cout << "Cannot save files with prefix " << output_prefix << endl;
    return;
  }
  SaveTopicDistributions(os);
  os.close();

  string topic_labels_file = output_prefix + ".doc_topic_labels";
  ofstream os2(topic_labels_file.c_str());
  if (!os2) {
    cout << "Cannot save files with prefix " << output_prefix << endl;
    return;
  }
  SaveTopicLabels(os2);
  os2.close();

  if (config_->model_targets_ && config_->n_real_targets_) {
    string pred_targets_file = output_prefix + ".doc_pred_targets";
    ofstream os(pred_targets_file.c_str());
    if (!os) {
      cout << "Cannot save files with prefix " << output_prefix << endl;
      return;
    }
    SavePredTargets(os);
    os.close();
  }
}

void Corpus::Setup(const string &file) {
  Allocate();

  ifstream ifs(file.c_str());
  if (!ifs) {
    cout << "Cannot read corpus file " << file << endl;
    exit(0);
  }
  Read(ifs);
  ifs.close();

  cout << "Reading doc labels " << endl;
  ReadDocLabels(config_->doc_label_file_, doc_labels_);
  cout << "...done reading labels " << endl;
}

void Corpus::CheckIntegrity() {
  for (int doc = 0; doc < n_docs_; ++doc) {
    if (config_->md_n_domains_) {
      if (doc != md_entropy_components_[doc][0]->sel_id_) {
        cout << "Incorrect id in doc " << doc  << doc << " != " << md_entropy_components_[doc][1]->sel_id_ << endl;
      } else  if (doc != md_entropy_components_[doc][1]->sel_id_) {
        cout << "Incorrect id in doc -- general component " << doc << " != " << md_entropy_components_[doc][1]->sel_id_ << endl;
      }
      double t = 0;
      for (int k = config_->md_split_start_indexes_[domains_[doc]] + (config_->md_seeds_?2:0); k < config_->md_split_start_indexes_[domains_[doc]] + config_->md_splits_[domains_[doc]]; ++k) {
        t += md_entropy_components_[doc][0]->counts_[doc][k];
      }
      if (t != md_entropy_components_[doc][0]->total_){
        cout << "Total not correct in domain component in doc " << doc << endl;
      }
      t = 0;
      for (int k = config_->md_split_start_indexes_[config_->md_n_domains_] + (config_->md_seeds_?2:0); k < config_->n_topics_; ++k) {
        t += md_entropy_components_[doc][0]->counts_[doc][k];
      }
      if (t != md_entropy_components_[doc][1]->total_){
        cout << "Total not correct in general component in doc " << doc << endl;
      }
    }
    double corpus_count = 0;
    for (int type = 0; type < config_->n_entity_types_; ++type) {
      corpus_count += doc_num_words_[type][doc] * config_->entity_weight_[type];
      for (int word = 0; word < doc_num_words_[type][doc]; ++word) {
        if (word_topic_assignments_[type][doc][word] < 0 || word_topic_assignments_[type][doc][word] > config_->n_topics_) {
          cout << "Weird topic assigment " << doc << " doc " << " word id " << word << " topic " << word_topic_assignments_[type][doc][word] << endl;
        }
      }
    } // end type

    double topic_weight_count = 0;
    double theta_h = 0.0;
    for (int topic = 0; topic < config_->n_topics_; ++topic) {
      topic_weight_count += counts_docs_topics_[doc][topic];
      double p = (counts_docs_topics_[doc][topic] + config_->alpha_) / weight_[doc];
      theta_h += p * log2(p);
    } // end topic
    if (config_->theta_constraint_) {
      if (abs(theta_h - accumulate(theta_entropy_components_[doc], theta_entropy_components_[doc] + config_->n_topics_, 0.0)) > 0.00001) {
        cout << "Bug in theta entropy ... should be " << theta_h << " is " << accumulate(theta_entropy_components_[doc], theta_entropy_components_[doc] + config_->n_topics_, 0.0) << endl;
      }
    } // end if checking for theta entropy integrity

    if (topic_weight_count != corpus_count) {
      cout << "In corpus: Weights not matching up in doc " << doc << " " << topic_weight_count << " " << corpus_count << " " << weight_[doc] << endl;
    }
    if ((topic_weight_count + config_->n_topics_ * config_->alpha_) != weight_[doc]) {
      cout << "In corpus: Weights not matching up to weight " << doc << endl;
    }
  } // end doc
}

void Corpus::CalculateRealTargetMSE(string prefix, Stats *stats) {

  vector<double> mse(config_->n_real_targets_);
  vector<double> accuracy(config_->n_real_targets_);
  vector<int> counts(config_->n_real_targets_);
  if (!config_->model_targets_)
    return;

  int nan_skips = 0;
  for (int doc = 0; doc < n_docs_; ++doc) {
    for (int target = 0; target < config_->n_real_targets_; ++target) {
      if (real_target_flags_[doc][target]) {
        if (isnan(real_targets_[doc][target]) || isnan(pred_targets_[doc][target])) {
          nan_skips++;
          continue;
        }
        counts[target]++;
        mse[target] += pow(real_targets_[doc][target] - pred_targets_[doc][target], 2.0);
        bool real_flag = real_targets_[doc][target] < 0;
        bool pred_flag = pred_targets_[doc][target] < 0;
        accuracy[target] += ((real_flag ^ pred_flag) == 0);
      }
    } // end target
  } // end doc

  if (nan_skips)
    cout << "Skipping over " << nan_skips << " predictions while calculating MSE" << endl;

  for (int target = 0; target < config_->n_real_targets_; ++target) {
    mse[target]       /= counts[target];
    accuracy[target]  /= counts[target];

    ostringstream oss;
    oss << prefix << "_target_" << target << "_mse";
    stats->Save(oss.str(), mse[target]);

    ostringstream oss2;
    oss2 << prefix << "_target_" << target << "_accuracy";
    stats->Save(oss2.str(), accuracy[target]);

    ostringstream oss3;
    oss3 << prefix << "_target_" << target << "_eval_over";
    stats->Save(oss3.str(), counts[target]);
  }
}

void Corpus::ReadDocLabels(const string &label_file, int* &labels) {
  if (labels)
    return;

  labels = new int[n_docs_];
  fill(labels, labels + n_docs_, -1);

  ifstream ifs(label_file.c_str());
  if (!ifs) {
    cout << "Cannot open doc label file " << label_file << endl;
    labels = NULL;
    return;
  } else {
    cout << "Reading doc labels from " << label_file << endl;
  }

  string line;
  int ctr = 0;
  while (getline(ifs, line)) {
    istringstream iss(line);
    int id, label;
    iss >> id >> label;
    if (id >= 0 && id < n_docs_ && label >= 0) {
      labels[id] = label;
    } else {
      cout << "Discarding doc label " << label << " for docid " << id << endl;
    }
    ctr++;
  } // end while - reading in file.
  ifs.close();
  cout << "Read " << ctr << " document labels" << endl;
}

double Corpus::GetAccuracyFromHungarian() {
  double accuracy = -1.0;

  if (corpus_labels_ == NULL) {
    if (config_->train_docs_label_file_ == "") {
      cout << "No true label file provided so exiting hungarian accuracy module" << endl;
      return accuracy;
    }
    ReadDocLabels(config_->train_docs_label_file_, corpus_labels_);
    if (!corpus_labels_) {
      cout << "Invalid true label file" << endl;
      return accuracy;
    }
  }

  hungarian_problem_t problem;
  int n_true_classes = GetNumTrueDocClasses();
  int **cost_matrix = new int*[n_true_classes]; 
  cout << "True Classes " << n_true_classes << endl;
  for (int t = 0; t < n_true_classes; ++t)  {
    cost_matrix[t] = new int[config_->n_topics_];
    for (int c = 0; c < config_->n_topics_; ++c)
      cost_matrix[t][c] = 0;
  } // end for

  for (int i = 0; i < n_docs_; ++i) {
    int pred_label = GetMostLikelyTopic(i);
    int true_label = corpus_labels_[i];
    if (true_label == -1) 
        continue;

    for (int c = 0; c < config_->n_topics_; ++c) {
      // For the true class, add a penalty to match it with any topic that's not the pred label.
      if (c != true_label) 
        cost_matrix[pred_label][c]++;
    }
  } // end doc

  hungarian_init(&problem, cost_matrix, n_true_classes, config_->n_topics_, HUNGARIAN_MODE_MINIMIZE_COST);
  // hungarian_print_costmatrix(&problem);
  hungarian_solve(&problem);
  // hungarian_print_assignment(&problem);

  // get assignment
  vector<int> matched_class(n_true_classes);
  for (int i = 0; i < n_true_classes; ++i) {
    bool done = false;
    for (int j = 0; j < config_->n_topics_; ++j) {
      if (problem.assignment[i][j] == 1) {
        matched_class[i] = j;
        done = true;
  //      cout << "Node " << i << " == Class " << j << endl;
        break;
      } // end if
    } // end class
    if (!done)
        cout << "Node " << i << " unmatche " << endl;
  } // end class

  hungarian_free(&problem);

  // get predicted labels and compare
  int total = 0;
  int correct = 0;

  for (int i = 0; i < n_docs_; ++i) {
    int pred_topic = GetMostLikelyTopic(i);
    int true_label = corpus_labels_[i];

    int pred_label = matched_class[pred_topic];
    if (pred_label == true_label)
      correct++;
    total++;
  } // end word
  accuracy = correct * 1.0 / total;

  for (int i = 0; i < n_true_classes; ++i)
    delete[] cost_matrix[i];
  delete[] cost_matrix; 
  return accuracy;
}

double Corpus::GetKNN() {
}

double Corpus::GetNMI() {
  double nmi = -1.0;

  if (corpus_labels_ == NULL) {
    if (config_->train_docs_label_file_ == "") {
      cout << "No true doc label file provided so exiting hungarian accuracy module" << endl;
      return nmi;
    }
    ReadDocLabels(config_->train_docs_label_file_, corpus_labels_);
    cout << "Boom!" << endl;
  }

  int n_true_classes    = GetNumTrueDocClasses();
  double *pred_distr       = new double[config_->n_topics_];
  double *true_distr       = new double[n_true_classes];
  double **contingency     = new double*[config_->n_topics_]; 
  for (int t = 0; t < config_->n_topics_; ++t) {
    contingency[t]     = new double[n_true_classes];
  }

  for (int t = 0; t < config_->n_topics_; ++t)  {
    pred_distr[t] = 0;
    for (int c = 0; c < n_true_classes; ++c) {
      contingency[t][c] = 0;
      if (t == 0)
        true_distr[c] = 0;
    } // end classes
  } // end topics
///
  double sum_pred_distr = 0.0;
  for (int i = 0; i < n_docs_; ++i) {
    int true_label = corpus_labels_[i];
    if (true_label == -1)
      continue;

    double norm = accumulate(counts_docs_topics_[i], counts_docs_topics_[i] + config_->n_topics_, 0.0) * 1.0;
    for (int topic = 0; topic < config_->n_topics_; ++topic) {
      pred_distr[topic] += counts_docs_topics_[i][topic] * 1.0 / norm;
      contingency[topic][true_label] += counts_docs_topics_[i][topic] / norm;
    }
    sum_pred_distr++;

    if (true_label >= n_true_classes) {
      cout << "Weird true class " << true_label << " for doc " << i << endl;
    }
    true_distr[true_label]++;
  } // end docs

  
  double h_cond = 0.0;
  for (int i = 0; i < config_->n_topics_; ++i) {
    if (pred_distr[i] > 0)
      h_cond += (pred_distr[i] * GetEntropy(contingency[i], n_true_classes));
    //cout << endl;
    //for (int c = 0; c < n_true_classes; ++c)
    //  cout << contingency[i][c] << ' ';
    //cout << endl;
  } // end class
  h_cond /= sum_pred_distr;
  double h_true = GetEntropy(true_distr, n_true_classes);
  double h_pred = GetEntropy(pred_distr, config_->n_topics_);
  nmi = 2 * (h_true - h_cond) / (h_true + h_pred);

  for (int i = 0; i < config_->n_topics_; ++i) {
    delete[] contingency[i];
  }
  delete[] contingency; 
  delete[] pred_distr;
  delete[] true_distr;
  cout << "NMI done " << endl;
  return nmi;
}

double Corpus::GetEntropy(double *distr, int n) {
  double h = 0.0;
  double sum = accumulate(distr, distr + n, 0.0) * 1.0;
  for (int i = 0; i < n; ++i) {
    if (distr[i] > 0)
      h -= (distr[i] / sum) * log2(distr[i] / sum);
  } // end for
  return h;
}

int Corpus::GetNumTrueDocClasses() {
  return *(std::max_element(corpus_labels_, corpus_labels_ + n_docs_)) + 1;
}

/*class Comparer {
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
}*/

void Corpus::CalculateAccuracy(Stats *stats) {
  if (config_->docs_nmi_flag_) {
    double nmi = GetNMI();
    ostringstream oss_p;
    oss_p << " DocNMI:" << setprecision(4) << nmi;
    if (stats) {
      stats->Save("doc_label_prediction_nmi", nmi);
    }
    if (!stats)
      cout << "  " << oss_p.str();
  } // end if
  if (config_->docs_hungarian_flag_) {
    double acc = GetAccuracyFromHungarian();
    ostringstream oss_p;
    oss_p << " DocHungarianAcc:" << setprecision(4) << acc;
    if (stats) {
      stats->Save("doc_label_prediction_accuracy", acc);
    }
    if (!stats)
      cout << "  " << oss_p.str();
  } // end if
}


