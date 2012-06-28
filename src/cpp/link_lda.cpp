/*******************************************
link_lda.cpp - Implements a Gibbs sampler for a combination of Link LDA, Topics over
Time and PSK with RoleEntropy and BalanceEntropy

Ramnath Balasubramanyan (rbalasub@cs.cmu.edu)
Language Technologies Institute, Carnegie Mellon University
*******************************************/

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include <stdlib.h>
#include <time.h>
#include <libgen.h>
#include <sys/stat.h>

#include "config.h"
#include "corpus.h"
#include "links.h"
#include "model.h"
#include "options.h"
#include "stats.h"
#include "util.h"

using namespace std;

int main(int argc, char **argv) {
  Options opt(argc, argv);
  InitRandomizer(atoi(opt.GetStringValue("randinit", "0").c_str()));
  string config_file = opt.GetStringValue("config_file", "");
  Config config(config_file, opt);
  config.Dump(cout);

  try {
    Stats stats;

    // Create and initialize corpora
    cout << "Reading in documents ... "; cout.flush();
    Corpus corpus(&config,      config.GetNumTrainingDocs(), config.train_file_);
    Corpus test_corpus(&config, config.GetNumTestDocs(),     config.test_file_);
    cout << "done" << endl;

    // Link setup.
    cout << "Reading links ... "; cout.flush();
    Links train_link_corpus(&config, config.GetNumTrainingLinks(), config.link_train_file_);
    Links test_link_corpus(&config,  config.GetNumTestLinks(),     config.link_test_file_);
    cout << "done" << endl;

    //config.DebugDisplay();
    config.Dump();
    if (opt.GetStringValue("check_config", "") == "YES") {
      exit(0);
    }

    char prefix[10000];
    strcpy(prefix, config.output_prefix_.c_str());
    const char *path = dirname(prefix);
    struct stat my_stat;
    if (stat(path, &my_stat) != 0) {
      if (mkdir(path, 0755) != 0) {
        cerr << "Cannot create output directory " << path << endl;
        exit(0);
      } else {
        cout << "Created directory " << path << endl;
      }
    }

    for (int run_num = 0; run_num < config.n_runs_; ++run_num) {
      Model link_lda(config);
      link_lda.Allocate();
      int **node_labels = NULL;
      if (config.use_node_labels_) {
        cout << "Loading labels" << endl;
        link_lda.LoadLabels(config.node_label_file_);
        node_labels = link_lda.input_labels_;
        cout << "Done Loading labels" << endl;
      }

      if (config.GetNumTrainingDocs()) {
        corpus.RandomInit(node_labels);
      }
      if (config.model_links_) {
        train_link_corpus.RandomInit(node_labels);
      }

      Corpus *test_corpus_ptr = config.GetNumTestDocs()  ? &test_corpus:NULL;
      Links  *test_links_ptr  = config.GetNumTestLinks() ? &test_link_corpus:NULL;
      if (test_corpus_ptr)
        test_corpus.RandomInit(node_labels);
      if (test_links_ptr)
        test_link_corpus.RandomInit(node_labels);

      if (config.use_input_topics_)
        link_lda.SampleFromInputTopics(config.input_topic_file_, corpus, &train_link_corpus);
      
      if (config.use_fake_input_topics_)
        link_lda.SampleFromFakeInputTopics(corpus, &train_link_corpus);

      Model *average_model  = link_lda.MCMC(corpus, &train_link_corpus, test_corpus_ptr, test_links_ptr, false);
      ostringstream oss;
      oss << config.output_prefix_ << ".run." << run_num;
      string prefix = oss.str();
      link_lda.Save(prefix + ".model");
      average_model->CalculateAccuracy(&stats);
      cout << endl;
      average_model->Save(prefix + ".avg.model");

      if (config.GetNumTrainingDocs()) {
        cout << "Sampling with average model" << endl;
        average_model->SampleTopics(corpus);
        corpus.Save(prefix + ".train");
        cout << "Done" << endl;
        stats.Save("train_doc_perplexity", average_model->perplexities_, config.n_entity_types_);
        stats.Save("train_avg_doc_theta_entropy", corpus.GetAverageTopicEntropy());
        if (config.model_targets_) {
          corpus.CalculateRealTargetMSE("train", &stats);
        }
      }

      if (config.GetNumTestDocs()) {
        average_model->SampleTopics(test_corpus);
        test_corpus.Save(prefix + ".test");
        stats.Save("test_doc_perplexity", average_model->perplexities_, config.n_entity_types_);
        stats.Save("test_avg_doc_theta_entropy", test_corpus.GetAverageTopicEntropy());
        if (config.model_targets_) {
          test_corpus.CalculateRealTargetMSE("test", &stats);
        }
      }

      if (config.GetNumTrainingLinks() && config.model_links_) {
        average_model->SampleTopicsForLinks(train_link_corpus);
        train_link_corpus.Save(prefix + ".train");
        stats.Save("train_link_perplexity", average_model->link_perplexity_);
      }

      if (config.GetNumTestLinks() && config.model_links_) {
        average_model->SampleTopicsForLinks(test_link_corpus);
        test_link_corpus.Save(prefix + ".test"); 
        stats.Save("test_link_perplexity", average_model->link_perplexity_);
      }

      if (config.mixedness_constraint_) {
        average_model->InitializePenaltyTerms();
        vector<double> average_node_role_entropy(2);
        average_node_role_entropy[0] = average_model->GetAverageNodeRoleEntropy(config.link_attr_[0]);
        average_node_role_entropy[1] = average_model->GetAverageNodeRoleEntropy(config.link_attr_[1]);
        stats.Save("train_avg_node_role_entropy", average_node_role_entropy);
      }
      stats.Save("volume_balance_entropy", link_lda.GetVolumeEntropy());

      average_model->Free();
      delete average_model;
      link_lda.Free();
    } // end runs

    if (config.GetNumTrainingDocs())
      corpus.Free();
    if (config.GetNumTestDocs())
      test_corpus.Free();
    if (config.GetNumTrainingLinks())
      train_link_corpus.Free();
    if (config.GetNumTestLinks())
      test_link_corpus.Free();

    ofstream ofs((config.output_prefix_ + ".report").c_str());
    stats.Dump(ofs);
    config.Dump(ofs);
    ofs.close();
  } catch (std::bad_alloc &exception) {
    cout << "Memory allocation fail! Exiting." << endl;
    return 1;
  } // end catch
  return 0;
}

