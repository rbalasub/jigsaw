#include <iostream>
#include <sstream>

#include "config.h"
#include "links.h"
#include "model.h"
#include "util.h"

void Links::AddToAverager() {
  double normalizer = config_->n_topics_  * config_->n_topics_ * config_->link_alpha_ + n_links_;
  // normalize to get pi_L for this iteration and save sum to average later.

  for (int i = 0; i < config_->n_topics_; ++i) {
    for (int j = 0; j < config_->n_topics_; ++j) {
      averager_link_topic_pair_counts_[i][j] += 
        (link_topic_pair_counts_[i][j] + config_->link_alpha_) / normalizer;
    } // end topic 2
  } // end topic 1
  averager_count_++;
}

void Links::Average() {
  for (int i = 0; i < config_->n_topics_; ++i) {
    for (int j = 0; j < config_->n_topics_; ++j) {
      averager_link_topic_pair_counts_[i][j] = averager_link_topic_pair_counts_[i][j] / averager_count_;
    } // end topic 2
  } // end topic 1
}

void Links::SaveTopicDistributions(ostream &os) {
  for (int i = 0; i < config_->n_topics_; ++i) {
    for (int j = 0; j < config_->n_topics_; ++j) {
      os << averager_link_topic_pair_counts_[i][j] << '\t';
    } // end topic 2
    os << '\n';
  } // end topic 1
}

void Links::SaveTopics(ostream &os) {
  for (int i = 0; i < n_links_; ++i) {
    os << "2\t" << link_topic_assignments_[i][0] << '\t' << link_topic_assignments_[i][1] << '\n';
  }
}

void Links::RandomInit(int **node_labels) {
  for (int i = 0; i < config_->n_topics_; ++i) {
    for (int j = 0; j < config_->n_topics_; ++j) {
      link_topic_pair_counts_[i][j] = 0;
      averager_link_topic_pair_counts_[i][j] = 0;
    } // end topic 2
  } // end topic 1
  averager_count_ = 0;

  int tot_cnt = 0;
  int assigned_cnt = 0;
  for (int i = 0; i < n_links_; ++i) {
    int t1 = UniformSample(config_->n_topics_);
    int t2 = UniformSample(config_->n_topics_);

    if (config_->use_node_labels_ && node_labels && node_labels[config_->link_attr_[0]][links_[i][0]] != -1) {
      if (Random() >= config_->node_label_randomness_) {
        t1 = node_labels[config_->link_attr_[0]][links_[i][0]];
        assigned_cnt++;
      }

      if (Random() >= config_->node_label_randomness_ && node_labels[config_->link_attr_[1]][links_[i][1]] != -1) {
        t2 = node_labels[config_->link_attr_[1]][links_[i][1]];
        assigned_cnt++;
      }
    }
    tot_cnt += 2;
    link_topic_assignments_[i][0] = t1;
    link_topic_assignments_[i][1] = t2;
    link_topic_pair_counts_[t1][t2]++;
  } // end all links
  // cout << "Of " << tot_cnt << " nodes involved in links " << assigned_cnt << " used assigned labels" << endl;
}

void Links::Allocate() {
  if (n_links_ <= 0)
    return;
  // allocate space for corpus wide topic pair distribution.
  link_topic_assignments_ = new int*[n_links_];
  links_ = new int*[n_links_];
  for (int i = 0; i < n_links_; ++i) {
    link_topic_assignments_[i] = new int[2];
    links_[i] = new int[2];
  }

  link_topic_pair_counts_ = new double*[config_->n_topics_];
  averager_link_topic_pair_counts_ = new double*[config_->n_topics_];
  for (int i = 0; i < config_->n_topics_; ++i) {
    link_topic_pair_counts_[i] = new double[config_->n_topics_];
    averager_link_topic_pair_counts_[i] = new double[config_->n_topics_];
  }
}

void Links::Free() {
  if (n_links_ <= 0 || !config_->model_links_) {
    return;
  }
  for (int i = 0; i < n_links_; ++i) {
    delete[] link_topic_assignments_[i]; 
    delete[] links_[i]; 
  }
  for (int i = 0; i < config_->n_topics_; ++i) {
    delete[] link_topic_pair_counts_[i];
    delete[] averager_link_topic_pair_counts_[i];
  }
  delete[] link_topic_assignments_;
  delete[] links_;
  delete[] averager_link_topic_pair_counts_;
  delete[] link_topic_pair_counts_;
}

void Links::Read(istream &is) {
  string line;
  int link_ctr = 0;

  int t1 = config_->link_attr_[0];
  int t2 = config_->link_attr_[1];

  while (getline(is, line)) {
    istringstream iss(line);
    int id_1, id_2;
    int cnt_unnecessary; 
    // the links file includes a count in the first column although we know it
    // will always be 2. This maintains uniformity with the docs file
    iss >> cnt_unnecessary >> id_1 >> id_2;
    links_[link_ctr][0] = id_1;
    links_[link_ctr][1] = id_2;

    if ((id_1 + 1) > config_->vocab_size_[t1])
      config_->vocab_size_[t1] = id_1 + 1;

    if ((id_2 + 1) > config_->vocab_size_[t2])
      config_->vocab_size_[t2] = id_2 + 1;

    link_ctr++;
  } // end while - reading file.
}

void Links::Save(const string &output_prefix) {
  string link_topics_file      = output_prefix + ".link_topic_pair_labels"; 
  ofstream os(link_topics_file.c_str());
  if (!os) {
    cout << "Cannot save output file with prefix " << output_prefix << endl;
    return;
  }
  SaveTopics(os); 
  os.close();

  string topic_pair_distr_file = output_prefix + ".link_topic_pair_distr";
  ofstream os_2(topic_pair_distr_file.c_str());
  if (!os_2) {
    cout << "Cannot save output file with prefix " << output_prefix << endl;
    return;
  }
  SaveTopicDistributions(os_2);
  os_2.close();
}

void Links::Setup(const string &file) {
  Allocate();

  ifstream ifs(file.c_str());
  if (!ifs) {
    cout << "Cannot read links file " << file << endl;
    exit(0);
  }
  Read(ifs);
  ifs.close();

  //cout << "From Setup" << endl;
  //RandomInit();
}

void Links::CheckIntegrity() {
  double sum = 0;
  for (int t1 = 0; t1 < config_->n_topics_; ++t1) {
    for (int t2 = 0; t2 < config_->n_topics_; ++t2) {
      sum += link_topic_pair_counts_[t1][t2];
    } // end topic 1
  } // end topic 1

  if (sum != n_links_) {
    cout << "Links: Counts not adding up" << endl;
  }
}
