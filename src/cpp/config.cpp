#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <unistd.h>

#include "config.h"
#include "options.h"
#include "util.h"

void Config::Help() {
  cout << "All Recognized options \n";
  cout << "----------------------\n\n";
  for (vector<string>::iterator i = order_.begin(); i != order_.end(); ++i) {
    cout << setw(max_option_length_ + 5) << left;
    cout << "--" + *i << "   " << option_desc_[*i] << endl;
  }
  cout << endl;
}

void Config::Dump(ostream &os) {
  ostringstream voc_str, ent_str, beta_str;
  for (int i = 0; i < n_entity_types_; ++i) {
    voc_str << vocab_size_[i];
    ent_str << entity_weight_[i];
    beta_str << beta_[i];
    if (i != n_entity_types_ - 1) {
      voc_str << ',';
      ent_str << ',';
      beta_str << ',';
    }
  }
  ostringstream md_split_str;
  if (md_n_domains_)
    for (int i = 0; i < md_n_domains_ + 1; ++i) {
      md_split_str << md_splits_[i];
      if (i != md_n_domains_) {
        md_split_str << ',';
      }
    }
  md_split_str << "(*)";
  ostringstream md_prob_str;
  md_prob_str << md_probs_[0] << ',' << md_probs_[1] << ',' << md_probs_[2] << "(*)";
  all_options_["md_splits"] = md_split_str.str();
  all_options_["md_probs"] = md_prob_str.str();

  ostringstream link_attr_str;
  link_attr_str << link_attr_[0] << ',' << link_attr_[1];
  all_options_["beta"] = beta_str.str();
  all_options_["vocab_sizes"] = voc_str.str();
  all_options_["entity_weights"] = ent_str.str();
  all_options_["link_attrs"] = link_attr_str.str();

  MakePathAbsolute(all_options_["output_prefix"]);
  MakePathAbsolute(all_options_["input_model_file"]);
  MakePathAbsolute(all_options_["train_file"]);
  MakePathAbsolute(all_options_["test_file"]);
  MakePathAbsolute(all_options_["link_test_file"]);
  MakePathAbsolute(all_options_["link_train_file"]);
  MakePathAbsolute(all_options_["node_label_file"]);
  MakePathAbsolute(all_options_["true_label_file"]);
  for (vector<string>::iterator i = order_.begin(); i != order_.end(); ++i)
    if (all_options_.count(*i))
      os << *i << '=' << all_options_[*i]<< endl;
}

void Config::DebugDisplay(ostream &os) {
  os << "Runs: " << n_runs_ << endl;
  os << "Iterations: " << n_iterations_ << endl;
  os << "Sample Iterations: " << n_sample_iterations_ << endl;
  os << "Averaged over samples: " << n_avg_ << endl;
  os << "Topics: " << n_topics_ << endl;
  os << endl;
  os << "Types: " << n_entity_types_ << endl;
  for (int i = 0; i < n_entity_types_; ++i) {
    os << "Vocab = " << vocab_size_[i] << " ";
    os << "Weight = " << entity_weight_[i] << " ";
    os << "Beta = " << beta_[i] << " ";
    os << endl;
  }
  os << "Alpha: " << alpha_ << endl;
  os << "Link alpha: " << link_alpha_ << endl;
  os << endl;

  os << "Real Attrs: " << n_real_valued_attrs_ << endl;
  os << "Model real: " << model_real_ << endl;
  os << "Lit weight: " << lit_weight_ << endl;
  os << "Link weight: " << link_weight_ << endl;
  os << "Model links: " << model_links_ << " " << link_attr_[0] << " -> " << link_attr_[1] << " Diag discount: " << off_diagonal_discount_ << endl;
  os << "Model targets: " << model_targets_ << " #targets: " << n_real_targets_ << endl;
  os << endl;
  os << "Output Prefix = " << output_prefix_ << endl;

  os << "Train file " << train_file_ <<  " " << n_docs_ << endl;
  os << "Test file " << test_file_ <<  " " << n_test_docs_ << endl;
  os << "Link Train file " << link_train_file_ <<  " " << n_train_links_ << endl;
  os << "Link Test file " << link_test_file_ <<  " " << n_test_links_ << endl;
  os << endl;

  os << "Mixedness " << mixedness_constraint_ << " Variance " << mixedness_variance_ << " Weight " << mixedness_penalty_ << endl;
  os << "Balance reg " << balance_constraint_ << " Variance " << balance_variance_ << " Weight " << balance_penalty_ << endl;
  os << "Theta constraint " << theta_constraint_ << " Variance " << theta_variance_ << " Weight " << theta_penalty_ << endl;
}

Config::~Config() {
  delete[] beta_;
  delete[] vocab_size_;
  delete[] entity_weight_;
  if (md_n_domains_ > 0) {
    delete[] md_splits_;
    delete[] md_split_start_indexes_;
  }
}

double ReadFloatLine(istream &ifs) {
  double val;
  string config_line;
  getline(ifs, config_line);
  istringstream iss(config_line);
  iss >> val;
  return val;
}

int ReadLine(istream &ifs) {
  int val;
  string config_line;
  getline(ifs, config_line);
  istringstream iss(config_line);
  iss >> val;
  return val;
}

void Config::ReadConfigMap(ifstream &ifs) {
  string config_line;
  string name, val;
  while (ifs) {
    getline(ifs, config_line);
    istringstream iss(config_line);
    getline(iss, name, '=');
    getline(iss, val);
    file_options_[name] = val;
  }
}

string Config::GetConfigValue(const string &key, const string &def, const string &desc) {
  option_desc_[key] = desc;
  order_.push_back(key);
  if (key.size() > max_option_length_)
    max_option_length_ = key.size();
  string cmd_option = options_.GetStringValue(key, "NOCMDVALUE");
  if (cmd_option != "NOCMDVALUE") {
    all_options_[key] = cmd_option;
    return cmd_option;
  }
  string value = file_options_.count(key) ? file_options_[key] : def;
  all_options_[key] = value;
  return value;
}

template<class T>
void SplitString(string &str, T *vec, int max, T def) {
  int cnt = 0;
  if (str.length()) {
    istringstream iss(str);
    while (iss && cnt < max) {
      string next_val;
      getline(iss, next_val, ',');
      vec[cnt] = atoi(next_val.c_str());
      cnt++;
    }
  }
  while (cnt < max) {
    vec[cnt++] = def;
  }
}

void Config::SetConfigValues() {

  string def;

  string corpus_prefix   = GetConfigValue("corpus_prefix",   "", "Prefix for docs and links train and test files. Easy way to specify all of them in one shot.");

  def = ((corpus_prefix == "") ? "" : corpus_prefix + ".train.docs");
  train_file_            = GetConfigValue("train_file", def, "Training document corpus.");

  def = ((corpus_prefix == "") ? "" : corpus_prefix + ".test.docs");
  test_file_             = GetConfigValue("test_file",  def, "Test document corpus.");

  def = ((corpus_prefix == "") ? "" : corpus_prefix + ".train.links");
  link_train_file_       = GetConfigValue("link_train_file", def, "Training network dataset.");

  def = ((corpus_prefix == "") ? "" : corpus_prefix + ".test.links");
  link_test_file_        = GetConfigValue("link_test_file", def, "Test network dataset.");

  n_docs_                = (train_file_ != "") ? GetNumLinesInFile(train_file_) : 0;
  n_test_docs_           = (test_file_  != "")  ? GetNumLinesInFile(test_file_)  : 0;
  n_train_links_         = (link_train_file_ != "") ? GetNumLinesInFile(link_train_file_) : 0;

  (n_train_links_ != 0) ? def = "1" : def = "0";
  model_links_           = atoi(GetConfigValue("model_links", def, "Flag to add network modeling.").c_str());

  n_real_targets_        = atoi(GetConfigValue("num_real_targets", "0", "#sLDA style targets.").c_str());
  (n_real_targets_ != 0) ? def = "YES" : def = "NO";
  model_targets_         = (GetConfigValue("model_targets", def, "Flag to additionally model sLDA style targets for documents.") == "YES");
  

  if (n_train_links_ == 0)
    model_links_ = 0;
  n_test_links_          = (link_test_file_ != "") ? GetNumLinesInFile(link_test_file_) : 0;

  n_real_valued_attrs_   = atoi(GetConfigValue("num_real_valued_attrs", "0", "#TOT style targets.").c_str());
  n_entity_types_        = atoi(GetConfigValue("num_entity_types", "1", "#fields in documents.").c_str());
  string link_attrs      = GetConfigValue("link_attrs", "0,0", "Which attributes are linked by edges.");
  SplitString(link_attrs, link_attr_, 2, 0);

  vocab_size_            = new int[n_entity_types_];
  string vocab_size      = GetConfigValue("vocab_sizes", "", "Manually specify vocabulary sizes. Must be larger than that implied from dataset or bad things will happen.").c_str();
  SplitString(vocab_size, vocab_size_, n_entity_types_, 0);

  n_topics_              = atoi(GetConfigValue("topics", "10", "Number of topics.").c_str());
  output_prefix_         = GetConfigValue("output_prefix",   "link_lda", "Output prefix for saved models and other output files.");
  n_iterations_          = atoi(GetConfigValue("iterations", "50", "Number of Gibbs sampling epochs.").c_str());
  n_sample_iterations_   = atoi(GetConfigValue("sample_iterations", "10", "Number of epochs for unseen docs/networks.").c_str());
  n_avg_                 = atoi(GetConfigValue("average_over", "10", "Number of samples to take after mixing.").c_str());
  n_runs_                = atoi(GetConfigValue("runs", "1", "Number of runs with random starting points.").c_str());

  alpha_                 = atof(GetConfigValue("alpha", "1.0", "Sym Dirichlet prior for doc topic distributions.").c_str());
  link_alpha_            = atof(GetConfigValue("link_alpha", "1.0", "Sym Dirichlet prior for network topic-pair distributions.").c_str());
  string beta            = GetConfigValue("beta", "", "Sym Dirichlet prior for topic distributions.").c_str();
  beta_                  = new double[n_entity_types_];
  SplitString(beta, beta_, n_entity_types_, 1.0);

  string entity_weights  = GetConfigValue("entity_weights", "", "Weights for attributes.");
  entity_weight_         = new int[n_entity_types_];
  SplitString(entity_weights, entity_weight_, n_entity_types_, 1);

  model_real_            = static_cast<RealDistr>(atoi(GetConfigValue("model_real", "0", "Flag to additionally model real attributes.").c_str()));
  real_weight_           = atoi(GetConfigValue("realweight", "1", "Weight of real attributes.").c_str());
  off_diagonal_discount_ = atoi(GetConfigValue("diagdiscount", "4", "Penalty in prior for off-diagonal topic pairs.").c_str());
  link_weight_           = atoi(GetConfigValue("linkweight", "1", "Relative weight of links.").c_str());
  lit_weight_            = atoi(GetConfigValue("litweight", "1", "Relative weight of documents.").c_str());

  mixedness_constraint_  = atoi(GetConfigValue("mixed_penalty", "0", "Enable/disable RoleEntropy.").c_str());
  mixedness_penalty_     = atoi(GetConfigValue("mixed_penalty_weight", "1", "Weight of RoleEntropy regularization.").c_str());
  mixedness_variance_    = atof(GetConfigValue("mixed_variance", "1.0", "RoleEntropy variance.").c_str());

  balance_constraint_    = atoi(GetConfigValue("balance_penalty", "0", "Enable/disable BalanceEntropy.").c_str());
  balance_penalty_       = atoi(GetConfigValue("balance_penalty_weight", "1", "Weight of BalanceEntropy regularization.").c_str());
  balance_variance_      = atof(GetConfigValue("balance_variance", "1.0", "BalanceEntropy variance.").c_str());

  theta_constraint_      = atoi(GetConfigValue("theta_penalty", "0", "Enable/disable ThetaEntropy.").c_str());
  theta_penalty_         = atoi(GetConfigValue("theta_penalty_weight", "1", "Weight of ThetaEntropy regularization.").c_str());
  theta_variance_        = atof(GetConfigValue("theta_variance", "1.0", "ThetaEntropy variance.").c_str());

  volume_constraint_     = atoi(GetConfigValue("volume_penalty", "0", "Enable/disable VolumeEntropy.").c_str());
  volume_penalty_        = atoi(GetConfigValue("volume_penalty_weight", "1", "Weight of VolumeEntropy regularization.").c_str());
  volume_variance_       = atof(GetConfigValue("volume_variance", "1.0", "VolumeEntropy variance.").c_str());

  md_theta_constraint_   = atoi(GetConfigValue("md_theta_penalty", "0", "Enable/disable MultiDomainThetaEntropy.").c_str());
  md_theta_penalty_      = atoi(GetConfigValue("md_theta_penalty_weight", "1", "Weight of MultiDomainThetaEntropy regularization.").c_str());
  md_theta_variance_     = atof(GetConfigValue("md_theta_variance", "1.0", "MultiDomainThetaEntropy variance.").c_str());

  md_mixed_constraint_   = atoi(GetConfigValue("md_mixed_penalty", "0", "Enable/disable MultiDomainRoleEntropy.").c_str());
  md_mixed_penalty_      = atoi(GetConfigValue("md_mixed_penalty_weight", "1", "Weight of MultiDomainRoleEntropy regularization.").c_str());
  md_mixed_variance_     = atof(GetConfigValue("md_mixed_variance", "1.0", "MultiDomainRoleEntropy variance.").c_str());

  md_n_domains_          = atoi(GetConfigValue("md_num_domains", "0", "Number of domains.").c_str());
  md_splits_string_      = GetConfigValue("md_splits", "", "Multi domain splits.");
  md_probs_string_       = GetConfigValue("md_probs", "0.8,0.18,0.02", "Multi domain probabilities: in-domain, general, out-domain.");
  if (md_n_domains_) {
    md_splits_               = new int[md_n_domains_ + 1];
    md_split_start_indexes_  = new int[md_n_domains_ + 1];
    if (SplitIntoArray(md_splits_string_, ',', md_splits_, "int") != md_n_domains_ + 1) {
      cout << "Number of splits must be number of domains plus one" << endl;
      exit(0);
    }
    int sum_domains = 0;
    for (int x = 0; x < md_n_domains_ + 1; ++x) {
      md_split_start_indexes_[x] = sum_domains;
      sum_domains += md_splits_[x];
    } // end for

    if (sum_domains != n_topics_) {
      cout << "Split sums must add up to topics" << endl;
      exit(0);
    }
    SplitIntoArray(md_probs_string_, ',', md_probs_, "double");
  } // end if

  node_label_randomness_ = atof(GetConfigValue("label_randomness", "0.2", "Randomness in node label initialization when node labels supplied.").c_str());
  clamp_rigidity_        = atof(GetConfigValue("clamp_rigidity", "0.8", "Rigidity of belief in clamped node labels.").c_str());
  node_label_file_       = GetConfigValue("node_label_file", "", "Input node labels.");
  use_node_labels_       = (node_label_file_ != "");
  md_seeds_              = (md_n_domains_ > 0 && use_node_labels_);

  input_topic_file_      = GetConfigValue("input_model_file", "", "Input model file to start from instead of a random point.");
  use_input_topics_      = (input_topic_file_ != "");
  use_fake_input_topics_ = atoi(GetConfigValue("use_fake_topics", "0", "Simulate input models by using random topics.").c_str());

  true_label_file_       = GetConfigValue("true_label_file", "", "Known node label file for evaluation.");
  def = (true_label_file_ == "") ? "YES" : "NO";
  hungarian_flag_        = (GetConfigValue("disable_hungarian", def, "Disable best align accuracy evaluation.") != "YES");
  nmi_flag_              = (GetConfigValue("disable_nmi",       def, "Disable NMI evaluation.") != "YES");
  knn_flag_              = (GetConfigValue("disable_knn",       def, "Disable KNN evaluation.") != "YES");
  
  check_integrity_       = (GetConfigValue("check_integrity", "NO", "For debugging: see if all data structures are healthy.") == "YES");
  fast_lda_              = (GetConfigValue("fast_lda", "NO", "Use fast LDA for faster inference in networks") == "YES");
}

void Config::CheckOptions() {
  if (train_file_ == "" && (!model_links_ || link_train_file_ == "")) {
    std::cout << "No documents OR links to model. I am done." << endl;
    exit(0);
  }
  if (n_docs_ == 0 && model_links_ == 0) {
    std::cout << "No documents OR links to model. I am done." << endl;
    exit(0);
  }
  if (model_real_ && !n_real_valued_attrs_) {
    model_real_ = NONE;
    cout << "model_real specified but there are no real attrs in data" << endl;
  }
  options_.CheckUserFlags();
}

Config::Config(string config_file, Options &opt) : options_(opt) {
  max_option_length_ = 0;
  option_desc_["help"] = "Display this help message and quit.";
  option_desc_["check_config"] = "Print config and exit without starting inference. Useful to double-check if everything is OK before a time-consuming run.";
  option_desc_["randinit"] = "If -1, use time to seed randomizer, if 0 no seeding, else use specified value as seed.";
  option_desc_["config_file"] = "Config file with options in a file to avoid entering options on the command line repeatedly.";
  order_.push_back("help");
  order_.push_back("check_config");
  order_.push_back("randinit");
  order_.push_back("config_file");

  if (config_file != "") {
    std::ifstream ifs(config_file.c_str());
    if (!ifs) {
      cout << "Cannot open config file - " << config_file << endl;
      exit(0);
    }
    ReadConfigMap(ifs);
    ifs.close();
  }
  // priority order - cmd line, config file, default value
  SetConfigValues();
  if (options_.GetStringValue("help", "") == "YES") {
    Help();
    exit(0);
  }
  options_.GetStringValue("check_config", ""); // to register this option
  CheckOptions();
  md_probs_[0] = 0.80;
  md_probs_[1] = 0.17;
  md_probs_[2] = 0.03;
}

