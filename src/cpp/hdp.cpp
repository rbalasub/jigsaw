/*******************************************
np_link_lda.cpp - Implements a Gibbs sampler for a non parametric version of  Link LDA, Topics over
Time and Stochastic block model

Ramnath Balasubramanyan (rbalasub@cs.cmu.edu)
Language Technologies Institute, Carnegie Mellon University
*******************************************/

/***
Restaurant - document
Tables     - topic realized in document
Franchise  - global big object that contains everything
Dish       - global topic
Chain      - corpus
***/
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <functional>
#include <list>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <stdlib.h>
#include "gsl/gsl_randist.h"

using namespace std;

enum RealDistr{NONE, GAUSSIAN, BETA};


template <class T, class Y>
void GetIterator(T &lst, int n, Y &return_val) {
  return_val = lst.begin();
  for (int ctr = 0; ctr < n; ++ctr) {
    ++return_val;
    if (return_val == lst.end()) {
      cerr << "Whoa";
    }
  } // end for
}

class Options {
 public:
  map<string, string> user_flags_;

  Options(int argc, char **argv) { ParseCommandLine(argc, argv); }
  void ParseCommandLine(int, char **);
  string GetStringValue(const string&, const string&);
  int GetIntValue(const string&, int);
  double GetFloatValue(const string&, double);

  void DebugDisplay(ostream &os);
};

class Shoutout {
 public:
  ostream &os_;
  ofstream dummy_;
  int limit_;
  Shoutout(ostream &o, int l = 50): os_(o), limit_(l), dummy_("/dev/null") {}

  int vol_;
  // Setting limit to 0 will amount to silent mode.
  // 10 - Normal messages
  // 20 - verbose info
  // 30 - debug info
  void SetLimit(int t) { limit_ = t;}
  ostream& operator()(int vol = 10) {
    if (vol <= limit_)
      return os_;
    else 
      return dummy_;
  }
};

Shoutout sout(cout);
Shoutout serr(cerr);

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

double Options::GetFloatValue(const string &key, double default_value = -1) {
  string val;
  if ((val = user_flags_[key]) == "") {
    if (default_value != -1) {
      return default_value;
    } else {
      cerr << "Please supply " << key << endl;
      exit(1);
    }
  } else {
    return atof(val.c_str());
  }
}

// Get a sample from Unif([0...max-1])
int UniformSample(int max) {
  return static_cast<int>((rand() * 1.0 / RAND_MAX) * max);
}

// Sample from a cdf of a multinomial.
// returns from range [0 ... n-1]
template <class T>
int MultSample(T &cdf, int n) {
  long double unif_sample = rand() * 1.0 / RAND_MAX;
  
  //sout(40) << "Sampling "<< unif_sample << "n = " << n << '\n' ;
  int sample_id = -1;
  for (int sample = 0; sample < n; ++sample) {
    if (unif_sample < (cdf[sample] / cdf[n - 1])) {
      sample_id = sample;
      break;
    }
  } // end iterating over cdf
  //sout(40) << ") ---> Selected " << sample_id << endl;

  /*
  if (sample_id == -1) {
    for (int sample = 0; sample < n; ++sample) {
      cout << cdf[sample] << '\n';
    }
    exit(0);
  }
  */

  return sample_id;
}

class Franchise;

class Config {
 public:

  void ReadConfig(const string&);

  int n_iterations_;
  int n_entity_types_;      // num of types of entities
                            // like words, proteins, authors etc.

  int *vocab_size_;         // size of vocab for each type of entity.
  int *entity_weight_;      // relative weight for each entity.
  int n_real_valued_attrs_; // how many real valued attributes in the dataset.
  int link_attr_[2];        // which types of entities are being linked?
  RealDistr model_real_;    // are we fitting a normal/gaussian too? which distr?
  int model_links_;         // are we modelling a link corpus too?

  double *beta_;            // sym dir prior for entity types. (params for H)
  int link_weight_;         // how important is each link compared to a doc.

  int gamma_;               // Top level concentration param.
  int alpha_;               // Restaurant level concentration param.

  double discount_;         // Pitman Yor discount for dish pair allocation for links.
  int num_init_topics_;     // number of topics initial.

  void DebugDisplay();
  int GetNumTrainingDocs()  const {return n_docs_;}
  int GetNumTestDocs()      const {return n_test_docs_;}
  int GetNumTrainingLinks() const {return n_train_links_;}
  int GetNumTestLinks()     const {return n_test_links_;}

  Franchise *franchise_;
  Config(RealDistr m, int l = 0):
      model_real_(m),
      model_links_(l) {
    link_attr_[0] = -1;
    link_attr_[1] = -1;
  }
  ~Config();
 private:
  int n_docs_;        // num of docs in test/main corpus.
  int n_test_docs_;   // num of docs in corpus.

  int n_train_links_; // num of links in train corpus.
  int n_test_links_;  // num of links in test corpus.
};

void Config::DebugDisplay() {
  cout << "Docs: " << n_docs_ << endl;
  cout << "Types: " << n_entity_types_ << endl;
  for (int i = 0; i < n_entity_types_; ++i) {
    cout << "Vocab = " << vocab_size_[i] << " ";
    cout << "Weight = " << entity_weight_[i] << " ";
    cout << endl;
  }
  cout << "Link Attr: " << link_attr_[0] << "->" << link_attr_[1] << endl;
  cout << gamma_ << alpha_ << endl;
}

Config::~Config() {
  delete[] beta_;
  delete[] vocab_size_;
  delete[] entity_weight_;
}

double ReadDouble(istream &ifs) {
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

void Config::ReadConfig(const string &config_file) {
  // Format: nDocs nTestDocs nTypesOfEntities vocab_sizes weights time_in_data
  ifstream ifs(config_file.c_str());
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

  getline(ifs, config_line);
  istringstream iss_3(config_line);
  iss_3 >> link_attr_[0] >> link_attr_[1]; // Watchout for old configs with 
                                           // only one link attr.

  n_train_links_ = ReadLine(ifs);
  n_test_links_ = ReadLine(ifs);
  link_weight_  = ReadLine(ifs);

  getline(ifs, config_line);
  istringstream iss_4(config_line);
  beta_ = new double[n_entity_types_];
  for (int i = 0; i < n_entity_types_; ++i)
    iss_4 >> beta_[i]; // Prior for words in each topic(dish).

  gamma_ = ReadLine(ifs); // Top level concentration param.
  alpha_ = ReadLine(ifs); // Restaurant level concentration param.
  discount_ = ReadDouble(ifs); // Pitmanyor
  num_init_topics_ = ReadLine(ifs); 
  ifs.close();
}

class Dish;
class Chain;
class Restaurant;

class Table {
 public:
  const Config &config_;
  typedef vector<map<int, int> > Hist; // [nTypes][id->count]
  Dish *dish_id_;                      // what dish served at this table.
  int count_;                          // how many people at this table.
  Hist hist_;                          // how many of each word in this table.
  int name_;
  Table(const Config &con, Dish *d, int n_types, int name, int c = 0):
      config_(con), dish_id_(d), hist_(n_types), name_(name), count_(c){}

  void RemoveWord(int type, int word_id);
  void AddWord(int type, int word_id);
};

class Links;
class HubDish;
class Franchise {
 public:
  const Config &config_;
  list<Dish *> menu_;
  Chain &train_chain_;
  Chain &test_chain_;

  Links *train_links_;
  Links *test_links_;

  Franchise(const Config &c, Chain &chain, Chain &t) :
            config_(c),
            train_chain_(chain), test_chain_(t),
            train_links_(NULL), test_links_(NULL) {}
  void SetLinks(Links *tr, Links *te) {
    train_links_ = tr;
    test_links_  = te;
  }
  void Allocate() {}
  void Free();

  void Sample();
  void SampleUnseen();
  void DiscontinueDish(Dish *);
  Dish* IntroduceNewDish();
  HubDish* IntroduceNewHubDish();

  void RemoveChain(Chain &chain);
  void AddRestaurant(Restaurant&);
  void RemoveRestaurant(Restaurant&);

  void Save(const string&);

  int TotalTableCount();

  void CheckIntegrity(int verbose, int extra);
  void RandomInitialize();
};

class Dish {
 public:
  const Config &config_;

  static int name_ctr_;
  int name_; // "Name" for dish. Only used for readability when dumped.
  int **counts_words_;    // [nTypes][vocab_size]
  int *sum_counts_words_; // [nTypes]
  int num_tables_; // how many tables in the franchise serve this dish?

  bool isHub() {
    return name_ == 100008;
  }

  typedef double Pair[2];
  typedef double Triple[3];
  Pair   *gaussian_parameters_;  // [nRealTypes][2]
  Triple *real_attr_suff_stats_; // [nRealTypes][3]
                                 // sum, sum_of_sq, count

  Dish(const Config &c):config_(c), name_(name_ctr_++), num_tables_(0) {}

  void Allocate();
  void Free();
  //void RemoveTable(list<Table>::iterator, bool);
  virtual void RemoveTable(Table &, bool);
  //void AddTable(list<Table>::iterator);
  virtual void AddTable(Table &);

  list<Dish *>::iterator hook_into_list_;

  void MakeSelfAware(list<Dish *>::iterator i) { 
    hook_into_list_ = i;
  }

  long double WordProb(int type, int word_id) { 
    //sout(50) << "Type " << type << "Word " << word_id << '\n';
    //sout(50) << "count " << counts_words_[type][word_id] << '\n';
    //sout(50) << "sum " << sum_counts_words_[type] << '\n';
    long double t = (counts_words_[type][word_id] + config_.beta_[type]) * 1.0L /
         (sum_counts_words_[type] + config_.beta_[type] * config_.vocab_size_[type]); 
    //sout(50) << "Prob " << t << endl;
    return t;
  }

  virtual void RemoveWord(int type, int word_id) {
    counts_words_[type][word_id]--;
    sum_counts_words_[type]--;
  }

  virtual void AddWord(int type, int word_id) {
    counts_words_[type][word_id]++;
    sum_counts_words_[type]++;
  }
  
  static bool LessThanZero(int a) {
    return a < 0;
  }

  bool CheckIntegrity() {
    bool ret = true;
    for (int t = 0; t < config_.n_entity_types_; ++t) {
      int *target = find_if(counts_words_[t], counts_words_[t] + config_.vocab_size_[t], &LessThanZero);
      if (target != counts_words_[t] + config_.vocab_size_[t]) {
        cout << "Name - " << name_ << "Type " << t << "," << target - counts_words_[t] << " broken" << endl;
        ret = false;
      }
      int s;
      if ((s = accumulate(counts_words_[t], counts_words_[t] + config_.vocab_size_[t], 0)) != sum_counts_words_[t]) {
        cout << "Name - " << name_ << " Type " << t << " sum broken " << sum_counts_words_[t] << " " << s << endl;
        ret = false;
      }
    } // end type
    return ret;
  }
};

class HubDish:public Dish {
 public:
  HubDish(const Config &c):Dish(c) {cout << "Creating hub dish\n"; name_ = 100008;}
  void AddWord(int type, int word_id) {}
  void RemoveWord(int type, int word_id) {}
  void BootInWord(int type, int word_id) {
    Dish::AddWord(type, word_id);
  }
  void RemoveTable(Table &, bool) {}
  void AddTable(Table &) {}
};

void Franchise::CheckIntegrity(int v = 0, int extra = 0) {
  int n_words = 0;
  for (list<Dish*>::iterator i = menu_.begin(); i != menu_.end(); ++i) {
    n_words += accumulate((*i)->sum_counts_words_, (*i)->sum_counts_words_ + config_.n_entity_types_, 0);
    if (!(*i)->CheckIntegrity()) {
      cout << "Extra " << extra << endl;
    }
  }
  if (v) {
    cout << "Total words " << n_words << " Tag " << extra << endl;
  }
}

int Franchise::TotalTableCount() {
  int tot = 0;
  for (list<Dish *>::iterator i = menu_.begin(); i != menu_.end(); ++i) {
    tot += (*i)->num_tables_;
  }
  return tot;
}

int Dish::name_ctr_ = 100;

void Franchise::Save(const string &output_prefix) {
  int type = 0;
  // Save multinomials.
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    ostringstream oss;
    oss << output_prefix << ".model" << "." << type;
    ofstream ofs(oss.str().c_str());
    oss << ".raw";
    ofstream ofs_raw(oss.str().c_str());
    int type_cnt = 0;
    for (list<Dish*>::iterator i = menu_.begin(); i != menu_.end(); ++i) 
      type_cnt += (*i)->sum_counts_words_[type];
    for (list<Dish*>::iterator i = menu_.begin(); i != menu_.end(); ++i) {
      ofs     << (*i)->name_ << ' ';
      ofs_raw << (*i)->name_ << ' ';
      for (int word = 0; word < config_.vocab_size_[type]; ++ word) {
        ofs << (*i)->WordProb(type, word) << ' ';
        ofs_raw << (*i)->counts_words_[type][word] << ' ';
      }
      ofs_raw << '\t';
      double topic_weight =  (*i)->sum_counts_words_[type] * 1.0 / type_cnt;
      ofs_raw << (*i)->sum_counts_words_[type] << ' ' 
              << type_cnt << ' ' << topic_weight << '\t';
      ofs_raw << (*i)->num_tables_ << '\n';

      ofs << topic_weight;
      ofs << '\n';
    }
    ofs.close();
    ofs_raw.close();
  }  // end type
}

void Table::RemoveWord(int type, int word_id) {
  count_ -= config_.entity_weight_[type];
  hist_[type][word_id]--;
  dish_id_->RemoveWord(type, word_id);
}

void Table::AddWord(int type, int word_id) {
  count_ += config_.entity_weight_[type];
  hist_[type][word_id]++;
  dish_id_->AddWord(type, word_id);
}

void Dish::AddTable(Table &table) {
  num_tables_++; // removing effect of table
  int type_ctr = 0;
  for (Table::Hist::iterator i = (table.hist_).begin();
       i != (table.hist_).end(); ++i, ++type_ctr) {
    for (map<int, int>::iterator j = i->begin(); j != i->end(); ++j) {
      counts_words_[type_ctr][j->first] += j->second;
      sum_counts_words_[type_ctr] += j->second;
    } // end word types
  } // end type
}

// remove_dishes flag indicates if a dish should be deleted if no table serves
// it.  when performing inference on test corpora we don't want to delete dish
// because the document can be reintroduced later.

void Dish::RemoveTable(Table &table, bool remove_dishes = true) {
  num_tables_--; // removing effect of table
  if (remove_dishes && num_tables_ == 0) { // last table offering this dish in the franchise.
    (config_.franchise_)->DiscontinueDish(table.dish_id_);
    return;
  }
  int type_ctr = 0;
  for (Table::Hist::iterator i = (table.hist_).begin();
       i != (table.hist_).end(); ++i, ++type_ctr) {
    for (map<int, int>::iterator j = i->begin(); j != i->end(); ++j) {
      counts_words_[type_ctr][j->first] -= j->second;
      sum_counts_words_[type_ctr] -= j->second;
    } // end word type
  } // end type
}

void Dish::Allocate() {
  counts_words_ = new int* [config_.n_entity_types_];
  for (int i = 0; i < config_.n_entity_types_; ++i) {
    counts_words_[i] = new int[config_.vocab_size_[i]]();
    fill_n(counts_words_[i], config_.vocab_size_[i], 0);
  }
  // Using value initialization.
  // The trailing parantheses to set elements to 0.
  sum_counts_words_     = new int[config_.n_entity_types_]();
  fill_n(sum_counts_words_, config_.n_entity_types_, 0);
  gaussian_parameters_  = new Pair[config_.n_entity_types_]();
  real_attr_suff_stats_ = new Triple[config_.n_entity_types_]();

  num_tables_ = 0;
}

void Dish::Free() {
  for (int i = 0; i < config_.n_entity_types_; ++i) {
    delete[] counts_words_[i];
  }
  delete[] counts_words_;
  delete[] sum_counts_words_;
  delete[] gaussian_parameters_;
  delete[] real_attr_suff_stats_;
}

//  Restaurant is synonym for Document.
class Restaurant {
 public:
  typedef list<Table>::iterator TableAllotment;
  class RealAttr {
   public:
    double val_;
    bool   valid_;
  };
  RealAttr *real_attrs_;    // [nRealTypes]
  
  const Config &config_;
  list<Table> tables_  ; // Dishes assigned to tables.

  int             *num_words_;       // [nTypes]
  int            **words_;           // [nTypes][words]
  TableAllotment **assigned_tables_; // [nTypes][words]
                                     // which table is the word assigned to. 

  bool CheckIntegrity() {
    int ctr = 0;
    for (list<Table>::iterator i = tables_.begin(); i != tables_.end(); ++i) {
      if (i->count_ <= 0) {
        cout << "Something weird in this document - table  "  << ctr << endl;
        return false;
      }  
      ctr++;
    } // end tables
    return true;
  }
  Restaurant(const Config &c):config_(c), table_num_(10) {}
  void Allocate();
  void Free();
  void Save(ostream &, ostream &);

  TableAllotment OpenNewTable(int type, int id); 
  void DeleteTable(list<Table>::iterator);

  void ComputeLikelihood(vector<long double> &);
  void SampleDishesForTables();
  void SampleTablesForCustomers();
  void Initialize();
  void ResetTables();

  int table_num_; // table number names.
 private:
  int doc_weight_;
  int ComputeDocWeight();

};

void Restaurant::ComputeLikelihood(vector<long double> &rest_ll) {
  fill_n(rest_ll.begin(), config_.n_entity_types_, 0.0);
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int word_idx = 0; word_idx < num_words_[type]; ++ word_idx) {
      long double word_prob = 0.0;
      for (list<Table>::iterator table_iter = tables_.begin();
            table_iter != tables_.end(); ++table_iter) {
        long double table_prob = table_iter->count_ *
          (table_iter->dish_id_)->WordProb(type, words_[type][word_idx]);
        word_prob += table_prob;
        if (isnan(word_prob) || (word_prob < 0) || isnan(table_prob) || (table_prob < 0)) {
          cout << table_iter->count_ << endl;
          cout << (table_iter->dish_id_)->WordProb(type, words_[type][word_idx]) << endl;
          cout << "Whoops " << " type " << type << " word " << word_idx << word_prob << " table prob " << table_prob << endl;
        }
      } // end tables
      word_prob /= doc_weight_; // normalize table prob
      rest_ll[type] += config_.entity_weight_[type] * log(word_prob);
    } // end word
  } // end type
}


void Restaurant::Initialize() {
  doc_weight_      = ComputeDocWeight();
}

void Restaurant::DeleteTable(list<Table>::iterator i) {
  (i->dish_id_)->num_tables_--;
  if ((i->dish_id_)->num_tables_ == 0) {
    (config_.franchise_)->DiscontinueDish(i->dish_id_);
  }
  tables_.erase(i);
}

int Restaurant::ComputeDocWeight() {
  int weight = 0;
  for (int i = 0; i < config_.n_entity_types_; ++i) {
    weight += config_.entity_weight_[i] * num_words_[i];
  }
  return weight;
}


void Restaurant::SampleDishesForTables() {
  int num_existing_dishes = (config_.franchise_)->menu_.size();
  vector<long double> cdf(num_existing_dishes + 1, 0);
  // Just reserving a lot of space in the cdf to avoid reallocations.
  cdf.reserve(tables_.size());

  for (list<Table>::iterator iter = tables_.begin();
         iter != tables_.end();
         ++iter) {
    // Delete table from dish.
    if (iter->dish_id_) { // check it's not booting up time.
      iter->dish_id_->RemoveTable(*iter);
      cdf.resize((config_.franchise_)->menu_.size() + 1); // what if a dish got removed.
    }

    // sample new dish
    int dish_num = 0;
    vector<int> type_counts(config_.n_entity_types_, 0);
    long double prev = 0.0;
    for (list<Dish *>::iterator i = (config_.franchise_)->menu_.begin();
           i != (config_.franchise_)->menu_.end();
           ++i, ++dish_num) {
      long double cur_dish_prob = ((*i)->num_tables_);
      // cdf[dish_num] = ((*i)->num_tables_);
      // Now computing prob of table for the cur dish.
      int type = 0;
      for (Table::Hist::iterator type_iter = (iter->hist_).begin(); 
           type_iter != (iter->hist_).end(); ++type_iter, ++type) {
        for (map<int, int>::iterator id_iter = type_iter->begin(); 
              id_iter != type_iter->end(); ++id_iter) {
          if (i == (config_.franchise_)->menu_.begin())
            type_counts[type] += id_iter->second;
          long double word_prob = (*i)->WordProb(type, id_iter->first);
          cur_dish_prob *= pow(word_prob, (id_iter->second) * config_.entity_weight_[type]);
        } // end words
      } // end types
      cdf[dish_num] = prev + cur_dish_prob;
      prev = cdf[dish_num];
    } // end dishes

    // prob of opening up a new dish.
    long double new_dish_prob = config_.gamma_;
    for (int type = 0; type < config_.n_entity_types_; ++type) {
      new_dish_prob *= 
        pow (static_cast<long double>(1.0) / config_.vocab_size_[type],
             type_counts[type]);
    }
    cdf[(config_.franchise_)->menu_.size()] = prev + new_dish_prob;

    // sampling dish.
    int new_dish_idx = MultSample(cdf, (config_.franchise_)->menu_.size() + 1);
    if (new_dish_idx == (config_.franchise_)->menu_.size()) { // opening new dish
      iter->dish_id_ = (config_.franchise_)->IntroduceNewDish();
      cdf.resize((config_.franchise_)->menu_.size() + 1); // make space in cdf.
    } else {
      list<Dish *>::iterator tmp;
      GetIterator((config_.franchise_)->menu_, new_dish_idx, tmp);
      iter->dish_id_ = *tmp;
    }
    // sout(30) << "Selected dish " << new_dish_idx << endl;
    // Add table to dish.
    iter->dish_id_->AddTable(*iter);
  } // end table
}

void Restaurant::SampleTablesForCustomers() {
  vector<long double> cdf(tables_.size() + 1, 0);
  // Just reserving a lot of space in the cdf to avoid reallocations.
  cdf.reserve(accumulate(num_words_, num_words_ + config_.n_entity_types_, 0));
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    for (int id = 0; id < num_words_[type]; ++id) {
      
      // remove this word.
      TableAllotment cur_table = assigned_tables_[type][id];
      // during initialization there is no remove word needed.
      if (cur_table != TableAllotment()) { 
        cur_table->RemoveWord(type, words_[type][id]);
        if (cur_table->count_ == 0) {
          DeleteTable(cur_table);
          cdf.resize(tables_.size() + 1);
        }
      }
      //sout(30) << "Removed word " << id << endl;

      // now compute CDF over tables.
      int table_ctr = 0;
      long double prev = 0.0;
      for (list<Table>::iterator i = tables_.begin();
            i != tables_.end();
            ++i, table_ctr++) {
        Dish *dish = i->dish_id_;
        //sout(40) << words_[type][id] << endl;
        cdf[table_ctr] = prev +
            i->count_ * dish->WordProb(type, words_[type][id]); // word prob
        //sout(40) << cdf[table_ctr] << '\n';
        prev = cdf[table_ctr];
      } // end computing prob for existing tables

      // Now we compute prob for opening up a new table.

      long double new_table_prob = 0.0;
      int total_table_cnt = 0;
      for (list<Dish *>::iterator i = (config_.franchise_)->menu_.begin();
             i != (config_.franchise_)->menu_.end();
             ++i) {
         new_table_prob += 
            (*i)->num_tables_ * (*i)->WordProb(type, words_[type][id]);
         total_table_cnt += (*i)->num_tables_;
      }
      // prob of opening up a new dish.
      new_table_prob += config_.gamma_ * 1.0 / config_.vocab_size_[type];
      //sout(40) << " New dish prob" <<  config_.gamma_ << " / " << config_.vocab_size_[type] << " = " << new_table_prob << '\n';

      cdf[table_ctr] = prev +
          config_.alpha_ * new_table_prob / (config_.gamma_ + total_table_cnt);
      //sout(40) << "Table ctr " << cdf[table_ctr] << '\n';

      int new_table_idx = MultSample(cdf, tables_.size() + 1);
      //sout(40) << new_table_idx << " selected table\n";
      //sout(30) << "Table num" << new_table_idx << endl;
      TableAllotment new_table;
      if (new_table_idx == tables_.size()) { // opening new table
        //sout(30) << " New table in the house\n";
        //sout(40) << "  New Table\n";
        new_table = OpenNewTable(type, words_[type][id]);
        //sout(40) << "  Done with New Table\n";
        cdf.resize(tables_.size() + 1);
      } else {
        GetIterator (tables_, new_table_idx, new_table);
      } // end if

      // reintroduce word with new table
      assigned_tables_[type][id] = new_table;
      new_table->AddWord(type, words_[type][id]);
      //sout (30) << " Added word back\n";
    } // end words
  } // end type
}

Restaurant::TableAllotment Restaurant::OpenNewTable(int type, int id) {
  long double *cdf = new long double[(config_.franchise_)->menu_.size() + 1];
  long double prev = 0.0;
  int dish_ctr = 0;
  for (list<Dish *>::iterator i = (config_.franchise_)->menu_.begin();
         i != (config_.franchise_)->menu_.end();
         ++i, ++dish_ctr) {

    long double dish_prob = (*i)->num_tables_ * (*i)->WordProb(type, id);
    cdf[dish_ctr] = prev + dish_prob;
    prev = prev + dish_prob;
  }
  // prob of opening up a new dish.
  cdf[dish_ctr] = prev + config_.gamma_ * 1.0 / config_.vocab_size_[type];

  Dish *dish_for_new_table;
  int new_dish_idx = MultSample(cdf, (config_.franchise_)->menu_.size() + 1);
  //sout(40) << "  Selecting dish num " << new_dish_idx << endl;
  if (new_dish_idx == (config_.franchise_)->menu_.size()) { // opening new dish.
    //sout(30) << "    New dish for new table\n";
    dish_for_new_table = (config_.franchise_)->IntroduceNewDish();
  } else {
    list<Dish *>::iterator tmp = (config_.franchise_)->menu_.begin();
    for (int ctr = 0; ctr < new_dish_idx; ++ctr)
      ++tmp;
    dish_for_new_table = *tmp;
  }
  delete[] cdf;
  dish_for_new_table->num_tables_++;
  return tables_.insert(tables_.end(),
                        Table(config_, dish_for_new_table, config_.n_entity_types_, table_num_++)
                          );
}

void Restaurant::Allocate() {
  num_words_       = new int[config_.n_entity_types_]();
  words_           = new int*[config_.n_entity_types_](); 
  assigned_tables_ = new TableAllotment*[config_.n_entity_types_](); 
  real_attrs_      = new RealAttr[config_.n_real_valued_attrs_]();
  // num_words_ words_ and assigned_tables_ will be initialized during Read and Init.
}

void Restaurant::Free() {
  delete[] real_attrs_;
  for (int i = 0; i < config_.n_entity_types_; ++i) {
    delete[] words_[i];
    delete[] assigned_tables_[i];
  }
  delete[] words_;
  delete[] assigned_tables_;
  delete[] num_words_;
}

class Link {
 public:
  int id_1;
  int id_2;
  int& operator[](int i) {
    if (i == 0)
      return id_1;
    if (i == 1)
      return id_2;
  }
  int operator[](int i) const {
    if (i == 0)
      return id_1;
    if (i == 1)
      return id_2;
  }
  operator int() const {
    return id_1 * 2000000 + id_2;
  }
};

// This class is a table where links can sit.
class TablePair {
 public:
  const Config &config_;
  pair<Table, Table> big_table_;
  map<Link, int> hist_;
  int count_;
  TablePair(const Config &c, pair<Dish *, Dish *> d):
                config_(c),
                count_(0), 
                big_table_(
                  Table(c, d.first, config_.n_entity_types_, table_name_idx_ + 0),
                  Table(c, d.second, config_.n_entity_types_, table_name_idx_ + 1)
                ) { table_name_idx_ += 2;}

 void RemoveLink(Link &l) {
   count_--;
   hist_[l]--;
   if (hist_[l] == 0)
     hist_.erase(l);

   big_table_.first.RemoveWord(config_.link_attr_[0], l[0]);
   big_table_.second.RemoveWord(config_.link_attr_[1], l[1]);
 }

 void AddLink(Link &l) {
   count_++;
   hist_[l]++;
   big_table_.first.AddWord(config_.link_attr_[0], l[0]);
   big_table_.second.AddWord(config_.link_attr_[1], l[1]);
 }

 void DisassociateFromDishPair() {
    if (big_table_.first.dish_id_) { // check it's not booting up time.
      big_table_.first.dish_id_->RemoveTable(big_table_.first);
      big_table_.second.dish_id_->RemoveTable(big_table_.second);
    }
 }

 void AddToDishPair(pair<Dish *, Dish *> dish_pair) {
   big_table_.first.dish_id_ = dish_pair.first;
   big_table_.second.dish_id_ = dish_pair.second;

   dish_pair.first->AddTable(big_table_.first);
   dish_pair.second->AddTable(big_table_.second);
 }
 private:
  static int table_name_idx_;
};

int TablePair::table_name_idx_ = 600; // just a random number to start numbering tables in pairs with.

class Links {
 public: 
  const Config &config_;
  vector<Link> edges_; // [nLinks]
  Links(const Config &c, int n): config_(c), 
                                 assigned_tables_(n, BigTableAllotment()), edges_(n) {}

  list<TablePair> tablepairs_; // n opened tables
  typedef list<TablePair>::iterator BigTableAllotment;
  vector<BigTableAllotment> assigned_tables_; // nLinks

  void Slurp(const string &file_name);
  void Save(const string&);
  long double ComputePerplexity();
  void SampleTablesForCustomers();

  void                 DeleteTablePair(BigTableAllotment); // remove table pair.
  BigTableAllotment    OpenNewTablePair(vector<long double> &cdf);
  pair<Dish *, Dish *> SampleDishPair(vector<long double> &); 
  void ComputeCDF(map<Link, int> &edge_hist, vector<long double> &op_cdf);
  void SampleDishesForTables();
  void ResetTables() {
    fill(assigned_tables_.begin(), assigned_tables_.end(), BigTableAllotment());
  }

 private:
  int dummy_link_[2];
  long double TablePairProb(BigTableAllotment b) {
    return b->count_ * 1.0 / edges_.size();
  }
};

// TODO: table counts when links are blended in.
// TODO: Ensure that count_ in constituents of TablePairs don't get used
// without checking it is set correctly.

void Links::ComputeCDF(map<Link, int> &links, vector<long double> &cdf) {
  // cdf must be m+1 ^ 2.

  // cout << "Computing CDF\n";
  int pair_ctr = 0;
  long double prev = 0.0;
  double denominator_norm = config_.gamma_;

  long double entity_1_prob, entity_2_prob;
  list<Dish *>::iterator d1_iter = (config_.franchise_)->menu_.begin();
  list<Dish *>::iterator d2_iter = (config_.franchise_)->menu_.begin();

  int m = config_.franchise_->menu_.size();

  for (int d1 = 0; d1 < m + 1; ++d1, ++d1_iter) {
    for (int d2 = 0; d2 < m + 1; ++d2, ++d2_iter) {
      long double count_1_prob;
      long double count_2_prob;
      if (d1 != m) // not a new dish
        count_1_prob  = (*d1_iter)->num_tables_ - config_.discount_;
      else
        count_1_prob = config_.gamma_ + config_.discount_ * m;

      if (d2 != m) // not a new dish
        count_2_prob  = (*d2_iter)->num_tables_ - config_.discount_;
      else
        count_2_prob = config_.gamma_ + config_.discount_ * m;
      long double tablepair_prob = sqrt(count_1_prob * count_2_prob);
      denominator_norm += tablepair_prob;

      for (map<Link, int>::iterator hist_iter = links.begin();
            hist_iter != links.end(); ++hist_iter)  {
        if (d1 != m) { // not a new dish
          entity_1_prob = (*d1_iter)->WordProb(config_.link_attr_[0], hist_iter->first[0]);
        } else {
          entity_1_prob = 1.0 / config_.vocab_size_[config_.link_attr_[0]];
        }
        if (d2 != m) { // not a new dish
          entity_2_prob = (*d2_iter)->WordProb(config_.link_attr_[1], hist_iter->first[1]);
        } else {
          entity_2_prob = 1.0 / config_.vocab_size_[config_.link_attr_[1]];
        }
        tablepair_prob *= pow(entity_2_prob * entity_2_prob, hist_iter->second);
        /*
          cout << count_1_prob << ", "
               << count_2_prob << ", "
               << sqrt(count_1_prob * count_2_prob) << " X "
               << entity_1_prob << " X "
               << entity_2_prob << " ^ "
               << hist_iter->second << " =  "
               << pow(sqrt(count_1_prob * count_2_prob) * entity_2_prob * entity_2_prob, hist_iter->second) << '\n';
               */
      } // end table's links
      // cout << pair_ctr << "\n";
      // TODO: link weights for dishes sampling
      cdf[pair_ctr] = prev + tablepair_prob;
      prev = cdf[pair_ctr];
      pair_ctr++;
    } // end dish 2
  } // end dish 1

  pair_ctr = 0;
  for (int d1 = 0; d1 < m + 1; ++d1, ++d1_iter) {
    for (int d2 = 0; d2 < m + 1; ++d2, ++d2_iter) {
      cdf[pair_ctr] /= denominator_norm;
      pair_ctr++;
    } // end d2
  } // end d1
}

void Links::SampleDishesForTables() {
//  sout(40) << "Sampling dishes for link tables" << endl;
  int m = (config_.franchise_)->menu_.size();
  vector<long double> cdf((m + 1) * (m + 1), 0); // Just reserving a lot of space in the cdf to avoid reallocations.

  cdf.reserve(tablepairs_.size());

  for (list<TablePair>::iterator iter = tablepairs_.begin();
         iter != tablepairs_.end();
         ++iter) {
    // Delete table from dish.
    iter->DisassociateFromDishPair();
    m = (config_.franchise_)->menu_.size();
    cdf.resize((m + 1) * (m + 1));

    // TODO: effects of links in dishes - investigate.
    // sample new dish
    ComputeCDF(iter->hist_, cdf);
    pair<Dish *, Dish *> sel_dishes = SampleDishPair(cdf);
    iter->AddToDishPair(sel_dishes);
  } // end table
  ostringstream oss;
  static int iter = 0;
  oss << "debug_after_dishes." << iter++;
  Save(oss.str());
}

Links::BigTableAllotment Links::OpenNewTablePair(vector<long double> &cdf) {
  /*
  cout << "Opening new table pair - "; cout.flush();
  cout << "CDF \n";
  copy(cdf.begin(), cdf.end(), ostream_iterator<long double>(cout, "\n"));
  cout << "\n\n\n";
  */

  pair<Dish *, Dish *> sel_dishes = SampleDishPair(cdf);
  sel_dishes.first->num_tables_++;
  sel_dishes.second->num_tables_++;

  return tablepairs_.insert(tablepairs_.end(), TablePair(config_, sel_dishes));
}

pair<Dish *, Dish *> Links::SampleDishPair(vector<long double> &cdf) {
  int m = static_cast<int>(sqrt(cdf.size())) - 1;
  pair <Dish *, Dish *> sel_dishes;

  
  /*
  sout(40) << "Distr (";
  for (int sample = 0; sample < cdf.size(); ++sample) {
    sout(40) << sample << " " << cdf[sample] / cdf[cdf.size() - 1] << '\n';
  }
  sout(40) << ")\n";
  */
  
  
  int new_dish_pair_idx = MultSample(cdf, (m + 1) * (m + 1));
  //cout << "Selected dish pair " << new_dish_pair_idx << " Dishes num " << m << endl;
  //cout << new_dish_pair_idx / (m + 1) << " , " << new_dish_pair_idx % (m + 1) << endl;
  if ((new_dish_pair_idx + 1) >= (m * (m + 1))) { // opening new dish for table 1.
    sel_dishes.first = (config_.franchise_)->IntroduceNewDish();
  } else {
    // list<Dish *>::iterator tmp = (config_.franchise_)->menu_.begin();
    // for (int ctr = 0; ctr < new_dish_pair_idx / (m + 1); ++ctr)
    //   ++tmp;
    // sel_dishes.first = *tmp;
    list<Dish *>::iterator tmp;
    GetIterator(config_.franchise_->menu_, new_dish_pair_idx / (m + 1), tmp);
    sel_dishes.first = *tmp;
  }
  if (new_dish_pair_idx % (m + 1) == m) { // opening new dish for table 2.
    sel_dishes.second = (config_.franchise_)->IntroduceNewDish();
  } else {
    // list<Dish *>::iterator tmp = (config_.franchise_)->menu_.begin();
    // for (int ctr = 0; ctr < new_dish_pair_idx % (m + 1) ; ++ctr)
    //   ++tmp;
    // sel_dishes.second = *tmp;
    list<Dish *>::iterator tmp;
    GetIterator(config_.franchise_->menu_, new_dish_pair_idx % (m + 1), tmp);
    sel_dishes.second = *tmp;
  }
  return sel_dishes;
}

void Links::DeleteTablePair(BigTableAllotment dead_pair) {
  (dead_pair->big_table_).first.dish_id_->num_tables_--;
  (dead_pair->big_table_).second.dish_id_->num_tables_--;
  if ((dead_pair->big_table_).first.dish_id_->num_tables_ == 0) {
    (config_.franchise_)->DiscontinueDish((dead_pair->big_table_).first.dish_id_);
  }
  if ((dead_pair->big_table_).second.dish_id_->num_tables_ == 0) {
    (config_.franchise_)->DiscontinueDish((dead_pair->big_table_).second.dish_id_);
  }
  tablepairs_.erase(dead_pair);
}

void Links::SampleTablesForCustomers() {
  //sout(40) << "Sampling tables for links" << endl;
  int m = (config_.franchise_)->menu_.size();

  // to accumulate stats for prob of new table making it easier to sample new
  // table in case the sampler wants it.
  vector<long double> newtable_cdf((m + 1) * (m + 1), 0.0); 

  newtable_cdf.reserve(edges_.size()); // reserving big enough block of memory.

  vector<long double> cdf(tablepairs_.size() + 1, 0); 
  cdf.reserve(edges_.size());      // Just reserving a lot of space in the cdf 
                                   // to avoid reallocations.

  for (int id = 0; id < edges_.size(); ++id) {
    // remove this link.
    BigTableAllotment cur_table = assigned_tables_[id];
    long double prev = 0.0;
    // during initialization there is no remove word needed.
    if (cur_table != BigTableAllotment()) { 
      cur_table->RemoveLink(edges_[id]);
      if (cur_table->count_ == 0) {
        DeleteTablePair(cur_table);
        cdf.resize(tablepairs_.size() + 1);
        m = (config_.franchise_)->menu_.size();
        newtable_cdf.resize((m + 1) * (m + 1));
      } // if empty table
    } // done with deleting link

    /* Sato
    // hub node
    double hub_prod_node_1 = ; //TODO
    double hub_prod_node_2 = ; //TODO
    double hub_prob = ; // TODO
    // hub_prob *= global prob of edge_1 and prob of edge 2 in topic 1
    */

    
    // now compute CDF over tables.
    int tablepair_ctr = 0;
    for (list<TablePair>::iterator i = tablepairs_.begin();
          i != tablepairs_.end();
          ++i, tablepair_ctr++) {
      Dish *dish_1 = (i->big_table_).first.dish_id_;
      Dish *dish_2 = (i->big_table_).second.dish_id_;
      double discount_factor = 1.0;
      if (dish_1->isHub() || dish_2->isHub()) {
        discount_factor = 0.5;
      }
      
      cdf[tablepair_ctr] = prev + 
          discount_factor *
          i->count_ * 
          dish_1->WordProb(config_.link_attr_[0], edges_[id][0]) *
          dish_2->WordProb(config_.link_attr_[1], edges_[id][1]);
      prev = cdf[tablepair_ctr];
    } // end computing prob for existing tables

    //cdf[tablepair_ctr + 1] = hub_prob * hub_prod_node_1 * regular_prod_node_2;
    //cdf[tablepair_ctr + 2] = hub_prob * hub_prod_node_2 * regular_prod_node_1;

    // Now we compute prob for opening up a new table.
    map<Link, int> tmp_hist_for_edge;
    tmp_hist_for_edge[edges_[id]] = 1; // insert edge into hist
    //cout << "For new table\n";
    ComputeCDF(tmp_hist_for_edge, newtable_cdf);
    cdf[tablepair_ctr] = prev +
      config_.alpha_ * newtable_cdf.back();

    int new_tablepair_idx = MultSample(cdf, tablepairs_.size() + 1);
    BigTableAllotment new_tablepair;

    if (new_tablepair_idx == tablepairs_.size()) { // opening new table pair.
      new_tablepair = OpenNewTablePair(newtable_cdf); 
      int m = (config_.franchise_)->menu_.size();
      newtable_cdf.resize((m + 1) * (m + 1));
      cdf.resize(tablepairs_.size() + 1);
    } else {
      GetIterator(tablepairs_, new_tablepair_idx, new_tablepair);
    } // end if

    // reintroduce word with new table
    assigned_tables_[id] = new_tablepair;
    new_tablepair->AddLink(edges_[id]);

    /*if (id == edges_.size() - 1 || id > 7000) {
      cout << '*'; cout.flush();
      config_.franchise_->CheckIntegrity(0, id);
    }*/
  } // end links
  // sout(40) << tablepairs_.size() << " table pairs for links" << endl;
  ostringstream oss;
  static int iter = 0;
  oss << "debug_after_tables." << iter++;
  Save(oss.str());
}

long double Links::ComputePerplexity() {
  long double link_perplexity = 0.0;
  for (int edge_ctr = 0; edge_ctr < edges_.size(); ++edge_ctr) {
    long double link_prob = 0.0;
    for (list<TablePair>::iterator i = tablepairs_.begin(); 
          i != tablepairs_.end(); ++i) {
      link_prob += i->count_ * 1.0 *
               (i->big_table_).first.dish_id_->WordProb(config_.link_attr_[0],
                                                         edges_[edge_ctr][0]) *
               (i->big_table_).second.dish_id_->WordProb(config_.link_attr_[1],
                                                         edges_[edge_ctr][1]);
    } // end tables
    link_prob /= edges_.size(); // normalize tablepair distr.
    if (isnan(link_prob) || isnan(log(link_prob))) {
      cout << "Link #" << edge_ctr << " Whoa! Whoa! - " <<  link_prob << endl;
    }
    link_perplexity += log(link_prob);
  } // end links.
  long double backup_1 = link_perplexity;
  link_perplexity /= edges_.size();
  long double backup = link_perplexity;
  link_perplexity = pow(static_cast<long double>(2.0),
                        -1.0 * link_perplexity / log(2.0));
  if (isnan(link_perplexity)) {
    cout << "Whoa" << backup << " "<< backup_1 << endl;
  }
  return link_perplexity;
} // end method

void Links::Save(const string &file_name) {
  ofstream ofs_links((file_name + ".links_topic_samples").c_str());
  for (vector<BigTableAllotment>::iterator i = assigned_tables_.begin();
        i != assigned_tables_.end(); ++i) {
    ofs_links << ((*i)->big_table_).first.name_  << '(' << ((*i)->big_table_).first.dish_id_->name_  << ")\t";
    ofs_links << ((*i)->big_table_).second.name_ << '(' << ((*i)->big_table_).second.dish_id_->name_ << ")\n";
  } // end links
  ofs_links.close();

  ofs_links.open((file_name + ".links_topic_distr").c_str());
  for (list<TablePair>::iterator i = tablepairs_.begin(); i != tablepairs_.end(); ++i) {
    ofs_links << (i->big_table_).first.name_  << '(' << (i->big_table_).first.dish_id_->name_  << "), "
              << (i->big_table_).second.name_ << '(' << (i->big_table_).second.dish_id_->name_ << ")\t"
              << i->count_ << '\t'
              << i->count_ * 1.0 / edges_.size() << '\n';
  } // end table pairs.

  ofs_links << '\n';

  // Using Link instead of pair because Link has < overloaded. Avoiding rewriting it 
  // for pairs.
  map<Link, int> dish_pair_hist; // DS to hold tablepairs collapsed to dish pairs
  for (list<TablePair>::iterator i = tablepairs_.begin(); i != tablepairs_.end(); ++i) {
    Link dish_pair;
    dish_pair[0] = i->big_table_.first.dish_id_->name_;
    dish_pair[1] = i->big_table_.second.dish_id_->name_;
    dish_pair_hist[dish_pair] += i->count_;
  } // end table pairs

  for (map<Link, int>::iterator i = dish_pair_hist.begin(); i != dish_pair_hist.end(); ++i) {
    ofs_links << i->first[0] << " " << i->first[1] << '\t'
              << i->second << '\t'
              << (i->second) * 1.0 / edges_.size() << '\n';
  } // end of dish pair histogram iteration

  ofs_links.close();
}

void Links::Slurp(const string &file_name) {
  string line;
  int doc_ctr = 0;
  ifstream ifs(file_name.c_str());
  sout(20) << "Reading in links data" << endl;
  while (getline(ifs, line)) {
    istringstream iss(line);

    int cnt, id_1, id_2;
    iss >> cnt >> id_1 >> id_2;
    edges_[doc_ctr][0] = id_1;
    edges_[doc_ctr][1] = id_2;
    doc_ctr++;

    if (id_1 >= config_.vocab_size_[config_.link_attr_[0]] || 
        id_2 >= config_.vocab_size_[config_.link_attr_[1]]) {
      cerr << "Id in links exceeds vocab size " << id_1 << " " << config_.vocab_size_[config_.link_attr_[0]] << " " << id_2 << " " << config_.vocab_size_[config_.link_attr_[1]] << endl;
      exit(0);
    } // end if

  } // end file
  sout(20) << "Done" << endl;
}

class Chain {
 public:
  void Allocate();
  void Free();
  void Save(const string &);
  const Config &config_;
  Chain(const Config &c, int n):config_(c), restaurants_(n, Restaurant(c)) {}

  vector<Restaurant> restaurants_;
  void Slurp(const string &file_name);
  vector<long double> ComputePerplexity(); 
  void ResetTables();

  void CheckIntegrity() {
    for (int i = 0; i < restaurants_.size(); ++i) {
      if (!restaurants_[i].CheckIntegrity()) {
        cout << "Bug in doc " << i << endl;
      }
    } // end for restaurants
  }
};

void Franchise::RandomInitialize() {
  if (!config_.num_init_topics_) {
    cout << "PY prior" << endl;
    return;
  }

  cout << "Random dish initialization" << endl;
  vector<Dish *> new_dishes(config_.num_init_topics_);
  int num_init_topics = config_.num_init_topics_ - 1; // last topic reserved for hub topic.
  for (int i = 0; i < num_init_topics; ++i) {
    Dish *new_creation = IntroduceNewDish();
    //cout << "New dish at " << new_creation << endl;
    new_dishes[i] = new_creation;
  } // end for dishes

  Chain *chains[2];
  chains[0] = &train_chain_;
  //chains[1] = &test_chain_;
  
  vector<Restaurant::TableAllotment> new_tables(num_init_topics);
  for (int chain_ctr = 0; chain_ctr < 1; ++chain_ctr) {
    for (vector<Restaurant>::iterator i = (chains[chain_ctr])->restaurants_.begin(); i != (chains[chain_ctr])->restaurants_.end(); ++i) {
      // Adding tables to dishes
      for (int table = 0; table < num_init_topics; ++table) {
        new_tables[table] = i->tables_.insert(i->tables_.end(),
                                 Table(config_, new_dishes[table], config_.n_entity_types_, i->table_num_++));
        new_dishes[table]->num_tables_++;
        //cout << "dish id for table is " << new_dishes[table] << endl;
      } // end for dishes

      for (int type = 0; type < config_.n_entity_types_; ++type) {
        for (int word_idx = 0; word_idx < i->num_words_[type]; ++word_idx) {
          Restaurant::TableAllotment rand_table = new_tables[UniformSample(num_init_topics)];
          i->assigned_tables_[type][word_idx] = rand_table;
          // Add word to table and associated dish.
          rand_table->AddWord(type, i->words_[type][word_idx]);
        } // end words
      } // end type

      // Remove unused tables from document.
      for (int table = 0; table < num_init_topics; ++table) {
        if (new_tables[table]->count_ == 0) {
          (new_tables[table]->dish_id_)->num_tables_--;
          (i->tables_).erase(new_tables[table]);
        } // end if table not used
      } // end for over tables

    } // end training documents
  } // end chains

  cout << "Random initialization in links" << endl;
  HubDish *hub_dish =  IntroduceNewHubDish();
  new_dishes[config_.num_init_topics_ - 1] = hub_dish;

  // add table pairs to links
  int pair_ctr = 0;
  int num_pairs = config_.num_init_topics_ * config_.num_init_topics_;
  vector<Links::BigTableAllotment> table_pairs(num_pairs);
  for (int i = 0; i < config_.num_init_topics_; ++i) {
    for (int j = 0; j < config_.num_init_topics_; ++j) {
      pair<Dish *, Dish *> sel_dishes = make_pair(new_dishes[i], new_dishes[j]);
      sel_dishes.first->num_tables_++;
      sel_dishes.second->num_tables_++;
      table_pairs[pair_ctr++] = train_links_->tablepairs_.insert(train_links_->tablepairs_.end(), TablePair(config_, sel_dishes));
    } // end for
  } // end for
  cout << "  Created blocks" << endl;

  // Adding all links to hub topic.
  for (vector<Link>::iterator i = train_links_->edges_.begin(); i != train_links_->edges_.end(); ++i) {
    hub_dish->BootInWord(config_.link_attr_[0], (*i)[0]);
    hub_dish->BootInWord(config_.link_attr_[1], (*i)[1]);
  }

  // Sampling blocks for edges
  for (vector<Link>::iterator i = train_links_->edges_.begin(); i != train_links_->edges_.end(); ++i) {
    Links::BigTableAllotment rand_pair = table_pairs[UniformSample(config_.num_init_topics_) * config_.num_init_topics_ + UniformSample(config_.num_init_topics_)];
    train_links_->assigned_tables_[i - train_links_->edges_.begin()] = rand_pair;
    rand_pair->AddLink(*i);
  }

  // remove unused table pairs 
  cout << " Initialized blocks for edges" << endl;

  for (int table_pair = 0; table_pair < num_pairs; ++table_pair) {
    if (table_pairs[table_pair]->count_ == 0) {
      (table_pairs[table_pair]->big_table_).first.dish_id_->num_tables_--;
      (table_pairs[table_pair]->big_table_).second.dish_id_->num_tables_--;
      train_links_->tablepairs_.erase(table_pairs[table_pair]);
    } // end if tablepair not used
  } // end for over tablepairs
  cout << " Removed unused blocks" << endl;

  // remove unused dishes.
  for (int i = 0; i < config_.num_init_topics_; ++i) {
    if (new_dishes[i]->num_tables_ == 0) {
      cout << "Removing random dish #" << i << "(" << new_dishes[i]->name_ << "_" << endl;
      if (i == config_.num_init_topics_ - 1) {
        cout << "Oh no! Removing hub node" << endl;
      }
      DiscontinueDish(new_dishes[i]);
    }
  } // end for
  cout << " Removed unused topics" << endl;

  cout << "Checking random initialization" << endl;
  train_chain_.CheckIntegrity();
  cout << "Done" << endl;
}

vector<long double> Chain::ComputePerplexity() {
  vector<long double> perplexity(config_.n_entity_types_, 0); //[nTypes]
  vector<int>        counts(config_.n_entity_types_, 0); //[nTypes]

  vector<long double> rest_ll(config_.n_entity_types_);
  int dbg_tnum = 0;
  for (vector<Restaurant>::iterator i = restaurants_.begin(); 
       i != restaurants_.end(); ++i) {
    i->ComputeLikelihood(rest_ll);
    for (int type = 0; type < config_.n_entity_types_; ++type) {
      perplexity[type] += rest_ll[type];
      counts[type] += i->num_words_[type];
    } // end type
    dbg_tnum++;
  } // end restaurants

  for (int type = 0; type < config_.n_entity_types_; ++type) {
    perplexity[type] /= counts[type];
    perplexity[type] = pow(static_cast<long double>(2.0),
                            -1.0 * perplexity[type] / log(2.0));
  }
  return perplexity;
}

void Chain::Save(const string &prefix) {
  string topic_distr_file = prefix + ".topic_distr";
                                        // topic distribution for docs
  string word_topics_file = prefix + ".topic_samples";
                                        // topics chosen for words and entities

  ofstream ofs_distr(topic_distr_file.c_str());
  ofstream ofs_sample(word_topics_file.c_str());
  for (vector<Restaurant>::iterator i = restaurants_.begin(); 
       i != restaurants_.end(); ++i) {
    i->Save(ofs_distr, ofs_sample);
  }
}

void Restaurant::Save(ostream &os_distr, ostream &os_sample) {
  for (list<Table>::iterator i = tables_.begin(); i != tables_.end(); ++i) {
    // table_id(dish_id):p(table | doc) 
    os_distr << i->name_ << '(' << (i->dish_id_)->name_ << "):"
             << i->count_ * 1.0 / doc_weight_ << ' '; 
  } // end table
  os_distr << '\n';

  for (int t = 0; t < config_.n_entity_types_; ++t) {
    for (int i = 0; i < num_words_[t]; ++i) {
      os_sample << assigned_tables_[t][i]->name_  
                << '(' << (*assigned_tables_[t][i]).dish_id_->name_ << ") ";
    }
    os_sample << '\t';
  }
  os_sample << '\n';
}

void Chain::Allocate() {
  for (int i = 0; i < restaurants_.size(); ++i) {
    restaurants_[i].Allocate();
  }
}

void Chain::Free() {
  for (int i = 0; i < restaurants_.size(); ++i) {
    restaurants_[i].Free();
  }
}

// Reset inference for all restaurants
void Chain::ResetTables() {
  for (vector<Restaurant>::iterator i = restaurants_.begin();
       i != restaurants_.end(); ++i) {
    i->ResetTables();
  } // end restaurant
}

void Restaurant::ResetTables() {
  tables_.clear();
  for (int type = 0; type < config_.n_entity_types_; ++type) {
    fill_n(assigned_tables_[type], num_words_[type], list<Table>::iterator());
  } // end type
}

void Chain::Slurp(const string &file_name) {
  string line;
  int doc_ctr = 0;
  ifstream ifs(file_name.c_str());
  sout(20) << "Reading in data" << endl;
  while (getline(ifs, line)) {
    istringstream iss(line);

    Restaurant &cur_restaurant = restaurants_[doc_ctr];
    for (int type = 0; type < config_.n_entity_types_; ++type) {
      int cnt;
      iss >> cnt;
      cur_restaurant.num_words_[type]       = cnt;
      cur_restaurant.words_[type]           = new int[cnt];
      cur_restaurant.assigned_tables_[type] =
          new Restaurant::TableAllotment[cnt];

      for (int i = 0; i < cnt; ++i) {
        int word_id;
        iss >> word_id;
        cur_restaurant.words_[type][i] = word_id;
        if (word_id >= config_.vocab_size_[type]) {
          cerr << "Id in links exceeds vocab size" << endl;
          exit(0);
        } // end if
      } // end reading entities
    } // end types

    for (int real_attr = 0; real_attr < config_.n_real_valued_attrs_;
         ++real_attr) {
      string s;
      iss >> s;
      if (s == "NA" || s == "") {
        cur_restaurant.real_attrs_[real_attr].valid_ = 0;
      } else {
        cur_restaurant.real_attrs_[real_attr].valid_ = 1;
        cur_restaurant.real_attrs_[real_attr].val_ = atof(s.c_str());
      }
    } // read in real valued attributes
    cur_restaurant.Initialize();
    doc_ctr++;
  } // end file
  sout(20) << "Done" << endl;
}

HubDish* Franchise::IntroduceNewHubDish() {
  HubDish *new_dish = new HubDish(config_);
  new_dish->Allocate();
  new_dish->MakeSelfAware(menu_.insert(menu_.end(), new_dish));
//  cout << "New dish born " << new_dish->name_ << endl; 
  return new_dish;
}

Dish* Franchise::IntroduceNewDish() {
  Dish *new_dish = new Dish(config_);
  new_dish->Allocate();
  new_dish->MakeSelfAware(menu_.insert(menu_.end(), new_dish));
//  cout << "New dish born " << new_dish->name_ << endl; 
  return new_dish;
}

void Franchise::DiscontinueDish(Dish *d) {
  menu_.erase(d->hook_into_list_);
  delete d;
}

void Franchise::RemoveChain(Chain &chain) {
  for (vector<Restaurant>::iterator i = chain.restaurants_.begin();
        i != chain.restaurants_.end(); ++i) { 
    RemoveRestaurant(*i); 
  } // end restaurants
}

void Franchise::RemoveRestaurant(Restaurant &restaurant) {
  for (list<Table>::iterator i = restaurant.tables_.begin(); i != restaurant.tables_.end(); ++i) {
    i->dish_id_->RemoveTable(*i);
  } // end tables
}

void Franchise::AddRestaurant(Restaurant &restaurant) {
  for (list<Table>::iterator i = restaurant.tables_.begin(); i != restaurant.tables_.end(); ++i) {
    i->dish_id_->AddTable(*i);
  } // end tables
}

void Franchise::Free() {
  for (list<Dish *>::iterator i = menu_.begin(); i != menu_.end(); ++i) {
    (*i)->Free();
    delete (*i);
  }
}

// do the MCMC
void Franchise::Sample() {
  RandomInitialize();
  if (config_.num_init_topics_) {
    train_chain_.Save("debug.begin.chain");
    Save("debug.begin.model");
  }
  //exit(0);

  for (int iteration = 0; iteration < config_.n_iterations_; ++iteration) {
    sout(10) << "Iteration " << iteration << ' ';
    int dbg_rest_ctr = 0;
    for (vector<Restaurant>::iterator rest_iter = train_chain_.restaurants_.begin();
         rest_iter != train_chain_.restaurants_.end(); ++rest_iter, ++dbg_rest_ctr) {
      rest_iter->SampleTablesForCustomers();
      rest_iter->SampleDishesForTables();
    } // end restaurant
    train_chain_.CheckIntegrity();
    //cout << "Sampled docs" << endl;
    CheckIntegrity();
    sout(10) << " Menu size " << menu_.size() << ' '; sout(10).flush();
    vector<long double> perplexity = train_chain_.ComputePerplexity();

    if (config_.model_links_) {
      //sout(10) << "** "; sout(10).flush();
      ostringstream oss;
      static int iter = 0;
      oss << "debug." << iter++ << ".model";
      Save(oss.str());
      //cout << "Sampling links\n";
      train_links_->SampleTablesForCustomers();
      CheckIntegrity();
      //cout << " ... " << endl;
      train_links_->SampleDishesForTables();
      //cout << "We dun sampled links\n";
      CheckIntegrity();
      //sout(10) << "** "; sout(10).flush();
    }

    sout(10) << "Perplexities: " ;
    copy (perplexity.begin(), perplexity.end(),
          ostream_iterator<long double>(sout(10), " "));

    long double link_perplexity = train_links_->ComputePerplexity();
    sout(10) << "Link - " << link_perplexity;
    sout(10).flush();

    if (iteration != 0 && (iteration % 5 == 0 || iteration == config_.n_iterations_ - 1)) {
      SampleUnseen();
      vector<long double> perplexity = test_chain_.ComputePerplexity();
      sout(10) << " Test perplexities: ";
      copy (perplexity.begin(), perplexity.end(),
            ostream_iterator<long double>(sout(10), " "));
      RemoveChain(test_chain_); // remove effects of test chain from model. 
      // TODO: Should perplexity calc be made after chain is removed?
      if (iteration != config_.n_iterations_ - 1) {
        test_chain_.ResetTables(); // reset table and customer inference in the test corpus.
      } // end if
    } // end test perplexity
    sout(10) << endl;
  } // end iteration
  sout(10) << "Done with MCMC" << endl;
}

void Franchise::SampleUnseen() {
  int dbg_num = 0;
  for (vector<Restaurant>::iterator rest_iter = test_chain_.restaurants_.begin();
       rest_iter != test_chain_.restaurants_.end(); ++rest_iter) {
    //AddRestaurant(*rest_iter); //DBG
    for (int iteration = 0; iteration < 10; ++iteration) {
      rest_iter->SampleTablesForCustomers();
      rest_iter->SampleDishesForTables();
    } // end iteration
    //RemoveRestaurant(*rest_iter); // DBG
    dbg_num++;
  } // end restaurant
}

int main(int argc, char **argv) {
  // cout << LDBL_MAX_10_EXP << " max exponent of long double\n";
  Options opt(argc, argv);
  string train_file       = opt.GetStringValue("train_file");
  string test_file        = opt.GetStringValue("test_file", "/dev/null");
  string config_file      = opt.GetStringValue("config_file");
  string output_prefix    = opt.GetStringValue("output_prefix");
  int debug               = opt.GetIntValue("debug", 0);
  int n_iterations        = opt.GetIntValue("iter", 4);
  int rand_initializer    = opt.GetIntValue("randinit", 0);
  RealDistr model_real    = static_cast<RealDistr>(opt.GetIntValue("model_real", 0));
  // To use links - set Flag and corpus using cmd line.
  int model_links         = opt.GetIntValue("model_links", 0);
  string link_train_file  = opt.GetStringValue("link_train_file", "/dev/null");
  string link_test_file   = opt.GetStringValue("link_test_file", "/dev/null");
  //Setting global values.
  int debug_time          = opt.GetIntValue("debug_time", 0);

  /*
  if (model_real == GAUSSIAN)
    cout << "Using gaussian for reals\n";
  else if (model_real == BETA)
    cout << "Using beta for reals\n";
  */

  if (rand_initializer)
    srand(rand_initializer);

  // Read in config.
  Config config(model_real, model_links);
  config.ReadConfig(config_file);
  //config.DebugDisplay();

  if (model_real && !config.n_real_valued_attrs_) {
    config.model_real_ = NONE;
    cout << "Cannot model time when real information not present in data"
         << endl;
    return 1;
  }

  config.n_iterations_ = n_iterations;

  // Create and initialize the corpus.
  Chain train_chain(config, config.GetNumTrainingDocs());
  Chain test_chain(config, config.GetNumTestDocs());
  train_chain.Allocate();
  train_chain.Slurp(train_file);

  test_chain.Allocate();
  test_chain.Slurp(test_file);

  // Create model object.
  Franchise franchise(config, train_chain, test_chain);
  franchise.Allocate();
  config.franchise_ = &franchise;

  Links *train_links, *test_links;
  if (config.model_links_) {
    train_links = new Links(config, config.GetNumTrainingLinks());
    test_links = new Links(config, config.GetNumTestLinks());
    train_links->Slurp(link_train_file);
    test_links->Slurp(link_test_file);
    franchise.SetLinks(train_links, test_links);
  }

  sout(30) << "Initialized model" << endl;
  franchise.Sample();
  franchise.Save(output_prefix);
  train_chain.Save(output_prefix);
  test_chain.Save(output_prefix + ".test.");
  sout(30) << "Saved models\n" << endl;

  if (config.model_links_) {
    train_links->Save(output_prefix);
    //test_links->Save(output_prefix + ".test.");
  }

  train_chain.Free();
  test_chain.Free();
  franchise.Free();
  return 0;
}
