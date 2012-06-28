#ifndef __COMPONENT_H__
#define __COMPONENT_H__

#include "config.h"
#include <cmath>

template<class T>
struct Component {
 public:
  static const Config *config_;
  T** counts_;
  int domain_idx_;
  int sel_id_;
  bool access_mode_; //0 - id applies to row
  double total_;
  double *entropy_parts_;

  int start_;
  int end_;

  T Get(int i) {
    return (access_mode_ == 0) ? counts_[sel_id_][i] : counts_[i][sel_id_];
  }

  Component(const Config *cfg, T** c, int sel, bool mode) {
    config_ = cfg;
    counts_ = c;
    sel_id_ = sel;
    //cout << "Id is " << sel_id_ << endl;
    access_mode_ = mode;
    domain_idx_ = -1;
    total_ = 0.0;
  }

  void SetDomain(int id) {
    domain_idx_ = id;
    int skip = 0;
    if (config_->md_seeds_)
      skip = 2; // skip senti topics; dealt with in SplComponent
    start_ = skip;
    end_ = config_->md_splits_[0] - 1;
    for (int i = 0; i < domain_idx_; ++i) {
      start_ += config_->md_splits_[i];
      end_   += config_->md_splits_[i + 1];
    }
    entropy_parts_ = new double[end_ - start_ + 1];
  } // end constructor

  double InitEntropy() {
    total_ = 0;
    for (int i = start_; i <= end_; ++i) {
      double p = (Get(i) + config_->alpha_);
      total_ += Get(i);
      entropy_parts_[i - start_] = p * log(p);
    } // end for
  }

  void Adjust(int topic, T delta) {
    total_ += delta;
    double p = Get(topic) + config_->alpha_;
    entropy_parts_[topic - start_] = p * log(p);
  }

  double Entropy() {
    double h = 0.0;
    for (int i = start_; i <= end_; ++i) {
      h -= entropy_parts_[i - start_];
    } // end for
    double t = (total_ + config_->alpha_ * (end_ - start_ + 1));
    h /= t;
    h += log(t);
    h /= log(2.0);
    return h;
  } // end function

  ~Component() {
    if (entropy_parts_)
      delete[] entropy_parts_;
  }
};

template<class T>
struct SplComponent {
 public:
  static const Config *config_;
  T** counts_;
  int sel_id_;
  bool access_mode_; //0 - id applies to row
  double total_;

  int checks;
  SplComponent(const Config *cfg, T** c, int sel, bool mode, bool dbg = false) {
    config_ = cfg;
    if (dbg)
      cout << "Init spl with " << config_ << endl;
    counts_ = c;
    sel_id_ = sel;
    access_mode_ = mode;

    checks = 1234;
  } // end constructor

  T Get(int i) {
    return (access_mode_ == 0) ? counts_[sel_id_][i] : counts_[i][sel_id_];
  }

  double Entropy() {
    double pos = config_->alpha_;
    double neg = config_->alpha_;
    int idx = 0;
    for (int i = 0; i < config_->md_n_domains_ + 1; ++i) {
      pos += Get(idx);
      neg += Get(idx + 1); //Pos and Neg
      idx += config_->md_splits_[i];
    }
    total_ = pos + neg;

    double h = (pos * log(pos) + neg * log(neg)) * -1;
    h /= total_;
    h += log(total_);
    h /= log(2.0);
    return h;
  } // end function
  ~SplComponent() {
    checks = 9999;
  }

};
template<class T> const Config* SplComponent<T>::config_;
template<class T> const Config* Component<T>::config_;
#endif
