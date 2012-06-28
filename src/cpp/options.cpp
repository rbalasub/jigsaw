#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "options.h"

using namespace std;

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
                                const string &default_value) {
  used_flags_.insert(key);
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

void Options::CheckUserFlags() {
  for (map<string, string>::iterator i = user_flags_.begin(); i != user_flags_.end(); ++i) {
    if (used_flags_.count(i->first) == 0) {
      cerr << "Unknown flag " << i->first << endl;
    } // end if
  } // end map iterator
}
