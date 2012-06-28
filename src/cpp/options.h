#ifndef __OPTIONS_H__
#define __OPTIONS_H__

#include <fstream>
#include <map>
#include <set>
#include <string>

using namespace std;

class Options {
 public:
  map<string, string> user_flags_;

  Options(int argc, char **argv) { ParseCommandLine(argc, argv); }
  void ParseCommandLine(int, char **);
  string GetStringValue(const string&, const string& = "NO");

  void CheckUserFlags();
  void DebugDisplay(ostream &os);

 private:
  set<string> used_flags_;
};

#endif
