#include <set>
#include <iostream>

using namespace std;

struct Obj {
  int t1;
  int t2;
  int clo;
  Obj(int a1, int a2, int c) : t1(a1), t2(a2), clo(c) {}
  bool operator==(const Obj &b) const {cout << "Calling ==" << endl;  return (t1 == b.t1) && (t2 == b.t2);}
  bool operator<(const Obj &b) const {cout << "Calling <" << this->clo << " < " << b.clo << endl;  return clo > b.clo;}
  bool operator()(const Obj &a, const Obj &b) const { cout << "Calling ()" << endl; return a.clo > b.clo;}
};

int main() {
  Obj a(1, 2, 100);
  Obj b(1, 3, 100);
  Obj c(1, 2, 20);
  multiset<Obj> myset;
  myset.insert(a);
  myset.insert(b);

  for (set<Obj>::iterator i = myset.begin(); i != myset.end(); ++i) {
    cout << i->t1 << " " << i->t2 << " " << i->clo << endl;
  }

  set<Obj>::iterator is = myset.find(a);
  cout << is->t1 << " " << is->t2 << " " << is->clo << endl;
  set<Obj>::iterator ix = myset.find(c);
  cout << (ix == myset.end()) << " found" << endl;
}
