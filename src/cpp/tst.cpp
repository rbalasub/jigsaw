#include <iostream>
#include <cfloat>
using namespace std;
int main() {
  double d;
  // cin >> d;
  // cout << "Echo " << d << endl;

  d = 3.4e-420;

  cout << LDBL_MAX_10_EXP << endl;
  cout << d << endl;

  int a;
  cout << sizeof(a) << endl;
  long int b;
  cout << sizeof(b) << endl;
  int *c, **e;
  cout << sizeof(c) << endl;
  cout << sizeof(e) << endl;

}

