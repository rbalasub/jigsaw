#include <stdlib.h>
#include <time.h>
#include <iostream>

int main() {
  // srand(time(NULL));
  for (int i = 1; i <= 10; ++i) {
    std::cout << (rand() % 2) << ' ';
  }
  std::cout << std::endl;

}
