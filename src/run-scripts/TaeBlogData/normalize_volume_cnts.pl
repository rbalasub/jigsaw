while (<STDIN>) {
  chomp;
  print int($_) * 1.0/1000, "\n";
}

