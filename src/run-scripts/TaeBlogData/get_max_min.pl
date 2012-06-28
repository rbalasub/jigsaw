$max = -1;
$min = 1000000;
while (<STDIN>) {
  chomp;
  $max = int($_) if (int($_) > $max);
  $min = int($_) if (int($_) < $min);
}

print "Max $max, Min $min\n";
