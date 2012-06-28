while (<STDIN>) {
  chomp;
  $c = `cat $_ | wc -l`;
  print "$c";
}
