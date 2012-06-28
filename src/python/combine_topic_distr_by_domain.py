import sys

domain_distrs = {}
t = 0
for line in sys.stdin:
  line = line.rstrip()
  (domain, distr) = line.split('')
  d = distr.split('\t')
  if domain not in domain_distrs:
    domain_distrs[domain] = [0] * len(d)
  for i in range(len(d)):
    domain_distrs[domain][i] += float(d[i])
    t += float(d[i])

for domain, distr in domain_distrs.iteritems():
#sys.stdout.write("Domain " + domain + ": ")
  sys.stdout.write(domain + "\t")
  total = 0
  for x in distr:
    sys.stdout.write(str(x) + "\t")
    total += x
#  sys.stdout.write(" Total:" + str(total) + "\n")
  sys.stdout.write("\n")

print t
