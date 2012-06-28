import sys

label_cnts = {}
group_cnts = {}

for line in sys.stdin:
  line = line.rstrip()
  node, group = line.split(',')
  if node not in label_cnts:
    label_cnts[node] = 0
  label_cnts[node] += 1
  if group not in group_cnts:
    group_cnts[group] = 0
  group_cnts[group] += 1


n_nodes = 0
n_labels = 0
hist = {}
for node, cnt in label_cnts.items():
  n_nodes += 1
  n_labels += cnt
  if cnt not in hist:
    hist[cnt] = 0
  hist[cnt] += 1

print "Average of %.2f labels per node" % (n_labels * 1.0 / n_nodes)
print "Histogram of counts"
for cnt, n in hist.items():
  print "%d labels: %d nodes" % (cnt, n)

sum = 0
for group, cnt in group_cnts.items():
  print "Group %s: %d" % (group, cnt)
  sum += cnt

print "Number of nodes = %d" % (n_nodes)
print "Sum of cluster members = %d" % (sum)

