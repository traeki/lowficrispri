# IPython log file

get_ipython().run_line_magic('ls', '')
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('cd', '~/gd/proj/lowficrispri/docs/20171205_lib2_ind/')
get_ipython().run_line_magic('ls', '')
targets = pd.DataFrame('/home/jsh/gd/genomes/bsu_168/genoscope/bsu_168.genoscope.targets.all.tsv')
targets = pd.readcsv('/home/jsh/gd/genomes/bsu_168/genoscope/bsu_168.genoscope.targets.all.tsv')
get_ipython().set_next_input('targets = pd.read_csv');get_ipython().run_line_magic('pinfo', 'pd.read_csv')
targets = pd.read_csv('/home/jsh/gd/genomes/bsu_168/genoscope/bsu_168.genoscope.targets.all.tsv', sep='\t')
targets.TARGET
targets.head()
targets.loc[targets['#GENE'] == dfrA].head()
targets.loc[targets['#GENE'] == 'dfrA'].head()
targets.loc[targets['#GENE'] == 'murAA'].head()
targets = pd.read_csv('/home/jsh/gd/genomes/bsu_168/genoscope/bsu_168.genoscope.targets.all.vs_eco.tsv', sep='\t')
targets.head()
import Bio
from Bio import SeqIO
foo = SeqIO.parse('/home/jsh/gd/genomes/bsu_168/BSU.1.gbk', 'genbank')
foo
next(foo)
foo = SeqIO.parse('/home/jsh/gd/genomes/bsu_168/BSU.1.cleaned.gbk', 'genbank')
next(foo)
foo = SeqIO.parse('/home/jsh/gd/genomes/bsu_168/genoscope/BSU.1.cleaned.gbk', 'genbank')
next(foo)
foo = SeqIO.parse('/home/jsh/gd/genomes/bsu_168/genoscope/BSU.clean.gbk', 'genbank')
next(foo)
foo = SeqIO.parse('/home/jsh/gd/genomes/bsu_168/genoscope/BSU.clean2.gbk', 'genbank')
next(foo)
foo = SeqIO.parse('/home/jsh/gd/genomes/bsu_168/genoscope/BSU.clean.gbk', 'genbank')
next(foo)
foo = SeqIO.parse('/home/jsh/gd/genomes/bsu_168/genoscope/BSU.clean.gbk', 'genbank')
next(foo)
foo = SeqIO.parse('/home/jsh/gd/genomes/bsu_168/genoscope/BSU.clean.gbk', 'genbank')
next(foo)
get_ipython().run_line_magic('pinfo', 'SeqIO.parse')
foo = SeqIO.parse('/home/jsh/gd/genomes/bsu_168/genoscope/BSU.clean.gbk', 'genbank')
next(foo)
foo = SeqIO.parse('/home/jsh/gd/genomes/bsu_168/genoscope/BSU.clean.gbk', 'genbank')
next(foo)
foo = SeqIO.parse('/home/jsh/gd/genomes/bsu_168/genoscope/BSU.clean.gbk', 'genbank')
next(foo)
next(foo)
foo = SeqIO.parse('/home/jsh/gd/genomes/bsu_168/genoscope/BSU.clean.gbk', 'genbank')
foo = next(foo)
foo
len(foo.seq)
foo.features
foo.features[0]
bar = foo.features[0]
bar.id
foo.head()
foo[:3]
foo[2]
foo
foo.features[:3]
bar = foo.features[1]
bar.id
bar
[x.id for x in foo.features if x.type=='gene']
bar.qualifiers
bar
bar.seq
'CTTAAAGTATGCAAGATCAT' in bar.seq
'CTTAAAGTATGCAAGATCAT' in foo.seq
'CTTAATGTATGCAAGATCAT' in foo.seq
'TTCAATCATTATGGGCCGGA' in foo.seq
'TCCGGCCCATAATGATTGAA' in foo.seq
'TTCAATCATTATGGGCCGGA' in foo.seq
'AAGTTAGTAATACCCGGCCT' in foo.seq
foo.seq.find('AAGTTAGTAATACCCGGCCT')
foo.seq.find('TTCAATCATTATGGGCCGGA')
foo.seq.find('TCCGGCCCATAATGATTGAA')
foo.seq.reverse_complement.find('TTCAATCATTATGGGCCGGA')
foo.seq.reverse_complement().find('TTCAATCATTATGGGCCGGA')
foo.seq.find('TTCAATCATTATGGGCCGGA')
foo.seq.find('TCCGGCCCATAATGATTGAA')
foo.seq.find('TCCGGCCCANAATGATTGAA')
foo.seq.find('TCCGGCCCA.AATGATTGAA')
get_ipython().run_line_magic('pwd', '')
oligos = open('data/hawk1234.oligos')
oligos.readlines
oligos.readlines()
oligos = [x.strip() for x in oligos]
oligos[:5]
oligos = open('data/hawk1234.oligos')
oligos = [x.strip() for x in oligos]
oligos[:5]
oligos[:5]
oligos[:5]
len(oligos)
parents = list()
dir
dir()
foo
bsu = foo
bsuback = foo.reverse_complement()
parents.extend([x for x in oligos if x in bsu or x in bsuback])
foo = oligos[0]
foo
foo in bsu or foo in bsuback
[x in bsu or x in bsuback for x in oligos]
len(oligos)
[x in bsu or x in bsuback for x in oligos[:100]]
foo = [x in bsu or x in bsuback for x in oligos[:100]]
foo.sum
foo.count()
foo.count(True)
foo.count(False)
foo = [x in bsu or x in bsuback for x in oligos[:1000]]
foo.count(False)
foo = [x in bsu or x in bsuback for x in oligos[1000:1200]]
foo.count(False)
foo.count(True)
oligos[1000:1050]
parents = list()
children = list()
get_ipython().run_line_magic('timeit', 'oligos[0] in bsu or oligos[0] in bsuback')
get_ipython().run_line_magic('timeit', '')
oligos[0] in bsu or oligos[0] in bsuback
get_ipython().run_line_magic('timeit', '')
get_ipython().run_line_magic('help', 'timeit')
get_ipython().run_line_magic('timeit', '--help')
get_ipython().run_line_magic('pinfo', '%timeit')
get_ipython().run_line_magic('timeit', 'oligos[0] in bsu or oligos[0] in bsuback')
from collections import Counter
Counter(oligos)
counts = Counter(oligos)
sorted(counts)
sorted(counts.items())
[x for x in counts if counts[x] == 4]
foo = [x for x in counts if counts[x] == 4]
counts[foo]
counts.most_common()
get_ipython().run_line_magic('pinfo', 'counts.most_common')
counts.most_common(10)
counts[:2]
counts
next(counts)
counts.values
counts.values()
counts.keys()
get_ipython().run_line_magic('pinfo', 'counts')
parents = [x for x in counts if counts[x] > 1]
parents
len(parents)
unknown = [x for x in counts if counts[x] == 1]
len(unknown)
len(counts)
fuzzy = dict()
for p in parents:
    for i in range(len(p)):
        c = p[:i] + 'N' + p[i+1:]
        fuzzy[c] = p
        for j in range(len(p)):
            d = c[:j] + 'N' + c[j+1]
            fuzzy[d] = p
            
for p in parents:
    for i in range(len(p)):
        c = p[:i] + 'N' + p[i+1:]
        fuzzy[c] = p
        for j in range(len(p)):
            d = c[:j] + 'N' + c[j+1:]
            fuzzy[d] = p
            
len (fuzzy)
from collections import defaultdict
fuzzy = defaultdict(list)
fuzzy = defaultdict(set)
for p in parents:
    for i in range(len(p)):
        c = p[:i] + 'N' + p[i+1:]
        fuzzy[c].add(p)
        for j in range(len(p)):
            d = c[:j] + 'N' + c[j+1:]
            fuzzy[d].add(p)
            
len(fuzzy)
fuzzy
[x for x in fuzzy.values() if len(x) > 1]
'GTCTTTGCCGATAAGCCTGT' in bsu
[x for x in fuzzy.values() if len(x) > 1][:100]
[x for x in fuzzy.values() if len(x) > 1][:10]
'GTCTTTGCCGATAAGCCTGT' in bsu
[x for x in fuzzy.values() if len(x) > 1][0]
'GTGTTTGCCGATAAGCCTGT' in bsu
'GTGTTTGCCGATAAGCCTGT' in bsuback
parents = set()
for x in oligos:
    if x in bsu or x in bsuback:
        parents.add(x)
        
get_ipython().run_line_magic('ls', '')
dir
parents
len(parents)
type(oligos)
type(parents)
set(oligos)
oligos = set(oligos)
oligos - parents
len(oligos)
len(parents)
len(oligos-parents)
unknown = oligos-parents
len(unknown)
fuzzy = defaultdict(set)
for p in parents:
    for i in range(len(p)):
        c = p[:i] + 'N' + p[i+1:]
        fuzzy[c].add(p)
        for j in range(len(p)):
            d = c[:j] + 'N' + c[j+1:]
            fuzzy[d].add(p)
            
overs = [x for x in fuzzy.values() if len(x) > 1]
overs
'CAAAAATAATACTTTTTTAT' in bsu
'CAAAAATAATACTTTTTTAT' in bsuback
'CAAAATTAATACTCTTTTAT' in bsu
'CAAAATTAATACTCTTTTAT' in bsuback
fuzzy
len(fuzzy)
subs = defaultdict(set)
a = {'a', 'b'}
b = {'c', 'd'}
a.add(b)
for u in unknown:
    for i in range(len(p)):
        c = p[:i] + 'N' + p[i+1:]
        fuzzy[c].add(p)
        for j in range(len(p)):
            d = c[:j] + 'N' + c[j+1:]
            fuzzy[d].add(p)
            
fuzzy = defaultdict(set)
for p in parents:
    for i in range(len(p)):
        c = p[:i] + 'N' + p[i+1:]
        fuzzy[c].add(p)
        for j in range(len(p)):
            d = c[:j] + 'N' + c[j+1:]
            fuzzy[d].add(p)
            
for u in unknown:
    for i in range(len(u)):
        s = u[:i] + 'N' + u[i+1:]
        if s in fuzzy:
            for item in fuzzy[s]:
                subs[s].add(item)
        for j in range(len(u)):
            t = s[:j] + 'N' + s[j+1:]
            if t in fuzzy:
                for item in fuzzy[t]:
                    subs[t].add(item)
            
subs
len(subs)
len(unknown)
overs = [x for x in subs if len(subs[x]) > 1]
overs
subs[overs[0]]
subs[overs[1]]
overs[1]
subs = defaultdict(set)
for u in unknown:
    for i in range(len(u)):
        s = u[:i] + 'N' + u[i+1:]
        if s in fuzzy:
            for item in fuzzy[s]:
                subs[u].add(item)
        for j in range(len(u)):
            t = s[:j] + 'N' + s[j+1:]
            if t in fuzzy:
                for item in fuzzy[t]:
                    subs[u].add(item)
            
subs
len(subs)
overs = [x for x in subs if len(subs[x]) > 1]
overs
subs[overs[0]]
subs[overs[1]]
subs[overs[2]]
other = set()
parents
other = unknown - set(subs)
other
len(other)
controls = other
get_ipython().run_line_magic('pwd', '')
outfile = open('new_orig_map.tsv', 'w')
for x in parents:
    outfile.write('\t'.join((x,x)) + '\n')
    
for x in controls:
    outfile.write('\t'.join((x,x)) + '\n')
    
for s in subs:
    for p in subs[s]:
    _ = outfile.write('\t'.join((s,p)) + '\n')
    
for s in subs:
    for p in subs[s]:
        _ = outfile.write('\t'.join((s,p)) + '\n')
    
outfile.close()
get_ipython().run_line_magic('logstart', '')
exit()
