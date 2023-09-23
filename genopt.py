import sys, os, argparse
import hashlib
import random
import multiprocessing
import glob
import copy
import logging
import statistics
from io import StringIO
import functools;

logging.basicConfig(level=logging.DEBUG,
                    filename="genopt.log")
console = logging.StreamHandler()
log_stream = StringIO()
log_stream_h = logging.StreamHandler(log_stream)
console.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(console)
logging.getLogger('').addHandler(log_stream_h)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


parser = argparse.ArgumentParser()
parser.add_argument("--collate")
parser.add_argument("--mutate", nargs=2)
parser.add_argument("--mutateAll")
parser.add_argument("--buildandrun")
parser.add_argument("--buildandrunall", action='store_true')
parser.add_argument("--run")
parser.add_argument("--measure", action='store', default='instructions')
parser.add_argument("--parallel", "-j", type=int)
args = parser.parse_args()


# Generally
#  Add -Wl,-mllvm=-gen-opt-prefix=filename to all files.
# Create a baseline:
#  Build with and -Wl,-mllvm=-debug-only=getopt to get base.genopt
#  Run python3 ~/davgre01/llvm/llvm-project/genopt.py --collate base.genopt > base.go
# Build:
#  Build with -mllvm -gen-opt-filename=$1.go
#  Run python3 ~/davgre01/llvm/llvm-project/genopt.py --run base.go -j 50


def collate(file):
  f = open(file, "r")
  map = {}
  for l in f:
    if not l.startswith("+"):
      split = l.strip().split(' ')
      if len(split) == 4:
        key = split[0] +' '+split[1]+' '+split[2]
        if key not in map:
          map[key] = []
        map[key].append(split[3])

  for k in sorted(map.keys()):
    print(k + " " + " ".join(map[k]))

if args.collate:
  collate(args.collate)


def openF(file):
  f = open(file, "r")
  D = {}
  for l in f:
    split = l.strip().split(' ')
    assert(len(split) > 3)
    D[split[0] + ' ' + split[1] + ' ' + split[2]] = split[3:]
  return D

def writeF(D):
  S = ""
  for d in sorted(D.keys()):
    S += d + ' ' + ' '.join(D[d]) + '\n'
  h = hashlib.sha1(S.encode('ASCII')).hexdigest()
  f = open('genopt/' + h + '.go', "w")
  f.write(S)
  f.close()
  return h


def mutateOne(key, v):
  g = key.split(" ")[1]
  if g == "LoopVectorizer":
    v = int(v)
    VF = (v & 0xff000000) >> 24
    IC = (v & 0xff0000) >> 16
    EF = (v & 0xff00) >> 8
    VFs = v & 0x4
    EFs = v & 0x2
    TF = v & 0x1
    while True:
      r = random.randrange(9)
      if (r == 1 and VF == 0) or (r == 3 and IC == 0) or (r == 5 and EF == 0):
        continue
      break
    if r == 0:
      VF += 1
    elif r == 1:
      VF -= 1
    elif r == 2:
      IC += 1
    elif r == 3:
      IC -= 1
    elif r == 4:
      EF += 1
    elif r == 5:
      EF -= 1
    elif r == 6:
      VFs = not VFs
    elif r == 7:
      EFs = not EFs
    elif r == 8:
      TF = not TF
    return str(VF << 24 | IC << 16 | EF << 8 | (0x4 if VFs else 0) | (0x2 if EFs else 0) | (0x1 if TF else 0))
  elif g == "FuncSpec" or g == "Inline":
    return "1" if v == "0" else "0"
  elif g == "LoopUnroll":
    if int(v) == 0:
      return "2"
    else:
      return str(random.choice([int(v)*2, int(v)//2]))
  assert(False)

def mutateOneSimple(key, v):
  g = key.split(" ")[1]
  if g == "LoopVectorizer":
    v = int(v)
    VF = (v & 0xff000000) >> 24
    IC = (v & 0xff0000) >> 16
    EF = (v & 0xff00) >> 8
    VFs = v & 0x4
    EFs = v & 0x2
    TF = v & 0x1
    if VF == 0:
      VF = 3
    else:
      VF = 0
      IC = 0
      VFs = 0
    return str(VF << 24 | IC << 16 | EF << 8 | (0x4 if VFs else 0) | (0x2 if EFs else 0) | (0x1 if TF else 0))
  elif g == "FuncSpec" or g == "Inline":
    return "1" if v == "0" else "0"
  elif g == "LoopUnroll":
    if int(v) == 0:
      return "4"
    else:
      return "0"
  assert(False)


def mutateAt(D, idx, mutate=mutateOne):
  d = 0
  keys = list(D.keys())
  while idx >= len(D[keys[d]]):
    idx -= len(D[keys[d]])
    d += 1
  E = copy.deepcopy(D)
  E[keys[d]][idx] = mutate(keys[d], D[keys[d]][idx])
  return E

def combine(D, E):
  keys = list(set(D.keys()) | set(E.keys()))
  F = {}
  for k in keys:
    Ds = D[k] if k in D else []
    Es = E[k] if k in E else []
    Fs = []
    for i in range(min(len(Ds), len(Es))):
      Fs.append(random.choice([Ds[i], Es[i]]))
    for i in range(len(Fs), len(Ds)):
      Fs.append(Ds[i])
    for i in range(len(Fs), len(Es)):
      Fs.append(Es[i])
    F[k] = Fs
  return F

def combineAndMutate(A, B):
  D = openF("genopt/"+A+".go")
  E = openF("genopt/"+B+".go")
  totalSizeD = sum(len(D[d]) for d in D.keys())
  totalSizeE = sum(len(E[d]) for d in E.keys())
  F = combine(D, E)
  maxE = max(totalSizeD, totalSizeE)
  for i in range(maxE):
    if random.randrange(maxE) == 0:
      F = mutateAt(F, i)
  sha = writeF(F)
  return sha

def loadResult(sha):
  if args.measure != "codesize":
    f = open("genopt/"+sha+".log", "r")
    for l in f:
      if args.measure in l:
        return float(l.strip().split(' ')[0].replace(',',''))
    raise ValueError("Didn't find instructions")
  else:
    return os.path.getsize("genopt/"+sha+".exe")

def buildAndRun(sha):
  Done = False
  while not Done:
    Done = True
    output = os.popen("bash ./build_genopt.sh " + sha + " 2>&1").read()
    print(output)
    if "GenOptNotFound" in output:
      Done = False
      D = openF("genopt/"+sha+".go")
      for l in output.split('\n'):
        if l.startswith("GenOptNotFound:"):
          split = l.strip().split(' ')
          key = split[1] + ' ' + split[2] + ' ' + split[3]
          if key not in D:
            D[key] = []
          D[key] += ["0"]
      sha = writeF(D)
      logging.debug(f"new sha: {sha}")
  os.system("bash ./run_genopt.sh " + sha)
  return sha

# Build and run if it the log file isn't already present
def buildAndRunIfNeeded(sha):
  try:
    return (sha, loadResult(sha))
  except (ValueError, OSError):
    sha = buildAndRun(sha)
    try:
      return (sha, loadResult(sha))
    except:
      return None


# Mutate a single entry from file arg0, at index arg1.
if args.mutate:
  D = openF(args.mutate[0])
  totalSize = sum(len(D[d]) for d in D.keys())
  if int(args.mutate[1]) < 0:
    args.mutate[1] = random.randrange(totalSize)
  assert(int(args.mutate[1]) < totalSize)
  D = mutateAt(D, int(args.mutate[1]))
  sha = writeF(D)
  (sha, s) = buildAndRunIfNeeded(sha)
  if s:
    logging.debug(f"Score of {sha} is {s}")

def runner(D, Idx):
  E = mutateAt(D, Idx, mutateOneSimple)
  sha = writeF(E)
  (sha, s) = buildAndRunIfNeeded(sha)
  logging.debug(sha + ": " + str(s))
  return (sha, s)
def mutateAll(basefile):
  D = openF(basefile)
  totalSize = sum(len(D[d]) for d in D.keys())
  logging.debug(f"Running {totalSize} mutations")
  pool = multiprocessing.Pool(args.parallel)
  R = pool.map(functools.partial(runner, D), range(totalSize))
  S = {}
  for s in R:
    if s[1]:
      S[s[0]] = s[1]
  return S
if args.mutateAll:
  mutateAll(args.mutateAll)



if args.buildandrun:
  sha = buildAndRun(args.buildandrun)
  logging.debug(loadResult(sha))
if args.buildandrunall:
  fns = sorted([fn[:-3] for fn in os.listdir("genopt") if (fn.endswith(".go") and not os.path.exists("genopt/"+fn[:-2]+"log"))])
  logging.debug(f"Running {len(fns)} hashes")
  pool = multiprocessing.Pool(args.parallel)
  pool.map(buildAndRun, fns)


def loadAllResults():
  S = {}
  best = 0
  worst = 0
  for fn in os.listdir("genopt"):
    if fn.endswith(".go"):
      H = fn[:-3]
      try:
        S[H] = loadResult(H)
      except ValueError:
        continue
      except FileNotFoundError:
        continue
      logging.debug(H + " = " + str(S[H]))
      if best == 0 or S[H] < best:
        best = S[H]
      if worst == 0 or S[H] > worst:
        worst = S[H]
  return S, best, worst

def pickOne(scores, best, worst):
  while True:
    r = random.randrange(len(scores))
    if not scores[r]:
      continue
    h = scores[r][0]
    s = scores[r][1]
    if not s:
      continue
    de = 1 if worst==best else worst-best
    p = pow(1 - (s - best)/de, 3)
    if random.random() < p:
      return h

if args.run:
  #L, best, worst = loadAllResults()
  #if best < worst * 2:
  #  worst = best * 2
  #logging.debug("Loaded " + str(len(L)) + " results, with a best score of " + str(best))
  #if len(L) == 0:
  #  logging.debug("No valid results found")
  #  sys.exit(-1)

  #def runner(Id):
  #  global L, best, worst
  #  while True:
  #    a = pickOne(L, best, worst)
  #    b = pickOne(L, best, worst)
  #    logging.debug(f"Picking {a} and {b}")
  #    c = combineAndMutate(a, b)
  #    logging.debug(f"  combined to {c}")
  #    score = buildAndRunIfNeeded(c)
  #    if not score:
  #      logging.debug(f"Failed to build/run {c}")
  #      continue
  #    logging.debug(f"New score of {c} is {score}")
  #    L[c] = score
  #    if L[c] < best:
  #      logging.debug(f"  new best!")
  #      best = L[c]

  def updateGeneration(scores):
    gen = [s[0] for s in scores if s]
    ss = [s[1] for s in scores if s]
    best = min(ss)
    worst = max(ss)
    ngen = []
    while len(ngen) < len(scores):
      a = pickOne(scores, best, worst)
      b = pickOne(scores, best, worst)
      c = combineAndMutate(a, b)
      if c not in ngen:
        ngen += [c]
    return ngen

  #baseshas = mutateAll(args.run)
  #logging.debug(f"Starting from {len(baseshas)} keys, with a best of {min(baseshas.values())}")
  #gen = sorted(baseshas, key=baseshas.get)[:50]
  gen = [args.run]*50
  pool = multiprocessing.Pool(args.parallel)
  while True:
    logging.debug(f"Current generation: {gen}")
    scores = pool.map(buildAndRunIfNeeded, gen)
    logging.debug(f"  scores: {scores}")
    logging.debug(f"  best score: {min([s[1] for s in scores if s])}, avg score: {statistics.mean([s[1] for s in scores if s])}")
    gen = updateGeneration(scores)
