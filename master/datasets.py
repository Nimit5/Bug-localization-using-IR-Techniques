from collections import namedtuple
from pathlib import Path

# Dataset and results root directory
_DATASET_ROOT = Path(__file__).parent / '../data'
RESULTS_ROOT = Path(__file__).parent / '../results'
RESULTS_ROOT.mkdir(exist_ok=True)

Dataset = namedtuple('Dataset', ['name', 'root', 'src', 'bug_repo'])

# Current dataset in use. (change this name to change the dataset)
txt = input("Choose the Dataset 1) codec 2) zxing 3) swarm 4) weaver 5) swt: ")
first = ""
second = ""
if txt == 'zxing':
    stuff = txt[0:2].upper()
    stuff += txt[2:]
    # print(stuff)
    first = stuff+'/'+stuff+'-1.6'
    second = stuff+'/'+stuff+'BugRepository.xml'
    obj = Dataset(
        txt,
        _DATASET_ROOT / stuff,
        _DATASET_ROOT / first,
        _DATASET_ROOT / second
    )
elif txt == 'swt':
    stuff = txt[0:3].upper()
    stuff += txt[3:]
    # print(stuff)
    first = stuff+'/'+stuff+'-3.1'
    second = stuff+'/'+stuff+'BugRepository.xml'
    obj = Dataset(
        txt,
        _DATASET_ROOT / txt.upper(),
        _DATASET_ROOT / first,
        _DATASET_ROOT / second
    )
else:

    first = txt.upper()+'/gitrepo'
    second = txt.upper()+'/bugrepo/repository.xml'
    obj = Dataset(
        txt,
        _DATASET_ROOT / txt.upper(),
        _DATASET_ROOT / first,
        _DATASET_ROOT / second,
    )
# print(obj)
DATASET = obj
print(DATASET)


if __name__ == '__main__':
    print(DATASET.name, DATASET.root, DATASET.src, DATASET.bug_repo)
