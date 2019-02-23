import os
import re



def lprint(l):
    for elem in l:
        print(elem)

def find_pardir(name):
    abspath = os.path.abspath(__file__)
    nlen = len(name)
    while abspath[-nlen:] != name:
        abspath = os.path.dirname(abspath)
    return abspath

def get_subdir(fpath, subdir):
    return os.path.join(fpath, subdir)

def recursive_pathsplit(filename, tagindex):
    tags = []
    filename = filename[tagindex:]
    while "\\" in filename or "/" in filename:
        filename, t = os.path.split(filename)
        # print(filename)
        # input()

        tags.append(t)
    tags.append(filename)
    return tags


def parse_tags(filename, tagindex):
    return (tuple(recursive_pathsplit(filename, tagindex)), filename)


class IDparser:
    def __init__(self):
        self.set_file_list()
        self.set_ftags()
        self.set_tagsdict()


    def iter_change(self):
        self.show_commondict()
        tags = input("enter tags:").split(" ")
        self.make_commondict(tags)

    def show_commondict(self):
        print(self.resdict.keys())

    def set_commondict(self, tags):
        result_sets = [self.tagsdict.get(tag) for tag in tags]
        print(tags, len(result_sets))
        try:
            commonset = set.intersection(*result_sets)
        except TypeError:
            print("invalid argument, given tag (" + " ".join(tags) + ") not in dictionary of index")
            return
        resdict = {}
        for elem in commonset:
            for tag in elem[0]:
                try:
                    resdict[tag].add(elem)
                except KeyError:
                    resdict[tag] = set()
                    resdict[tag].add(elem)
        return resdict


    def set_ftags(self):
        tagindex = self.files[0].index("results")
        self.ftags = [parse_tags(fn, tagindex) for fn in self.files]
        print("ftags", self.ftags[0][0])

    def set_file_list(self):
        sdname = get_subdir(find_pardir("Paper_netlists"), "results")
        files = []
        for fname in os.walk(sdname):
            if fname[2]:
                for n in fname[2]:
                    files.append(os.path.join(fname[0], n))
        self.files = files

    def set_tagsdict(self):
        self.tagsdict = {}
        for ftag in self.ftags:
            for tag in ftag[0]:
                try:
                    self.tagsdict[tag].add(ftag)
                except KeyError:
                    self.tagsdict[tag] = set()
                    self.tagsdict[tag].add(ftag)
        print(self.tagsdict.keys())



if __name__ == '__main__':
    idp = IDparser()
    while True:
        idp.iter_change()
        idp.show_commondict()
