from core_function.pdf_classes.para import PTS,approxiamtion,inside
from core_function.pdf_classes.cell import Cell
import numpy as np


class Cluster:
    """
    a cluster is a collection of points
    all points in same cluster should
    be defined as 'same' to one another
    and a cluster should contain all
    same points in a page and given a
    coordinates represent the cluster
    it should be a graph data type and
    the represent value should be the
    centre of the cluster

    """
    pageCluster = {}
    pageSortedClusters = {}
    pageRoots = {}
    pageCells = {}
    starters = {}

    def __init__(self):
        self.key = (-1,-1)
        self.value = set()
        self.pageid = None
        self.id =None
        self.pointsid =set()
        self.midtargets =[]
        self.target_info =[]
        self.s1_matrix =[]
        self.rootid = None
        self.tableRoot = False
        self.rowChild = None
        self.colChild = None
        self.rowParent = None
        self.colParent = None
        self.color = 'White'
        self.byte_mark = 0

    def same_cluster(self,other):
        # compare two clusters
        if approxiamtion(self.key[0],other.key[0]) \
                and approxiamtion(self.key[1],other.key[1]):
            return True
        else:
            return False

    def get_key(self):
        return self.key

    def get_value(self):
        return self.value

        # need refine
    def is_acceptable(self, other):
        acceptable = False
        check = [other.is_same_point(i) for i in self.value]
        if all(check):
            acceptable = True
        return acceptable

    def __repr__(self):
        return ('<pageid: %s , cluster_id: %s, '
                'cluster_key: %s, number_of_points: %s ,'
                'pointsID: %s ,'
                'points:%s ,'
                'children: %s ,'
                'number_of_children = %s>\n' %
                (self.pageid, self.id, self.key,
                 len(self.value), self.pointsid,
                 [(item.x, item.y) for item in self.value],
                 [item.children for item in self.value if item.children],
                 len([item.children for item in self.value if item.children])))

    def fill_(self,point):
        if len(self.value) == 0:
            self.key = (point.x,point.y)
            self.value.update([point])
            self.pageid = point.pageid
            self.pointsid.update([point.id])
        else:
            self.value.update([point])
            self.calculate_key()
            if not all([approxiamtion(self.key[0],item.x) for item in self.value] +
                       [approxiamtion(self.key[1],item.y) for item in self.value]):
                # the new key shall be removed
                self.value.remove(point)
                self.calculate_key()
            else:
                # keep the new point, update points id
                self.getPointsID()

    def calculate_key(self):
        # calculate new key value for each time a new point added in
        self.key = (sum([i.x for i in self.value])/len(self.value),
                    sum([i.y for i in self.value])/len(self.value))
        return self.key

    def getPointsID(self):
        for id in self.value:
            self.pointsid.update([id.id])
        return self.pointsid

    def isrowChild(self):
        if self.rowParent:
            return self.rowParent.rowChild == self

    def iscolChild(self):
        if self.colParent:
            return self.colParent.colChild == self

    def isRoot(self):
        if self.s1_matrix[0]==[0,1,1,0]:
            self.tableRoot = True
        return self.tableRoot

    def set_s1_matrix(self):
        v = []

        for i in range(8):
            v.append((0, [(-1,0)]))
        for m in self.target_info:
            v[m[0] - 1] = m
        self.s1_matrix = [[int(v[i][0]!=0) for i in range(len(v))],[i[1][0] for i in v ]]
        # 这里需要把list的前4个转为二进制代码 ， 并将相对应的起始态存入dict

        return self.s1_matrix

    # need refine
    def get_middle(self):
        # connect with other clusters
        for item in self.value:
            if item.children:
                self.midtargets += item.children
                for t in item.children:
                    self.target_info.append((t[0],[n for n in t[1].parents if n not in self.pointsid]))
        print(self.target_info)
        # O(n2) max 4*N N is all points in one page
        for s in self.target_info:
            for j in self.pageSortedClusters[self.pageid]:
                if j != self:
                    for m in range(len(s[1])):
                        if s[1][m] in j.pointsid:
                            s[1][m] = (j.id,m)

        print(self.target_info)
        self.set_s1_matrix()
        return

    def net_up(self):
        if self.s1_matrix[0][0] == 1:
            self.colParent = Cluster.pageSortedClusters[self.pageid][self.s1_matrix[1][0][0]]
        if self.s1_matrix[0][1] == 1:
            self.rowChild = Cluster.pageSortedClusters[self.pageid][self.s1_matrix[1][1][0]]
        if self.s1_matrix[0][2] == 1:
            self.colChild = Cluster.pageSortedClusters[self.pageid][self.s1_matrix[1][2][0]]
        if self.s1_matrix[0][3] == 1:
            self.rowParent = Cluster.pageSortedClusters[self.pageid][self.s1_matrix[1][3][0]]

    def do_detect(self):
        return sum(self.s1_matrix[0][1:3])

    def do1(self):
        # row child judgement put outside this function
        l = Cluster.pageSortedClusters[self.pageid][self.s1_matrix[1][1][0]].id
        m = Cluster.pageSortedClusters[self.pageid][self.s1_matrix[1][1][0]].s1_matrix[1][0][0]
        if m!=-1:
            return l
        else:
            return Cluster.pageSortedClusters[self.pageid][l].do1()

    def do2(self,key):
        # row child judgement put outside this function
        l = Cluster.pageSortedClusters[self.pageid][self.s1_matrix[1][1][0]].id
        m = Cluster.pageSortedClusters[self.pageid][self.s1_matrix[1][1][0]].s1_matrix[1][2][0]
        if Cluster.pageSortedClusters[self.pageid][l].do_detect()==2:
            Cluster.starters[key].update([l])
        if m!=-1:
            return l
        else:
            return Cluster.pageSortedClusters[self.pageid][l].do2(key)

    def do3(self,key):
        # row child judgement put outside this function
        l = Cluster.pageSortedClusters[self.pageid][self.s1_matrix[1][2][0]].id
        m = Cluster.pageSortedClusters[self.pageid][self.s1_matrix[1][2][0]].s1_matrix[1][1][0]
        if Cluster.pageSortedClusters[self.pageid][l].do_detect()==2:
            Cluster.starters[key].update([l])
        if m!= -1:
            return l
        else:
            return Cluster.pageSortedClusters[self.pageid][l].do3(key)

    def do4(self):
        # row child judgement put outside this function
        l = Cluster.pageSortedClusters[self.pageid][self.s1_matrix[1][2][0]].id
        m = Cluster.pageSortedClusters[self.pageid][self.s1_matrix[1][2][0]].s1_matrix[1][3][0]
        if m!= -1:
            return l
        else:
            return Cluster.pageSortedClusters[self.pageid][l].do4()

    @classmethod
    def pageClusterisEmpty(cls, pageid):
        return pageid in cls.pageCluster.keys()

    @classmethod
    def setpageCluster(cls, pageid):
        if not cls.pageClusterisEmpty(pageid):
            cls.pageCluster[pageid] = set()
            cls.pageRoots[pageid] = set()
            cls.pageCells[pageid] = {}

    # starting point
    @classmethod
    def group_cluster(cls, pageid, cleaned_points):
        for point in cleaned_points:
            if point.pageid == pageid:
                if not cls.pageClusterisEmpty(pageid):
                    cls.setpageCluster(pageid)
                if cls.pageCluster[pageid]:
                    found = False
                    for cluster in cls.pageCluster[pageid]:
                        cluster.fill_(point)
                        if point.id in cluster.pointsid:
                            found = True
                            break
                    if not found:
                        new_cluster = Cluster()
                        new_cluster.fill_(point)
                        cls.addPagecluster(new_cluster)
                else:
                    new_cluster = Cluster()
                    new_cluster.fill_(point)
                    cls.addPagecluster(new_cluster)
        return

    @classmethod
    def addPagecluster(cls, cluster):
        if cluster.pageid in list(cls.pageCluster.keys()):
            cls.pageCluster[cluster.pageid].update([cluster])
        else:
            cls.pageCluster[cluster.pageid] = set()
            cls.pageCluster[cluster.pageid].update([cluster])

    @classmethod
    def assignid(cls, pageid):
        # O(page*n), better use a page parameter, or assign id after caching?
        for j, v in enumerate(list(cls.pageSortedClusters[pageid])):
            v.id = j
        return

    @classmethod
    def sortpageCluster(cls, pageid):
        cls.pageSortedClusters[pageid] = sorted(sorted(list(cls.pageCluster[pageid]), key=lambda x: x.key[0])
                                                , key=lambda x: round(x.key[1]), reverse=True)
        return cls.pageSortedClusters[pageid]

    @classmethod
    def locat_roots(cls, pageid):
        # use after sort clusters
        for c in cls.pageSortedClusters[pageid]:
            if c.s1_matrix[0][0:4] == [0, 1, 1, 0]:
                cls.pageRoots[pageid].update([c.id])

    @classmethod
    def do_group(cls, pageid):
        for c in cls.pageSortedClusters[pageid]:
            c.net_up()
        return

    @classmethod
    def removeHeaderCluster(cls):
        pass

    @classmethod
    def do_cell(cls,pageid):
        # f = det + do2+do3+do_compare start from root
        cls.starters = {}
        while len(cls.pageRoots[pageid])>0:
            c_root = cls.pageRoots[pageid].pop()
            cls.pageCells[pageid][c_root] = []
            # all starters comes form the c_roots
            cls.starters[c_root] = set()
            cls.starters[c_root].update([c_root])
            while len(cls.starters[c_root])>0:
               # print(len(cls.starters[c_root]))
                c_starter = cls.starters[c_root].pop()
                found = False
                if cls.pageSortedClusters[pageid][c_starter].do_detect() == 2:
                    m = cls.pageSortedClusters[pageid][c_starter].do2(c_root)
                    n = cls.pageSortedClusters[pageid][c_starter].do3(c_root)
                    k = cls.pageSortedClusters[pageid][m].do4()
                    l = cls.pageSortedClusters[pageid][n].do1()
                    if k ==l:
                        found = True
                        s = cls.pageSortedClusters[pageid][k].s1_matrix[0][0:4]
                        x ={'pageid':pageid, 'table_root': c_starter==c_root,'table_end':s ==[1,0,0,1],'root': c_starter,'rtc':m,'lbc':n,'rbc':k}
                        # set to a cell
                        cls.pageCells[pageid][c_root].append(Cell(x))
                    else:
                        print('incorrect')
                        print([('table_root', c_starter == c_root), ('root', c_starter), ('rtc', m), ('lbc', n),
                               ('rbc', k,l)])