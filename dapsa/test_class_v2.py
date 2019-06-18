# this file define point, cluster, cell, row, column and table
import re
import numpy as np


PTS = 0.2

def approxiamtion(xc,x0,width=3,pts = PTS):
    return abs(xc-x0) < width *(1+pts)

def inside(obj1,obj2,width =3, pts =0.2):
    matrix = np.array(obj1.bbox)>np.array(obj2.bbox)
    distinct = np.array(obj1.bbox)>(np.array(obj2.bbox)+
            np.array((width*(1+pts),width*(1+pts),0-width*(1+pts),0-width*(1+pts))))
    if not any(matrix[0:2]) and all(matrix[2:]):
        return True
    else:
        if not any(distinct[0:2]) and all(distinct[2:]):
            return True
        else:
            return False



class Point:
    """
    a child Point represent the middle point of
    two points from a LTCurve, it is the deep
    tag indicate that if two lines belong to a
    table

    """
    pagePoints = {}
    pageMiddlePoints = {}

    def __init__(self,x,y,pageid,id,line_width= 3,ischild = False):
        self.x = x
        self.y = y
        self.pageid = pageid
        self.id = id
        self.line_width = line_width
        self.ischild = ischild
        self.parents = []
        self.children = []

    def getpointid(self):
        return self.id

    def assignchild(self,child):
        """
         id   location
         1     up
         2     right
         3     down
         4     left
         5     right up
         6     right down
         7     left down
         8     left up
        """
        def defineDirection(x, y, child, linewidth):
            if approxiamtion(x,child.x) and not approxiamtion(y,child.y):
                if child.y - y > linewidth:
                    return 1
                elif child.y - y < -linewidth:
                    return 3
            elif not approxiamtion(x,child.x) and not approxiamtion(y,child.y):
                if child.x - x >linewidth:
                    if child.y -y >linewidth:
                        return 5
                    elif child.y -y <-linewidth:
                        return 6
                elif child.x - x < -linewidth:
                    if child.y -y < -linewidth:
                        return 7
                    elif child.y-y > linewidth:
                        return 8
            elif not approxiamtion(x,child.x) and approxiamtion(y,child.y):
                if child.x -x >linewidth:
                    return 2
                elif child.x-x <-linewidth:
                    return 4
        assert child.ischild

        if self.is_same_point(child):
            child.parents.append(-999)
            return
        else:
            pipid = defineDirection(self.x,self.y,child,linewidth=self.line_width*(1+PTS))
            child.parents.append(self.id)
            self.children.append((pipid,child))
        return

    def is_same_point(self,other,pts=0.2):
        dist = self.get_dist(other)
        if dist < (1+pts)* self.line_width:
            return True
        else:
            return False

    def get_dist(self,other):
        dist = np.sqrt(np.square(abs(self.x - other.x)) + np.square(abs(self.y - other.y)))
        return dist

    def is_starter(self):
        pass

    def __repr__(self):
        return ('<pageid: %s , id: %s, coordinates: (%s %s), childpoint: %s , parents: %s >' %
                (self.pageid,self.id,self.x,self.y,self.ischild,self.parents))

    @classmethod
    def setpage_Points(cls,pageid):
        cls.pagePoints[pageid] = []
        cls.pageMiddlePoints[pageid] =[]

    def addTopagePoints(self,pageid):
        if not self.ischild:
            Point.pagePoints[pageid].append(self)
        else:
            if not all([id==-999 for id in self.parents]):
                Point.pageMiddlePoints[pageid].append(self)

    @classmethod
    def cleanPoints(cls,pageid,textboxlist):
        def checkpointsvaliadation(point_list, textboxlist):
            # remove all points in a textbox e.g a html address with bottom line
            valid_points = []
            unvalid_points = []
            for point in point_list:
                unvalid = False
                for t in textboxlist:
                    if all([point.x >= t.bbox[0], point.x <= t.bbox[2],
                            point.y >= t.bbox[1], point.y <= t.bbox[3]]):
                        unvalid = True
                if unvalid:
                    unvalid_points.append(point)
                else:
                    valid_points.append(point)
            return valid_points
        cls.pagePoints[pageid] = checkpointsvaliadation(cls.pagePoints[pageid],textboxlist=textboxlist)
        return cls.pagePoints[pageid]


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
                        cls.pageCells[pageid][c_root].append(aCell(x))
                    else:
                        print('incorrect')
                        print([('table_root', c_starter == c_root), ('root', c_starter), ('rtc', m), ('lbc', n),
                               ('rbc', k,l)])


class aCell:
    starters ={}
    pageCells = Cluster.pageCells
    # {pageid : {table_root:[cells]}}

    def __init__(self,matrix):
        self.matrix = matrix
        self.pageid = matrix['pageid']
        self.id = None
        self._height = -1
        self._width = -1
        self._upper_boundary = -1
        self._lower_boundary = -1
        self._left_boundary = -1
        self._right_boundary = -1
        self._centre = (-1,-1)
        self._bbox = None
        self._text = []

    def __call__(self, *args, **kwargs):
        self.upper_boundary = self.matrix
        self.lower_boundary = self.matrix
        self.left_boundary = self.matrix
        self.right_boundary = self.matrix
        self.height = (self._upper_boundary-self._lower_boundary)
        self.width = (self._right_boundary-self._left_boundary)
        self.centre = [self._left_boundary,self._right_boundary,self._lower_boundary,self._upper_boundary]
        self.bbox = (self._left_boundary,self._lower_boundary,self._right_boundary,self._upper_boundary)

    @property
    def upper_boundary(self):
        return self._upper_boundary

    @upper_boundary.setter
    def upper_boundary(self,matrix):
        self._upper_boundary = round(max(Cluster.pageSortedClusters[self.pageid][matrix['root']].key[1],
                                         Cluster.pageSortedClusters[self.pageid][matrix['rtc']].key[1]),2)

    @property
    def lower_boundary(self):
        return self._lower_boundary

    @lower_boundary.setter
    def lower_boundary(self,matrix):
        self._lower_boundary = round(min(Cluster.pageSortedClusters[self.pageid][matrix['lbc']].key[1],
                                         Cluster.pageSortedClusters[self.pageid][matrix['rbc']].key[1]),2)

    @property
    def left_boundary(self):
        return self._left_boundary

    @left_boundary.setter
    def left_boundary(self,matrix):
        self._left_boundary = round(min(Cluster.pageSortedClusters[self.pageid][matrix['lbc']].key[0],
                                        Cluster.pageSortedClusters[self.pageid][matrix['root']].key[0]),2)

    @property
    def right_boundary(self):
        return self._right_boundary

    @right_boundary.setter
    def right_boundary(self,matrix):
        self._right_boundary = round(max(Cluster.pageSortedClusters[self.pageid][matrix['rtc']].key[0],
                                         Cluster.pageSortedClusters[self.pageid][matrix['rbc']].key[0]),2)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self,h):
        self._height= round(h,2)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self,w):
        self._width = round(w,2)

    @property
    def centre(self):
        return self._centre

    @centre.setter
    def centre(self,value):
        self._centre = (round((value[1]-value[0])/2 + value[0],2),round((value[3]-value[2])/2+value[2],2))

    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self,t):
        self._bbox = t

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self,t):
        self._text += [t]

    def __repr__(self):
        return ('<pageid: %s , cell_id: %s, '
                'matrix: %s ,'
                'bbox: %s ,'
                'centre: %s >\n'
                %(self.pageid,self.id,self.matrix,self._bbox,self._centre))

    @classmethod
    def AssignText(cls,pageid,textlist):
        while len(textlist)>0:
            current = textlist.pop(0)
            for v in cls.pageCells[pageid].values():
                for c in v:
                    if inside(c,current):
                        c.text = current.get_text()

    @classmethod
    def check_text(cls,pageid):
        for v in cls.pageCells[pageid].values():
            for c in v:
                print("the current cell {} contains text {} ".format(c.bbox,c.text))

    @classmethod
    def __init_subclass__(cls, **kwargs):
        pass


class Cell:
    """ I tried inherit cell from LTTextbox,
    but it has been proven a bad idea,
    however if you have a better way to
    utilise LTTextbox to build this obj,
    please let me know
    jacky.jingyi@outlook.com

    this Obj is a graphical area in a pdf plane,
    a cell should constructed by 4 clusters
    ( for some bad formed pdf tables, such
    as the table shows as below it might not
    be valid theoretically)
    and no other points or clusters should
    located within a cell Obj
    _________
        |
    ____|____

    attr:
        cluster_collection: （left_top_cluster,
                             left_bottom_cluster,
                             right_top_cluster,
                             right_bottom_cluster]
        cell_identification (page, cell centre coordinates ,
                            cell upper boundary , cell lower boundary,
                            cell left boundary, cell right boundary,
                            cell_table_id ,cell_row_id , cell_column_id)
    """
    pageCells = []
    def __init__(self, anchor, width=3):
        self.anchor = anchor
        self.width = width
        self.cluster_collection ={'left_top_cluster': None,
                          'left_bottom_cluster': None,
                          'right_top_cluster': None,
                          'right_bottom_cluster': None
                          }
        self.parent_row = None
        self.parent_column = None
        self.identification = {
            'pageid': None,
            'centre_coordinates': None,   # (x,y)
            'cell_upper_boundary': None,   # is a approximate line computed by top clusters
            'cell_lower_boundary': None,
            'cell_left_boundary': None,
            'cell_right_boundary':None,
            'cell_width':None,
            'cell_height':None,
            'cell_table_id': 0,     # default zero for only one page in a table
            'cell_row_id': 0,  # default zero represent the first row in a table
            'cell_column_id': 0  # default zero represent the first (the leftest) column in a table
            }
        self.cluster_collection['left_top_cluster'] = self.anchor
        self.text = []

    def textbelongto(self,lttextbox):
        """双保险是边界和中心点都查看"""
        x0 = lttextbox.bbox[0]
        y0 = lttextbox.bbox[1]
        x1 = lttextbox.bbox[2]
        y1 = lttextbox.bbox[3]
        xc = (x1 - x0)/2 + x0
        yc = (y1 - y0)/2 + y0
        # maching_rate = 0
        matchcell = False
        if all([xc > self.identification['cell_left_boundary'],
                xc < self.identification['cell_right_boundary'],
                yc > self.identification['cell_lower_boundary'],
                yc < self.identification['cell_upper_boundary']]):
            matchcell =True
        return matchcell

    def set_text(self,lttextbox):
        if self.textbelongto(lttextbox):
            self.text.append(lttextbox)
            self.text = self.sorttext()
            #print("successfully put {} in  {}".format(lttextbox,self.identification['centre_coordinates']))


    def sorttext(self):
        def sortkey(t):
            return t.bbox[1]
        return sorted(self.text,key=sortkey,reverse=True)

    def get_text(self):
        result = ''
        if len(self.text)>0:
            for text in self.text:
                result += text.get_text()
        return result

    def lbc_update(self,cls):
        xc = cls.key[0]
        yc = cls.key[1]
        update = False
        if self.cluster_collection['left_bottom_cluster']:
            original_value = [self.cluster_collection['left_bottom_cluster'].key[0],
                              self.cluster_collection['left_bottom_cluster'].key[1]]
            diff = [xc - original_value[0], yc - original_value[1]]
            if diff[1] < 0 and approxiamtion(xc,self.anchor.key[0]):
                # update
                self.cluster_collection['left_bottom_cluster'] = cls
                update = True
               # print('updating lbc ({}, {})'.format(xc,yc))
            else:
               # print('lbc ({} {}) not belong to this cell'.format(cls.key[0],cls.key[1]))
                pass
        else:
            if approxiamtion(self.anchor.key[0],xc):
                self.cluster_collection['left_bottom_cluster'] = cls
                update = True
                #print('updating lbc ({} -{})'.format(cls.key[0], cls.key[1]))

        return update

    def set_row(self):
        if self.parent_row:
            self.identification['cell_row_id'] = self.parent_row.rowid

    def rtc_update(self,cls,pts = 0.2):
        xc = cls.key[0]
        yc = cls.key[1]
        update = False
        if self.cluster_collection['right_top_cluster']!=None:
            original_value = [self.cluster_collection['right_top_cluster'].key[0],
                              self.cluster_collection['right_top_cluster'].key[1]]
            diff = [xc - original_value[0], yc - original_value[1]]
            #print('diff is {} {}'.format(diff[0],diff[1]))
            if diff[0] < -self.width * (1 + pts) and abs(diff[1])<self.width *(1 + pts):
                # update
                self.cluster_collection['right_top_cluster'] = cls
                update = True
            else:
                pass
                #print('rtc ({} {}) not belong to this cell'.format(cls.key[0],cls.key[1]))
        else:
            if approxiamtion(self.anchor.key[1],yc):
                self.cluster_collection['right_top_cluster'] = cls
                #print('updating rtc ({} -{})'.format(cls.key[0],cls.key[1]))
                update =True
        return update

    def rbc_update(self,cls,pts=0.2):
        """check if this is the rbc based on
        1. the current rbc is exist
        2. other two corner must exist """

        xc = cls.key[0]
        yc = cls.key[1]
        update = False
        if self.cluster_collection['right_bottom_cluster']:
            rbc = self.cluster_collection['right_bottom_cluster'].key
            if all([approxiamtion(xc,rbc[0]),approxiamtion(yc,rbc[1])]):
                print('same cluster')
        else:
            if self.cluster_collection['right_top_cluster'] and self.cluster_collection['left_bottom_cluster']:
                rtc = self.cluster_collection['right_top_cluster'].key
                lbc = self.cluster_collection['left_bottom_cluster'].key
                if approxiamtion(xc,rtc[0]) and approxiamtion(yc,lbc[1]):
                    # update this cluster
                    self.cluster_collection['right_bottom_cluster'] = cls
                    #print('updating rbc ({}-{})'.format(cls.key[0],cls.key[1]))
                    update = True
        return update

    def set_center(self):
        if all([isinstance(i,Cluster) for i in list(self.cluster_collection.values())]):
            self.identification['cell_left_boundary'] = (self.anchor.key[0] +
                                                            self.cluster_collection['left_bottom_cluster'].key[0])/2
            self.identification['cell_right_boundary'] = (self.cluster_collection['right_top_cluster'].key[0]+
                                                             self.cluster_collection['right_bottom_cluster'].key[0])/2
            self.identification['cell_upper_boundary'] = (self.anchor.key[1]+
                                                             self.cluster_collection['left_top_cluster'].key[1])/2
            self.identification['cell_lower_boundary'] = (self.cluster_collection['left_bottom_cluster'].key[1]+
                                                             self.cluster_collection['right_bottom_cluster'].key[1])/2
            self.identification['cell_height'] = self.identification['cell_upper_boundary'] -\
                                                 self.identification['cell_lower_boundary']
            self.identification['cell_width'] = self.identification['cell_right_boundary'] -\
                                                self.identification['cell_left_boundary']
            self.identification['centre_coordinates'] = (self.identification['cell_left_boundary'] +
                                                         self.identification['cell_width']/2,
                                                         self.identification['cell_lower_boundary'] +
                                                         self.identification['cell_height']/2)
            return self.identification
        else:
            temp = []
            for i in self.cluster_collection.items():
                if not i[1]:
                    temp.append(i[0])
            print('need more information {}'.format(temp))

    def build(self,cluster,pts=0.2):
        """fill in a cluster if it belong to this cell
         1. locate its position
         2. if it is a right bottom cluster, then fill it  *not sure yet
         需要refine"""
        #print('building for ({}-{})'.format(self.anchor.key[0],self.anchor.key[1]))
        if not self.anchor:
            print('need an anchor')
        else:
            x0 = self.anchor.key[0]
            y0 = self.anchor.key[1]
            xc = cluster.key[0]
            yc = cluster.key[1]
            output = [approxiamtion(x0,xc),approxiamtion(y0,yc)]
            if all(output):
                #print('same cluster with anchor')
                return cluster
            else:
                if any(output):
                    #if output.index(True) == 0:  # when x0 == xc they are in the same vertical line
                    true_id = output.index(True)     # 0 => 'lbc' 1 || 1 => 'rtc' 0
                    if true_id == 0:
                        self.lbc_update(cls=cluster)
                    elif true_id ==1:
                        self.rtc_update(cls=cluster)
                else:  # not matching at all
                    self.rbc_update(cluster,pts)

    def isvalid(self):
        return self.anchor != self.cluster_collection['left_top_cluster'] and self.anchor!=self.cluster_collection['right_bottom_cluster'] \
            and self.anchor!=self.cluster_collection['right_top_cluster']

    def get_centre(self):
        pass

    def clear(self):
        pass


class aRow:

    def __init__(self,bbox):
        self.bbox = bbox
        self.id = None



        pass
    pass








class Row:
    """redesign row based on cell Obj
     a row should contains all cells with same or approximately same center y coordinates
     and of course once a cell is put into a row , the row id should be written back to
     cell' s identification
     a. a row is a container with unique row id
     b. it contains all cells in the row ( or address in cpu)
     c. how to build an unqiue id about the row?
     Note: a row should have the longest length(or width) in the table , which means it may contains merged cells
     attr : {
            row_id :  a row should have an unique id for easy identification
            upper_boundary : approximately equal to all cell's upper boundary in the row
            lower_boundary : approximately equal to all cells' lower boundary in the row
            row_mid: approximately equal to all cells' center y value in the row, this would be the major
                    attribute to identify whether a cell is belong to a row
            cell_amount: how many cells in this row
            is_header: True if the row is the first row in selection range of a page False otherwise
            is_footer: True if the row is on the bottom of the table False otherwise
            sort : sort cells by their x value
            get_text: return a json or dictionary    & this may be more appropriate if put onto table Obj
            }
     """
    pageRow = []
    def __init__(self,pts = 0.2,max_width = 558):
        self.rowid = None
        self.max_width = max_width
        self.cells = set()
        self.pts = pts
        self.centre_y_coordinates = None
        self.top_boundary = None
        self.lower_boundary = None
        self.height = None
        self.width = 0
        #self.upper_boundary = 0
        #self.lower_boundary = 0
        #self.rowmid = 0
        self.cell_amount = 0
        self.cells_list = []
        #self.is_header = False
        #self.is_footer = False

    def get_center_ymid(self):
        return self.centre_y_coordinates

    def assign_rowid(self,pageid,tableid,rowletter):
        """
        # generate an unique id
        # easy to locate a row from pdf file level,
        # a row letter is an int
          A1 represent the first cell (first left) in a row
        # deal with merge cells in next version
        """
        self.rowid =(pageid,tableid,rowletter)
        return self.rowid

    def get_rowid(self):
        if self.rowid:
            return self.rowid

    def set_row(self,cell):
        # this is for init a row
        # waiting for judgement, will do when i happy
        if len(self.cells) == 0:
            # init a row
            # xmid = cell.identification['centre_coordinates'][0]
            ymid = cell.identification['centre_coordinates'][1]
            tb = cell.identification['cell_upper_boundary']
            lb = cell.identification['cell_lower_boundary']
            rib = cell.identification['cell_right_boundary']
            leb = cell.identification['cell_left_boundary']
            height = cell.identification['cell_height']
            self.lower_boundary = lb
            self.top_boundary = tb
            self.height = height
            self.centre_y_coordinates = ymid
            self.cells.update([cell])
            self.width = rib - leb
            self.cell_amount += 1
        else:
            self.fill_cell(cell)

    def isbelong(self,cell):
        # this is different from fill_cell judging method
        # for some merged cells , the center may shifting from a well formed cell
        #xmid = cell.identification['centre_coordinates'][0]
        ymid = cell.identification['centre_coordinates'][1]
        tb = cell.identification['cell_upper_boundary']
        lb = cell.identification['cell_lower_boundary']
        #rib = cell.identification['cell_right_boundary']
        #leb = cell.identification['cell_left_boundary']
        result = [(tb <= self.top_boundary or approxiamtion(tb,self.top_boundary)),
                  (lb >= self.lower_boundary or approxiamtion(lb,self.lower_boundary)),
                  (ymid < self.top_boundary and ymid > self.lower_boundary)
                  ]
        #print("cell.key is {} fill into {}".format(cell.identification['centre_coordinates'],self.centre_y_coordinates ))
        #print("is belong to result is {}".format(result))
        return all(result)

    def check_width(self):
        return approxiamtion(self.width,self.max_width,width=4,pts=0.4)

    def __contains__(self, item):
        return True if item in self.cells else False

    def fill_cell(self,cell):
        # currently not writing a judgement
        """ is belong, but with those scenarios:
            [sameheight?, same top boundary?, same lower boundary
            [True, False, False]     _________________
               a cell like c        |___a___|         |
                                    |___c___|         |
                                    |___b___|_________|

            [False,True,False] like cell a on the top

            [False,False,True] like cell b on the top
                                                                                    this is a machine gun
            [False,False,False] Impossible , I hope. if this scenario happens      ||======================{:}
                                the guy who create the pdf file should             {||||}         {|}
                                be hang on a cross.                                {}             {|}
        """
        added = False
        if self.cell_amount ==0:
            return self.set_row(cell)
        else:
            if self.isbelong(cell):
                # xmid = cell.identification['centre_coordinates'][0]
                ymid = cell.identification['centre_coordinates'][1]
                tb = cell.identification['cell_upper_boundary']
                lb = cell.identification['cell_lower_boundary']
                rib = cell.identification['cell_right_boundary']
                leb = cell.identification['cell_left_boundary']
                results = [approxiamtion(ymid,self.centre_y_coordinates),
                           approxiamtion(tb,self.top_boundary),
                           approxiamtion(lb,self.lower_boundary)]
                if all(results):   # scenario b
                    # a well formed cell without merging
                    # it wont do a shit if you keep adding same cell into the row
                    self.cells.update([cell])
                    if len(self.cells) > self.cell_amount:
                        # a new cell successfully update
                        self.cell_amount += 1
                        self.width += (rib - leb)
                        #self.check_width()
                        #print("the cell is filled in {}".format(self.centre_y_coordinates))
                        added = True
                    else:
                        pass
                        #print('the cell already in the row')
                elif results == [True,False,False]:
                    # wait when i happy
                    # merge
                    # width issue
                    pass
                elif results == [False,True,False]:
                    # still not happy
                    pass
                elif results == [False,False,True]:
                    pass
                else:
                    print('where is my baseball bat? oh wait there is a machine gun')
            else:
                # should not in this row
                print('the cell not belonging to this row')
        return added

    def sort_row(self):
        def sortway(c):
            return c.identification['centre_coordinates'][0]
        self.cells_list = sorted(self.cells,key=sortway)
        return self.cells_list

    def get_cell_list(self):
        return self.sort_row()

    def is_empty(self):
        return self.cell_amount == 0

    def isfullrow(self):
        return approxiamtion(self.width,self.max_width)




class Column:


    pass


class Table:
    """this will be a pandas dataframe with unique table id , maybe by page id and other parameters
    potentially  a table should provide {
    table name , table field ( column names) , row index and so on
    another issue here is for some row and columns with for instance visually same but not share precise cooridinates
    it can be reorganised here
    and last a table better provide a write into function to write a table to a excel workbook with sheet name
    equals table names (maybe the header or few words of left up text box)"""
    pagetables = {}

    def __init__(self):
        self.root = None
        pass


