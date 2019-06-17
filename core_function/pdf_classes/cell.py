from core_function.pdf_classes.para import PTS,approxiamtion,inside
from .cluster import Cluster


class Cell:
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



