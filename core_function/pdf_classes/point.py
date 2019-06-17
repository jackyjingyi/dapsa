from core_function.pdf_classes.para import PTS,approxiamtion,inside
import numpy as np

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

    def assign_child(self,child):
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
        def define_direction(x, y, child, linewidth):
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
            pipid = define_direction(self.x,self.y,child,linewidth=self.line_width*(1+PTS))
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
    def set_page_points(cls,pageid):
        cls.pagePoints[pageid] = []
        cls.pageMiddlePoints[pageid] =[]

    def add_to_pagepoints(self,pageid):
        if not self.ischild:
            Point.pagePoints[pageid].append(self)
        else:
            if not all([id==-999 for id in self.parents]):
                Point.pageMiddlePoints[pageid].append(self)

    @classmethod
    def clean_points(cls,pageid,textboxlist):
        def check_points_valiadation(point_list, textboxlist):
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
        cls.pagePoints[pageid] = check_points_valiadation(cls.pagePoints[pageid],textboxlist=textboxlist)
        return cls.pagePoints[pageid]

