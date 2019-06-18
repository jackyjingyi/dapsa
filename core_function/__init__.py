from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice,PDFTextDevice
from pdfminer.layout import LAParams,LTChar,LTTextLine,LTText,LTTextBox,LTLine,LTRect,LTImage,LTTextBoxHorizontal
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfpage import PDFPage
from pdfminer.utils import Plane
import pdfminer
import re
import copy
import openpyxl
import pandas as pd
import time
import datetime
import nltk
import os

LINE_MARGIN = 0.01
CHAR_MARGIN = 0.1


class AnalyzePDF:
    page_dict = {}
    page_warehouse = {}
    class page_mining:

        def __init__(self, page):
            self._page_info = None
            self.pageid = page.pageid
            self._tables = None

        def __repr__(self):
            return '<page number : %s >' % (AnalyzePDF.page_dict[self.pageid])

        @property
        def tables(self):
            return self._tables


    def __init__(self, fp, target):
        self.fp = fp
        self.target_pages = target
        self.parser = PDFParser(self.fp)
        self.document = PDFDocument(self.parser)
        if not self.document.is_extractable:
            raise PDFTextExtractionNotAllowed
        self.rsrcmgr = PDFResourceManager()
        self.device = PDFDevice(self.rsrcmgr)
        self.textdevice = PDFTextDevice(self.device)
        self.laparams = LAParams(line_margin=LINE_MARGIN,char_margin=CHAR_MARGIN)
        self.pageagg = PDFPageAggregator(rsrcmgr=self.rsrcmgr, laparams=self.laparams)
        self.interpreter = PDFPageInterpreter(rsrcmgr=self.rsrcmgr, device= self.pageagg)
        for pageid,page in enumerate(PDFPage.get_pages(fp),start=1):
            AnalyzePDF.page_dict[pageid] = page.pageid
        self._page_info = None

    @property
    def page_info(self):
        return self._page_info

    @page_info.setter
    def page_info(self, page,):
        textbox_list = []
        self.interpreter.process_page(page)
        layout = self.pageagg.get_result()
        Point.set_page_points(pageid=page.pageid)
        point_count = 0
        mid_count = 0
        length = [999,0]
        for element in layout:
            if isinstance(element, LTChar):
                print(element.fontname)
            elif isinstance(element, pdfminer.layout.LTCurve):
                print(element)
                length[0] = min(element.bbox[0], length[0])
                length[1] = max(element.bbox[3], length[1])
                point_left_bottom = Point(x=element.bbox[0], y=element.bbox[1], pageid=page.pageid, id=point_count)
                point_right_top = Point(element.bbox[2], element.bbox[3], pageid=page.pageid, id=point_count + 1)
                middle_point = Point(x=(element.bbox[0] + element.bbox[2]) / 2,
                                     y=(element.bbox[1] + element.bbox[3]) / 2, pageid=page.pageid, id=mid_count,
                                     ischild=True)
                point_left_bottom.add_to_pagepoints(pageid=page.pageid)
                point_right_top.add_to_pagepoints(pageid=page.pageid)
                point_left_bottom.assign_child(middle_point)
                point_right_top.assign_child(middle_point)
                middle_point.add_to_pagepoints(pageid=page.pageid)
                # print(point_left_bottom.__repr__())
                # print(point_right_top.__repr__())
                # print(point_left_bottom.children)
                # print(point_right_top.children)
                # print(middle_point.parents)
                point_count += 2
                mid_count += 1
            elif isinstance(element, LTTextBox):
                print(element)
                textbox_list.append(element)
            elif isinstance(element, pdfminer.layout.LTLine):
                print('this is lttline {}'.format(element))
        self._page_info = (Point.pagePoints[page.pageid], textbox_list)
        return

    def warehouse(self):
        for page in PDFPage.get_pages(self.fp):
            for tp in self.target_pages:
                if page.pageid == AnalyzePDF.page_dict[tp]:
                    AnalyzePDF.page_warehouse[(tp, page.pageid)] = self.page_info

    def extratacttable(self, pageid, textbox, top_limit = 800):
        print('the len is {}'.format(len(Point.pagePoints[pageid])))
        clean_points = Point.clean_points(pageid=pageid, textboxlist=textbox)
        Cluster.group_cluster(pageid=pageid, cleaned_points=clean_points)
        Cluster.sortpageCluster(pageid=pageid)
        Cluster.assignid(pageid=pageid)
        for c in Cluster.pageSortedClusters[pageid]:
            c.get_middle()
            print(c.id, c.s1_matrix, c.key)
        print(Cluster.pageSortedClusters)
        # checking point 1
        Cluster.locat_roots(pageid=pageid)
        Cluster.do_cell(pageid=pageid)
        for c in list(Cell.pageCells[pageid].values()):
            for j in c:
                j()
                print((j.left_boundary, j.lower_boundary, j.right_boundary, j.upper_boundary, j.width, j.height,
                       j.centre))
        Cell.AssignText(pageid=pageid, textlist=textbox)
        Cell.check_text(pageid=pageid)
        return










