
import math

from OCC.Core.GeomFill import GeomFill_IsFrenet
from OCC.Core.gp import (gp_Pnt, gp_OX, gp_Vec, gp_Trsf, gp_DZ, gp_Ax2, gp_Ax3,
                         gp_Pnt2d, gp_Dir2d, gp_Ax2d, gp_Pln, gp_Circ, gp_Dir)
from OCC.Core import gp
from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeSegment
from OCC.Core.GCE2d import GCE2d_MakeSegment
from OCC.Core.Geom import Geom_CylindricalSurface
from OCC.Core.Geom2d import Geom2d_Ellipse, Geom2d_TrimmedCurve
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire,
                                     BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeBox
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeThickSolid, BRepOffsetAPI_ThruSections, BRepOffsetAPI_MakePipe
from OCC.Core.BRepLib import breplib
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.TopoDS import topods, TopoDS_Compound, TopoDS_Face
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopTools import TopTools_ListOfShape
from OCC.Extend.DataExchange import write_iges_file,write_step_file,write_stl_file

# Set up the curves for the threads on the bottle's neck
from OCC.Extend.TopologyUtils import TopologyExplorer

#from examples.core_webgl_threejs_helix import aLine2d

from OCC.Core.gp import gp_Pnt2d, gp_XOY, gp_Lin2d, gp_Ax3, gp_Dir2d
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCC.Core.Geom import Geom_CylindricalSurface
from OCC.Core.GCE2d import GCE2d_MakeSegment
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder

def makeHelix(Helix_r=12,a=2.0,b=10,c=500,d=200):
    aCylinder = Geom_CylindricalSurface(gp_Ax3(gp_XOY()), Helix_r)#生成圆柱面（螺旋形b半径）
    aLine2d = gp_Lin2d(gp_Pnt2d(0.0, 0.0), gp_Dir2d(a, b))#斜率COS(X)=a/b
    aSegment = GCE2d_MakeSegment(aLine2d, 0.0, math.pi * c)
    helix_edge = BRepBuilderAPI_MakeEdge(aSegment.Value(), aCylinder, 0.0, d * math.pi).Edge()#d*pi为螺线线长度 d为圈数
    return helix_edge

#------------------------------------------------------------------------------------
#生成钻头
def Create_drill(D=20,Helix_angle=15,Lead=20,Helix_length=80,Total_length=100,handle_D=20):
    pass
    aHelixEdge = makeHelix()
    breplib.BuildCurve3d(aHelixEdge)  # 将二维曲线转换为3D
    # 获取螺旋线两个端点
    first_point_v = topexp.FirstVertex(aHelixEdge, True)
    end_point_v = topexp.LastVertex(aHelixEdge, True)
    first_point = BRep_Tool.Pnt(first_point_v)
    end_point = BRep_Tool.Pnt(end_point_v)
    first_point_cord = [first_point.X(), first_point.Y(), first_point.Z()]  # 起点坐标
    end_point_cord = [end_point.X(), end_point.Y(), end_point.Z()]  # 终点坐标
    # 绘制圆柱
    my_cylinder = BRepPrimAPI_MakeCylinder(12.5, end_point.Z()).Shape()
    # 扫掠生成螺旋线体
    circel_pln = gp_Ax2(end_point, gp_Dir(0, 1, 0), gp_Dir(0, 0, 1))  # 草绘平面
    pipe_circel = gp_Circ(circel_pln, 5.0)  # 截面
    Ec = BRepBuilderAPI_MakeEdge(pipe_circel).Edge()  # 截面
    Wc = BRepBuilderAPI_MakeWire(Ec).Wire()  # 截面
    F = BRepBuilderAPI_MakeFace(Wc, True)  # 截面
    W = BRepBuilderAPI_MakeWire(aHelixEdge)  # 引导线
    S = BRepOffsetAPI_MakePipe(W.Wire(), F.Shape(), GeomFill_IsFrenet)  # 第一个参数 轨迹线，第二个参数是轮廓线
    Cut_result = BRepAlgoAPI_Cut(my_cylinder, S.Shape()).Shape()
    return  Cut_result,aHelixEdge






if __name__== "__main__":
    from OCC.Display.SimpleGui import init_display
    display, start_display, add_menu, add_function_to_menu = init_display()
    #display.DisplayShape(aHelixEdge, update=True)

    #display.DisplayShape(my_cylinder, update=True)
    #display.DisplayShape(F.Shape(), update=True)
    #display.DisplayShape(W.Shape(), update=True)
    display.DisplayShape(S.Shape(), update=True)
    drill=Create_drill()
    display.DisplayShape(drill[1], update=True)
    start_display()
