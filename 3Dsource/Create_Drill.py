
import math

from OCC.Core.gp import (
    gp_Pnt,
    gp_OX,
    gp_Vec,
    gp_Trsf,
    gp_DZ,
    gp_Lin2d,
    gp_Dir,
    gp_Ax1,
    gp_Ax2,
    gp_Ax3,
    gp_Pnt2d,
    gp_Dir2d,
    gp_Ax2d,
    gp_Pln,
    gp_XOY,
)
from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeSegment
from OCC.Core.GC import GC_MakeCircle
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.GCE2d import GCE2d_MakeSegment
from OCC.Core.Geom import Geom_CylindricalSurface
from OCC.Core.Geom2d import Geom2d_Ellipse, Geom2d_TrimmedCurve
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Transform,
    BRepBuilderAPI_Copy,
)
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeCylinder
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.BRepOffsetAPI import (
    BRepOffsetAPI_MakeThickSolid,
    BRepOffsetAPI_ThruSections,
)
from OCC.Core.BRepLib import breplib
from OCC.Core.BRep import BRep_Builder
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.Geom import Geom_Plane
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.TopoDS import topods, TopoDS_Compound, TopoDS_Face
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_ShapeEnum
from OCC.Core.TopTools import TopTools_ListOfShape

from OCC.Core.Geom2d import Geom2d_Line
from OCC.Core.Geom import Geom_RectangularTrimmedSurface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Geom import Geom_TrimmedCurve
from OCC.Core.Geom import Geom_Circle
from OCC.Core.TopoDS import topods_Vertex, topods_Edge
from OCC.Core.BOPAlgo import BOPAlgo_Builder, BOPAlgo_BOP
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common


def Create_Drill_fun(R=4,Rr=3.5,b=1,H=80,a=3,sigma=118):
    #R = 4.  # outer radius
    a = a * math.pi
    D = 2 * R  # diameter
    #Rr = 3.5  # chisel radius (outer radius minus body clearance)
    #b = 1.  # web thickness (approximate)
    d = b / 2

    #H = 80.  # height of the spiral part
    #a = 3. * math.pi  # total angle of spiral rotation
    # ss
    #sigma = 118  # point angle, in degrees

    # Create section profile by sequence of Boolean operations
    # on simple planar objects
    # polyline rectangle1 d -R 0  R -R 0  -d R 0  -R R 0  d -R 0
    aPnt1 = gp_Pnt(d, -R, 0)
    aPnt2 = gp_Pnt(R, -R, 0)
    aPnt3 = gp_Pnt(-d, R, 0)
    aPnt4 = gp_Pnt(-R, R, 0)
    aPnt5 = gp_Pnt(d, -R, 0)

    rectangle1 = BRepBuilderAPI_MakePolygon()
    rectangle1.Add(aPnt1)
    rectangle1.Add(aPnt2)
    rectangle1.Add(aPnt3)
    rectangle1.Add(aPnt4)
    rectangle1.Add(aPnt5)
    rectangle1.Close()

    Circle1 = GC_MakeCircle(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), R).Value()

    Circle1 = BRepBuilderAPI_MakeEdge(Circle1)

    Circle1 = BRepBuilderAPI_MakeWire(Circle1.Edge())

    Circle2 = GC_MakeCircle(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), Rr).Value()

    Circle2 = BRepBuilderAPI_MakeEdge(Circle2)

    Circle2 = BRepBuilderAPI_MakeWire(Circle2.Edge())

    aPnt00 = gp_Pnt(0, 0, 0)
    aDir = gp_Dir(0, 0, 1)
    aDir2 = gp_Dir(1, 0, 0)
    loc = gp_Ax3(aPnt00, aDir, aDir2)

    p0 = Geom_Plane(loc)
    rectangle1 = BRepBuilderAPI_MakeFace(p0, rectangle1.Wire()).Face()
    Circle1 = BRepBuilderAPI_MakeFace(p0, Circle1.Wire()).Face()
    Circle2 = BRepBuilderAPI_MakeFace(p0, Circle2.Wire()).Face()

    sec = BRepAlgoAPI_Common(rectangle1, Circle1).Shape()
    sec = BRepAlgoAPI_Fuse(sec, Circle2).Shape()
    sec = ShapeUpgrade_UnifySameDomain(sec)
    sec.Build()

    lip = BRepBuilderAPI_MakePolygon(gp_Pnt(d, -d / 2, 0),
                                     gp_Pnt(d, -R, -R / math.tan(sigma / 2 * math.pi / 180))).Wire()

    sp = BRepBuilderAPI_MakePolygon(gp_Pnt(0, 0, 0), gp_Pnt(0, 0, H)).Wire()
    # cylinder cc 0 0 0  0 0 1  0 -4 0  4
    cc_gp_Ax3 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(0, -4, 0))
    cc = Geom_CylindricalSurface(cc_gp_Ax3, 4)

    ll = Geom2d_Line(gp_Pnt2d(0, 0), gp_Dir2d(a, 80))
    ll = Geom2d_TrimmedCurve(ll, 0, math.sqrt(a * a + H * H))

    v1 = BRepBuilderAPI_MakeVertex(gp_Pnt(0, -R, 0))
    v2 = BRepBuilderAPI_MakeVertex(gp_Pnt(0, -R, H))

    v2_tsf = gp_Trsf()
    v2_tsf_ax1 = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0),
                        gp_Dir(0.0, 0.0, 1.0))
    v2_tsf.SetRotation(v2_tsf_ax1, 180. * a / math.pi * math.pi / 180)  # 此处要*math.pi/180，因为要角度转弧度

    v2 = BRepBuilderAPI_Transform(v2.Shape(), v2_tsf)
    ee = BRepBuilderAPI_MakeEdge(ll, cc, v1.Shape(), v2.Shape())
    gg = BRepBuilderAPI_MakeWire(ee.Edge())

    spiral = BRepOffsetAPI_MakePipeShell(sp)
    spiral.SetMode(gg.Shape(), False, 0)
    #print('lip：', lip)
    spiral.Add(lip)
    spiral.Build()
    spiral.MakeSolid()
    f0 = BRepBuilderAPI_MakeFace(p0, -R, R, -R, R, 1e-7).Face()
    sflute = BRepAlgoAPI_Section(spiral.Shape(), f0, False)
    sflute.Approximation(True)
    sflute.ComputePCurveOn1(True)
    sflute.ComputePCurveOn2(True)
    sflute.SetFuzzyValue(1e-7)
    sflute.SetRunParallel(False)
    sflute.SetNonDestructive(False)
    sflute.SetGlue(0)
    sflute.SetUseOBB(True)
    sflute.Build()

    sflute_1 = TopExp_Explorer(sflute.Shape(), TopAbs_ShapeEnum.TopAbs_EDGE, TopAbs_ShapeEnum.TopAbs_SHAPE)

    cflute = BRep_Tool().Curve(sflute_1.Value())
    sflute_1_1 = cflute[0].Copy()

    cflute = Geom_TrimmedCurve(cflute[0], cflute[1], cflute[2])

    c1_pnt = gp_Pnt()
    c2_pnt = gp_Pnt()
    cflute.D0(0., c1_pnt)
    print(c1_pnt.X(), c1_pnt.Y())
    cflute.D0(1., c2_pnt)
    x0 = c1_pnt.X()
    y0 = c1_pnt.Y()
    z0 = c1_pnt.Z()
    x1 = c1_pnt.X()
    y1 = c1_pnt.Y()
    z1 = c1_pnt.Z()

    vf0 = BRepBuilderAPI_MakeVertex(c1_pnt)
    vf1 = BRepBuilderAPI_MakeVertex(c2_pnt)

    cround = Geom_Circle(gp_Ax2(gp_Pnt(x0 + Rr / 2, y0, 0), gp_Dir(0, 0, 1)), Rr / 2)

    vf3 = BRepBuilderAPI_MakeVertex(gp_Pnt(x0 + Rr, y0, 0))

    sflute_2 = BRepBuilderAPI_MakeEdge(cround, vf3.Shape(), vf0.Shape())
    vf2 = BRepBuilderAPI_MakeVertex(gp_Pnt(R, -R, 0))

    sflute_3 = BRepBuilderAPI_MakeEdge(vf3.Shape(), vf2.Shape())

    sflute_4 = BRepBuilderAPI_MakeEdge(vf2.Shape(), vf1.Shape())

    sflute_1_1 = BRepBuilderAPI_MakeEdge(cflute)

    w2 = BRepBuilderAPI_MakeWire(sflute_1_1.Shape(), sflute_2.Shape(), sflute_3.Shape(), sflute_4.Shape())

    flute = BRepBuilderAPI_MakeFace(p0, w2.Shape())

    sec = BRepAlgoAPI_Cut(sec.Shape(), flute.Shape()).Shape()

    flute_tsf = gp_Trsf()
    flute_tsf_ax1 = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0),
                           gp_Dir(0.0, 0.0, 1.0))
    flute_tsf.SetRotation(v2_tsf_ax1, 180 * math.pi / 180)  # 此处要*math.pi/180，因为要角度转弧度
    flute = BRepBuilderAPI_Transform(flute.Shape(), flute_tsf)
    # # bcut sec sec flute
    sec = BRepAlgoAPI_Cut(sec, flute.Shape()).Shape()

    print("Sweeping the profile...")

    base = BRepOffsetAPI_MakePipeShell(sp)
    base.SetMode(gg.Shape(), False, 0)

    sec_1 = TopExp_Explorer(sec, TopAbs_ShapeEnum.TopAbs_WIRE, TopAbs_ShapeEnum.TopAbs_SHAPE)

    sec_1 = BRepBuilderAPI_MakeWire(sec_1.Value())

    base.Add(sec_1.Shape())
    base.Build()
    base.MakeSolid()

    #print("Sharpening...")

    theta = a * R / H * math.sin((90 - sigma / 2) * math.pi / 180)
    loc_ax3 = gp_Ax3(gp_Pnt(d, 1.9 * D, H + 1.9 * D / math.tan(math.pi / 180. * sigma / 2.)), gp_Dir(0, -1, -1))
    ax3 = Geom_Plane(loc_ax3)
    sh1 = BRepPrimAPI_MakeCone(ax3.Pln().Position().Ax2(), 0, 100 * math.sin((sigma - 90) / 2 * math.pi / 180.), 100)
    sh1_tsf = gp_Trsf()
    sh1_tsf_ax1 = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0),
                         gp_Dir(0.0, 0.0, 1.0))
    sh1_tsf.SetRotation(sh1_tsf_ax1, -theta * 180 / math.pi * math.pi / 180)
    sh1 = BRepBuilderAPI_Transform(sh1.Shape(), sh1_tsf)

    sh2 = BRepBuilderAPI_Copy(sh1.Shape(), True, False)

    sh2_tsf = gp_Trsf()
    sh2_tsf_ax1 = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0),
                         gp_Dir(0.0, 0.0, 1.0))
    sh2_tsf.SetRotation(sh2_tsf_ax1, 180 * math.pi / 180)

    sh2 = BRepBuilderAPI_Transform(sh2.Shape(), sh2_tsf)

    sh = BRepPrimAPI_MakeBox(gp_Pnt(-D / 2, -D / 2, 72), D, D, 20).Shape()
    qq = BRepAlgoAPI_Common(sh1.Shape(), sh2.Shape()).Shape()
    sharpener = BRepAlgoAPI_Cut(sh, qq).Shape()
    body = BRepAlgoAPI_Cut(base.Shape(), sharpener).Shape()
    print("Making a shank...")
    loc_pl2 = gp_Ax3(gp_Pnt(0, 0, -40), gp_Dir(0, 0, 1))
    pl2 = Geom_Plane(loc_pl2)
    shank = BRepPrimAPI_MakeCylinder(pl2.Pln().Position().Ax2(), 4, 40)
    transit = BRepPrimAPI_MakeCone(R, 0, R)
    loc_pl3 = gp_Ax3(gp_Pnt(0, 0, -40), gp_Dir(0, 0, -0.5))
    pl3 = Geom_Plane(loc_pl3)
    tail = BRepPrimAPI_MakeCone(pl3.Pln().Position().Ax2(), R, 0, 0.5)
    shank = BRepAlgoAPI_Fuse(shank.Shape(), tail.Shape()).Shape()
    shank = BRepAlgoAPI_Fuse(shank, transit.Shape()).Shape()
    drill = BRepAlgoAPI_Fuse(body, shank).Shape()
    return drill


'''
if __name__ == "__main__":
    from OCC.Display.SimpleGui import init_display
    display, start_display, add_menu, add_function_to_menu = init_display()
    drill=Create_Drill_fun()
    display.DisplayShape(drill, update=True)
    start_display()
'''
