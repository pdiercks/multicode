Mesh.RecombinationAlgorithm = 0;
Mesh.Algorithm = 1;
Mesh.Algorithm3D = 1;
Mesh.Optimize = 2;
Mesh.Smoothing = 2;

DefineConstant[ num_points = 21.0 ];
meshMatrix = 20.0 / (num_points - 1.0);
meshAggreg = 0.7 * meshMatrix;

Function Aggregate
  // circle at (xC, yC) and radius "R"
  p1 = newp; Point(p1) = {xC,  yC,  0, meshAggreg};
  p2 = newp; Point(p2) = {xC+R,yC,  0, meshAggreg};
  p3 = newp; Point(p3) = {xC-R,yC,  0, meshAggreg};
  c1 = newreg; Circle(c1) = {p2,p1,p3};
  c2 = newreg; Circle(c2) = {p3,p1,p2};
  l1 = newreg; Line Loop(l1) = {c1,c2};
  s1 = newreg; Plane Surface(s1) = {l1};
  theOuterInterface[i] = l1;
  theAggregates[i] = s1;
Return

i=0; xC=8.124435628293494; yC=16.250990871336494; R=2.2;
Call Aggregate;
i=1; xC=3.104265948507514; yC=3.072789217500327; R=1.9;
Call Aggregate;
i=2; xC=16.205618753300654; yC=16.37885427346391; R=1.5;
Call Aggregate;
i=3; xC=3.8648187874608415; yC=10.576264325380615; R=2.1;
Call Aggregate;
i=4; xC=12.807996595076595; yC=12.686751823841977; R=1.7;
Call Aggregate;
i=5; xC=16.23956045449863; yC=7.686853577410513; R=1.9;
Call Aggregate;
i=6; xC=7.9915552082180366; yC=6.689767983295199; R=2.0;
Call Aggregate;
i=7; xC=12.561194629950934; yC=2.7353694913178512; R=1.6;
Call Aggregate;

Function Matrix
  // rectangle from (xS, yS) to (xE, yE)
  // points:
  p0 = newp; Point(p0) = {xS, yS, 0, meshMatrix};
  p1 = newp; Point(p1) = {xE, yS, 0, meshMatrix};
  p2 = newp; Point(p2) = {xE, yE, 0, meshMatrix};
  p3 = newp; Point(p3) = {xS, yE, 0, meshMatrix};
  // lines
  l0 = newreg; Line(l0) = {p0, p1};
  l1 = newreg; Line(l1) = {p1, p2};
  l2 = newreg; Line(l2) = {p2, p3};
  l3 = newreg; Line(l3) = {p3, p0};
  theBox = newreg; Line Loop(theBox) = { l0, l1, l2, l3};
  Transfinite Line {l0, l1, l2, l3} = num_points;
Return

xS=0.0; yS=0.0; xE=20.0; yE=20.0;
Call Matrix;

theMatrix = newv;
Surface(theMatrix) = {theBox, theOuterInterface[]};
Physical Surface("Matrix")= {theMatrix};
Physical Surface("Aggregates")= {theAggregates[]};
