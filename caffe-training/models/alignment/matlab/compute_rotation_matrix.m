function R = compute_rotation_matrix(pose)

pose = pose*pi/180;
a=pose(1);
e=pose(2);
r=pose(3);
Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];   %rotate by a
Rx = [1 0 0; 0 cos(e) -sin(e); 0 sin(e) cos(e)];   %rotate by e
Ry = [cos(r) 0 sin(r); 0 1 0; -sin(r) 0 cos(r)];
R = Ry*Rx*Rz;