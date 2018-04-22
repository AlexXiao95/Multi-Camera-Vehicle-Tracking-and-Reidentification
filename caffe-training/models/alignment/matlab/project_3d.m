%This function generally follow the PASCAL 3D+ function of project_3d
%not the different annotation of azimuth of our protocle and theirs
function x = project_3d(x3d,pose,focal,bbox)

    pose = pose*pi/180;
    a = 2*pi- pose(1);%diffrent from PASCAL
    e = pose(2);
    theta = pose(3);
    
    bb_l = bbox(4)-bbox(2);
    bb_w = bbox(3) - bbox(1);
    bb_mid = [bbox(1)+bbox(3),bbox(4)+bbox(2)]/2;
    d = 1+focal;
    %camera center
    
    C = zeros(3,1);
    C(1) =  d *cos(e)*sin(a);
    C(2) =  -d*cos(e)*cos(a);
    C(3) = d*sin(e);
%     C = -C;
%     C = [0,0,0]';
    a = -a;
    e = -(pi/2-e);

    % rotation matrix
    Rz = [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];   %rotate by a
    Rx = [1 0 0; 0 cos(e) -sin(e); 0 sin(e) cos(e)];   %rotate by e
    R = Rx*Rz;
    M =1;
    P = [M*focal 0 0; 0 M*focal 0; 0 0 -1] * [R -R*C];
    
    % project
    x = P*[x3d ones(size(x3d,1), 1)]';
    x(1,:) = x(1,:) ./ x(3,:);
    x(2,:) = x(2,:) ./ x(3,:);
    x = x(1:2,:);
    % rotation matrix 2D
    R2d = [cos(theta) -sin(theta); sin(theta) cos(theta)];
    x = (R2d * x)';
    x(:,2) = -1 * x(:,2);%due to difference of plot and image coordinates
    %fit to bbox
    min_x = min(x(:,1));
    max_x = max(x(:,1));
    min_y = min(x(:,2));
    max_y = max(x(:,2));
    mid  =[min_x + max_x, min_y + max_y]/2;
    area = (max_x-min_x)*(max_y-min_y);
    %compare ratio

    bb_area = bb_w*bb_l;
    scale = sqrt(bb_area/area);
    x = (x-repmat(mid,size(x,1),1)) * scale + repmat(bb_mid,size(x,1),1);