function [x3d,faces] = get_polar_mesh(polar)
    [theta_n,phi_n] = size(polar);
    theta = ([0:theta_n-1]+0.5)/theta_n * 2 * pi;
    phi = asin(2*([0:phi_n-1]+0.5)/phi_n-1);
%     phi(end) = pi/2;%fill the hole on the top
    theta_exp = repmat(theta',1,phi_n);
    phi_exp = repmat(phi,theta_n,1);
    x = polar .* sin(theta_exp) .* cos(phi_exp); 
    y = -polar .* cos(theta_exp) .* cos(phi_exp);
    z = polar .* sin(phi_exp);
    x3d = [x(:),y(:),z(:)];
    %4-polygon mesh
    faces = zeros(theta_n*(phi_n-1),4);
  
    for j = 1:phi_n -1 
        for i=1: theta_n
            if i<theta_n
                subs = [i,j;i+1,j;i+1,j+1;i,j+1];
                
            else
                subs = [i,j;1,j;1,j+1;i,j+1];
            end
            idx = sub2ind([theta_n,phi_n],subs(:,1),subs(:,2));
            faces((j-1)*theta_n+i,:) = idx';
        end
    end
        
end