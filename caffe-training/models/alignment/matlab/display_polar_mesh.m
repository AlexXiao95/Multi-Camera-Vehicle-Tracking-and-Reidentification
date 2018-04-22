function display_polar_mesh(polar)
    [theta_n,phi_n] = size(polar);
    theta = ([0:theta_n-1]+0.5)/theta_n * 2 * pi;
    phi = asin(2*([0:phi_n-1]+0.5)/phi_n-1);
    phi(end) = pi/2;%fill the hole on the top
    theta_exp = repmat(theta',1,phi_n);
    phi_exp = repmat(phi,theta_n,1);
    x = polar .* sin(theta_exp) .* cos(phi_exp); 
    y = polar .* cos(theta_exp) .* cos(phi_exp);
    z = polar .* sin(phi_exp);
    %complete the strip on the front side
    x = [x;x(1,:)];
    y = [y;y(1,:)];
    z = [z;z(1,:)];
    mesh(x,y,z);
    axis equal