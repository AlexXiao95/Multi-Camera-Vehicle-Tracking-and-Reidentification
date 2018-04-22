%This is the viewpoint error function from:
%Tulsiani, S., & Malik, J. (2014). Viewpoints and keypoints. arXiv preprint arXiv:1411.6067.
function dist =compute_R_distance(R1,R2)
R = R1'*R2;
theta = acos((trace(R)-1)/2);
if theta==0
    logR = zeros(3,3);
else
    logR = theta/2/sin(theta)*(R-R');
end
temp = logR.^2;
dist = sqrt(sum(temp(:))/2);
