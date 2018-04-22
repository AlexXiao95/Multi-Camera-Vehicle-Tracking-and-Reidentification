clear,close all
load model_3d_info
load test_images_3d
len = length(model_3d_info);
imDir ='/home/ljyang/work/data/CompCars/image/';
labelDir = '/home/ljyang/work/data/CompCars/label/';
for i=1:len
    close all
    p = test_images(i).path;
    model_3d_id = test_images(i).model_3d_id;
    shape_polar = model_3d_info(model_3d_id).shape_polar;
    [theta_n,phi_n] = size(shape_polar);
    [x3d,faces] = get_polar_mesh(shape_polar);
%     mesh(mx,my,mz);
%     axis equal
    
    pose = test_images(i).pose;
%     pose

    focal = test_images(i).focal;
%     focal
    im = imread([imDir,p]);
    label = textread([labelDir,p(1:end-3),'txt'],'%d');
    bbox = label(3:6);
    x2d = project_3d(x3d,pose,focal,bbox);
    figure;
    imshow(im);
    hold on;
%     figure;
%     scatter(x2d(:,1),x2d(:,2),'filled');
    patch('vertices', x2d, 'faces', faces, ...
            'FaceColor', 'blue', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

    axis off;
    hold off;
    pause;
end
