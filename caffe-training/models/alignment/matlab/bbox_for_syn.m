clear, close all
imageDir = '/home/ljyang/work/alignment/3d_models/car_models/obj_merge3/';
load('model_3d_info');
% for i=1:length(temp)
% fd_name{i} = temp(i).name;
% if ~strcmp(model_name_3d{i,1},fd_name{i})
%     error('fd_name dismatch');
% end
% end
len = length(model_3d_info);
im_per_m = 1200;
bbox_syn=zeros(len,im_per_m,4);
for i=1:len
    i
    for im_id = 1:im_per_m
        im_name = [imageDir,model_3d_info(i).model_name,'/',...
            model_3d_info(i).model_name,'_',num2str(im_id-1),'.png'];
        im = 1-double(imread(im_name))/255;%reverse
        h_hist = sum(im,1);
        v_hist = sum(im,2);
        h_bb = find(h_hist>0);
        v_bb = find(v_hist>0);
%         imshow(im);
%         line([h_bb(1),h_bb(end),h_bb(end),h_bb(1),h_bb(1)],...
%             [v_bb(1),v_bb(1),v_bb(end),v_bb(end),v_bb(1)]);
%         pause;
        bbox_syn(i,im_id,1) = h_bb(1);
        bbox_syn(i,im_id,3) = h_bb(end);
        bbox_syn(i,im_id,2) = v_bb(1);
        bbox_syn(i,im_id,4) = v_bb(end);
    end
end
save bbox_syn5 bbox_syn