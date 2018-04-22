clear, close all
pred_h =load('/home/ljyang/work/data/matlab/alexnet_domain3_view_h.mat');
pred_v =load('/home/ljyang/work/data/matlab/alexnet_domain3_view_v.mat');
sp = 6833;
pred_h = pred_h.feats(1:sp,:);
pred_v = pred_v.feats(1:sp,:);
[~,pred_h_id] = max(pred_h,[],2);
[~,pred_v_id] = max(pred_v,[],2);
f=fopen('../lists/transfer_test');
list = textscan(f,'%s %d %d %d %d %d');

path = list{1};
for i=1000:sp
    im = imread(path{i});
    imshow(im);
    pred_h_cur = (pred_h_id(i)-1)*6;
    pred_v_cur = (pred_v_id(i)-1)*6;
    title(sprintf('azumith %d altitude %d',pred_h_cur,pred_v_cur));
    pause;
end
    