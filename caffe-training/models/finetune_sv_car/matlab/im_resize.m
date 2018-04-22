clear,close all,clc
datDir='/home/ljyang/work/data/models_org_281/release';
savDir='/home/ljyang/work/data/models_org_281/resize2';
temp = dir(datDir);
temp = temp(3:end);
n =length(temp);
for i=1:n
 im_list = dir([datDir,'/',temp(i).name,'/*.jpg']);
 if ~exist([savDir,'/',temp(i).name],'dir')
     mkdir([savDir,'/',temp(i).name]);
 end
 for j=1:length(im_list)
    im = imread([datDir,'/',temp(i).name,'/',im_list(j).name]);
    im = imresize(im,[256,256]);
%     imshow(im);
%     title(num2str(m));
%     pause;
      imwrite(im,[savDir,'/',temp(i).name,'/',im_list(j).name]);
 end
end
