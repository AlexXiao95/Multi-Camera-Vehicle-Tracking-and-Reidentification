clear,close all
imDir ='/home/ljyang/work/data/real_and_syn/real/';
temp = dir(imDir);
temp = temp(3:end);
im_s = zeros(256,256,3);
c=0;
for i=1:length(temp)
    im_list = dir([imDir,temp(i).name,'/*.jpg']);
    for j=1:length(im_list)
        im = imread([imDir,temp(i).name,'/',im_list(j).name]);
        im =imresize(im,[256,256]);
        im_s = im_s +double(im)/255;
        c=c+1;
    end
end
im_s = im_s/c;
figure;
imshow(im_s);