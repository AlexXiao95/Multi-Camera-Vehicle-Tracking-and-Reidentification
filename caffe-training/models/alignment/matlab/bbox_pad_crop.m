function crop = bbox_pad_crop(im,bbox,pad)
[l,w,ch]=size(im);
 if isa(im,'uint8')
     im = double(im)/255;
 end

    %keep aspect ratio
    pad_l = (bbox(4)-bbox(2))*(1+2*pad);
    pad_w = (bbox(3)-bbox(1))*(1+2*pad);
    max_s = max(pad_l,pad_w);
    half_s = floor(max_s/2);
    max_s = half_s*2;
    mid_l = floor((bbox(4)+bbox(2))/2);
    mid_w = floor((bbox(3)+bbox(1))/2);

    pad(1) = max(0, - mid_w + half_s);
    bbox_new(1) = mid_w-half_s + pad(1)+1; 
%   
    pad(2) = max(0, - mid_l + half_s);
    bbox_new(2) = mid_l-half_s+pad(2)+1;
%  
    pad(3) = max(0,mid_w+half_s-w);
    pad(4) = max(0,mid_l+half_s-l);
    bbox_new(3) = min(w,mid_w + half_s);
    bbox_new(4) = min(l,mid_l + half_s);
    m = mean(im(:));
    crop = ones(max_s,max_s,ch)*m;
    crop(pad(2)+1:end-pad(4),pad(1)+1:end-pad(3),:) = im(bbox_new(2):bbox_new(4),...
        bbox_new(1):bbox_new(3),:);
end